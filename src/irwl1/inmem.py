import math
from copy import deepcopy
import torch
import irwl1.config as config
from irwl1.utils import _init_wandb_run, validate_in_memory, calculate_WL1_norm, calculate_L1_norm, calculate_thresholded_sparsity, calculate_real_sparsity, _advance_lambda, _advance_epsilon, _regularization_update_batches, L1_penalty_update, _regularized_sparsity_stop
def train_in_memory(model, train_image_tensor, train_label_tensor, val_image_tensor, val_label_tensor, optimizer=None,
                     is_new_run=True, run=None, run_name="default", apply_reg=False, **kwargs):
    if apply_reg:
        return train_in_memory_regularized(
            model,
            train_image_tensor,
            train_label_tensor,
            val_image_tensor,
            val_label_tensor,
            optimizer=optimizer,
            is_new_run=is_new_run,
            run=run,
            run_name=run_name,
            **kwargs,
        )

    return train_in_memory_plain(
        model,
        train_image_tensor,
        train_label_tensor,
        val_image_tensor,
        val_label_tensor,
        optimizer=optimizer,
        is_new_run=is_new_run,
        run=run,
        run_name=run_name,
        **kwargs,
    )

def train_in_memory_plain(
    model,
    train_image_tensor,
    train_label_tensor,
    val_image_tensor,
    val_label_tensor,
    optimizer=None,
    is_new_run=True,
    run=None,
    run_name="default",
    weight_decay=False,
    patience=None,
):
    criterion = torch.nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer_kwargs = {"lr": config.LEARNING_RATE}
        if weight_decay:
            optimizer_kwargs["weight_decay"] = config.WEIGHT_DECAY
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)

    best_params = None
    best_optimizer_state = None
    best_val_loss = float("inf")
    patience = config.PATIENCE if patience is None else patience
    stalled_epochs = 0

    if is_new_run:
        run = _init_wandb_run(config.REG_TYPE, run_name, is_new_run)

    train_size = train_image_tensor.shape[0]

    for epoch in range(config.EPOCHS):
        model.train()
        total_train_loss = 0

        indices = torch.randperm(train_size)
        train_image_tensor = train_image_tensor[indices]
        train_label_tensor = train_label_tensor[indices]

        for batch_index in range(0, train_size, config.BATCH_SIZE):
            inputs = train_image_tensor[batch_index:batch_index + config.BATCH_SIZE].to(config.DEVICE)
            targets = train_label_tensor[batch_index:batch_index + config.BATCH_SIZE].to(config.DEVICE)

            outputs = model(inputs)
            loss = criterion(input=outputs, target=targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            total_train_loss += loss.item()
            optimizer.step()

        avg_train_loss = total_train_loss / max(1, math.ceil(train_size / config.BATCH_SIZE))
        current_val_loss, val_acc = validate_in_memory(model, val_image_tensor, val_label_tensor)
        current_lr = optimizer.param_groups[0]["lr"]

        if run is not None:
            run.log({"train/loss": avg_train_loss, "val/loss": current_val_loss, "val/acc": val_acc, "train/lr": current_lr})

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_params = deepcopy(model.state_dict())
            best_optimizer_state = deepcopy(optimizer.state_dict())
            stalled_epochs = 0
        else:
            stalled_epochs += 1

        if stalled_epochs >= patience:
            break

    if best_params is not None:
        model.load_state_dict(best_params)
    if best_optimizer_state is not None:
        optimizer.load_state_dict(best_optimizer_state)

    print(f"Ended training after {epoch}")

    return model, run


def train_in_memory_regularized(
    model,
    train_image_tensor,
    train_label_tensor,
    val_image_tensor,
    val_label_tensor,
    optimizer=None,
    is_new_run=True,
    run=None,
    run_name="default",
    sparsity_delta_threshold=None,
    sparsity_patience=None,
    epsilon_schedule_fn=None,
    lambda_schedule_fn=None,
):
    criterion = torch.nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

    previous_thresholded_sparsity = None
    sparsity_stall_count = 0
    epsilon_update_count = 0
    lambda_update_count = 0
    current_lambda_reg = config.LAMBDA_REG
    current_epsilon = config.EPSILON_START
    sparsity_delta_threshold = config.THRESHOLDED_SPARSITY_MIN_DELTA if sparsity_delta_threshold is None else sparsity_delta_threshold
    sparsity_patience = config.PATIENCE if sparsity_patience is None else sparsity_patience
    config.EPSILON = config.EPSILON_START

    if is_new_run:
        run = _init_wandb_run(config.REG_TYPE, run_name, is_new_run)

    train_size = train_image_tensor.shape[0]
    total_num_batches = math.ceil(train_size / config.BATCH_SIZE)
    update_batches = _regularization_update_batches(total_num_batches)

    for epoch in range(config.EPOCHS):
        # advance lambda once per epoch
        lambda_update_count += 1
        current_lambda_reg = _advance_lambda(lambda_update_count, lambda_schedule_fn)
        model.train()
        total_train_loss = 0

        indices = torch.randperm(train_size)
        train_image_tensor = train_image_tensor[indices]
        train_label_tensor = train_label_tensor[indices]

        for batch_index in range(0, train_size, config.BATCH_SIZE):
            inputs = train_image_tensor[batch_index:batch_index + config.BATCH_SIZE].to(config.DEVICE)
            targets = train_label_tensor[batch_index:batch_index + config.BATCH_SIZE].to(config.DEVICE)

            outputs = model(inputs)
            data_loss = criterion(input=outputs, target=targets)

            match(config.REG_TYPE):
                case "WL1":
                    batch_number = (batch_index // config.BATCH_SIZE) + 1
                    if batch_number in update_batches:
                        epsilon_update_count += 1
                        current_epsilon = _advance_epsilon(epsilon_update_count, epsilon_schedule_fn)
                        config.EPSILON = current_epsilon
                        L1_penalty_update(model)

                    reg_loss = current_lambda_reg * calculate_WL1_norm(model)
                    loss = data_loss + reg_loss
                case "L1":
                    reg_loss = config.LAMBDA_REG * calculate_L1_norm(model)
                    loss = data_loss + reg_loss
                case "L0":
                    loss = data_loss
                case "None":
                    loss = data_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            total_train_loss += data_loss.item()
            optimizer.step()

        avg_train_loss = total_train_loss / total_num_batches
        current_val_loss, val_acc = validate_in_memory(model, val_image_tensor, val_label_tensor)
        current_thresholded_sparsity = calculate_thresholded_sparsity(model)
        current_real_sparsity = calculate_real_sparsity(model)
        current_lr = optimizer.param_groups[0]["lr"]
        sparsity_delta = None if previous_thresholded_sparsity is None else abs(current_thresholded_sparsity - previous_thresholded_sparsity)

        if run is not None:
            run.log({
                "train/loss": avg_train_loss,
                "val/loss": current_val_loss,
                "val/acc": val_acc,
                "train/thresholded_sparsity": current_thresholded_sparsity,
                "train/thresholded_sparsity_delta": sparsity_delta,
                "train/real_sparsity": current_real_sparsity,
                "train/lr": current_lr,
                "train/epsilon": current_epsilon,
                "train/lambda_reg": current_lambda_reg,
            })

        sparsity_stall_count, should_stop, sparsity_delta = _regularized_sparsity_stop(
            previous_thresholded_sparsity,
            current_thresholded_sparsity,
            sparsity_stall_count,
            sparsity_delta_threshold,
            sparsity_patience,
            current_epsilon if config.REG_TYPE == "WL1" else None,
        )

        previous_thresholded_sparsity = current_thresholded_sparsity

        if should_stop:
            break

    print(f"Ended training after {epoch}")

    return model, run
