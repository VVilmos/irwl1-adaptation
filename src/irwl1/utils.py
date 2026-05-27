import torch
import torch.nn as nn
import wandb
from copy import deepcopy
import math

import pandas
from pathlib import Path

import irwl1.config as config
from irwl1.regularization import calculate_L1_norm, calculate_WL1_norm, L1_penalty_update


def _should_skip_module(name):
    return any(part in {"out", "fc", "downsample"} for part in name.split("."))


def calculate_thresholded_sparsity(model):
    threshold = config.WEIGHT_PRUNING_THRESHOLD
    num_total_weights, num_zero_weights = 0, 0

    for name, layer in model.named_modules():
        if _should_skip_module(name):
            continue

        if type(layer) in [nn.Conv2d, nn.Linear]:
            num_total_weights += torch.numel(layer.weight)
            num_zero_weights += torch.sum(torch.abs(layer.weight) < threshold).item()

    if num_total_weights == 0:
        return 0.0

    return num_zero_weights / num_total_weights * 100


def calculate_real_sparsity(model):
    num_total_weights, num_zero_weights = 0, 0

    for name, layer in model.named_modules():
        if _should_skip_module(name):
            continue

        if type(layer) in [nn.Conv2d, nn.Linear]:
            num_total_weights += torch.numel(layer.weight)
            num_zero_weights += torch.sum(layer.weight == 0.0).item()

    if num_total_weights == 0:
        return 0.0

    return num_zero_weights / num_total_weights * 100


def _regularization_update_batches(num_batches):
    update_batches = set()

    for update_index in range(1, config.UPDATE_PER_EPOCH + 1):
        batch_number = math.ceil(update_index * num_batches / (config.UPDATE_PER_EPOCH + 1))
        update_batches.add(min(max(batch_number, 1), num_batches))

    return update_batches


def _normalize_progress(step_count, total_steps):
    return min(max(step_count / max(1, total_steps), 0.0), 1.0)


def _sigmoid_progress(progress, center, steepness):
    center = min(max(center, 0.0), 1.0)
    steepness = max(1e-6, steepness)
    sigmoid_start = 1 / (1 + math.exp(steepness * center))
    sigmoid_end = 1 / (1 + math.exp(-steepness * (1 - center)))
    sigmoid_progress = 1 / (1 + math.exp(-steepness * (progress - center)))
    eased_progress = (sigmoid_progress - sigmoid_start) / (sigmoid_end - sigmoid_start)
    return min(max(eased_progress, 0.0), 1.0)


def _exponential_decay(start_value, end_value, progress, steepness, center):
    eased_progress = _sigmoid_progress(progress, center, steepness)
    log_value = math.log(start_value) + eased_progress * (math.log(end_value) - math.log(start_value))
    return max(end_value, math.exp(log_value))


def _exponential_growth(start_value, end_value, progress, steepness, center):
    eased_progress = _sigmoid_progress(progress, center, steepness)
    log_value = math.log(start_value) + eased_progress * (math.log(end_value) - math.log(start_value))
    return min(end_value, math.exp(log_value))


def _linear_growth(start_value, end_value, progress):
    progress = min(max(progress, 0.0), 1.0)
    return start_value + progress * (end_value - start_value)


def _quadratic_growth(start_value, end_value, progress):
    progress = min(max(progress, 0.0), 1.0)
    eased_progress = progress * progress
    return start_value + eased_progress * (end_value - start_value)


def _default_epsilon_schedule(update_count):
    progress = _normalize_progress(update_count, getattr(config, "EPSILON_DECAY_STEPS", 1))
    return _exponential_decay(
        config.EPSILON_START,
        config.EPSILON_END,
        progress,
        getattr(config, "EPSILON_SIGMOID_STEEPNESS", 10),
        getattr(config, "EPSILON_SIGMOID_CENTER", 0.5),
    )


def _default_lambda_schedule(update_count):
    growth_steps = getattr(config, "LAMBDA_REG_GROWTH_STEPS", getattr(config, "EPSILON_DECAY_STEPS", 1))
    progress = _normalize_progress(update_count, growth_steps)
    lambda_start = getattr(config, "LAMBDA_REG_START", 1e-14)
    lambda_end = config.LAMBDA_REG_END
    safe_lambda_start = max(1e-30, min(lambda_start, lambda_end))

    lambda_value = _exponential_growth(
        safe_lambda_start,
        lambda_end,
        progress,
        getattr(config, "LAMBDA_SIGMOID_STEEPNESS", 10),
        getattr(config, "LAMBDA_SIGMOID_CENTER", 0.5),
    )

    # Keep the first active regularization below Adam epsilon so optimization remains data-loss dominated.
    if update_count == 1:
        first_active_max = getattr(config, "LAMBDA_FIRST_ACTIVE_MAX", getattr(config, "ADAM_EPSILON", 1e-8) * 0.1)
        lambda_value = min(lambda_value, first_active_max)

    return lambda_value


def _advance_epsilon(update_count, epsilon_schedule_fn=None):
    epsilon_schedule_fn = epsilon_schedule_fn or _default_epsilon_schedule
    config.EPSILON = epsilon_schedule_fn(update_count)
    return config.EPSILON


def _advance_lambda(update_count, lambda_schedule_fn=None):
    lambda_schedule_fn = lambda_schedule_fn or _default_lambda_schedule
    config.LAMBDA_REG = lambda_schedule_fn(update_count)
    return config.LAMBDA_REG


def _regularized_sparsity_stop(previous_thresholded_sparsity, current_thresholded_sparsity, stall_count, threshold, patience, current_epsilon=None):
    if previous_thresholded_sparsity is None:
        return 0, False, None

    if (current_epsilon is not None) and (current_epsilon > config.EPSILON_END + 1e-7):
        return 0, False, None

    sparsity_delta = abs(current_thresholded_sparsity - previous_thresholded_sparsity)
    if sparsity_delta <= threshold:
        stall_count += 1
    else:
        stall_count = 0

    return stall_count, stall_count >= patience, sparsity_delta


def _init_wandb_run(reg_type, run_name, is_new_run):
    if not is_new_run:
        return None

    if reg_type == "L1":
        return wandb.init(project=f"pruning_{config.MODEL}_{config.REG_TYPE}", name=f"{config.MODE[:-5]}_lambda{round(config.LAMBDA_REG, 3)}", mode=config.WANDB_MODE)
    if reg_type == "WL1":
        return wandb.init(project=f"pruning_{config.MODEL}_{config.REG_TYPE}", name=f"{config.MODE[:-5]}_threshold{config.WEIGHT_PRUNING_THRESHOLD}_updatefreq{config.UPDATE_PER_EPOCH}_eps{config.EPSILON}", mode=config.WANDB_MODE)
    if reg_type == "None":
        return wandb.init(project=f"pre_pruning_tests_{config.MODEL}", name=run_name, mode=config.WANDB_MODE)

    return None


def _save_rewind_checkpoint(model, path, epoch, optimizer=None):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": deepcopy(model.state_dict()),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = deepcopy(optimizer.state_dict())

    torch.save(checkpoint, path)


def rewind_model_to_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    current_masks = {
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers()
        if name.endswith("mask")
    }

    model.load_state_dict(state_dict)

    with torch.no_grad():
        for name, layer in model.named_modules():
            if _should_skip_module(name):
                continue

            mask_name = f"{name}.mask" if name else "mask"
            mask = current_masks.get(mask_name)
            if mask is None:
                continue

            layer.mask.copy_(mask)
            if hasattr(layer, "weight"):
                layer.weight.mul_(layer.mask.to(layer.weight.dtype))

    return model



def train_regularized(model, train_loader, val_loader, optimizer=None, is_new_run=True, run=None, run_name="default", weight_decay=False,
                      sparsity_delta_threshold=None, sparsity_patience=None, epsilon_schedule_fn=None, lambda_schedule_fn=None):

    criterion = torch.nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer_kwargs = {"lr": config.LEARNING_RATE}
        if weight_decay:
            optimizer_kwargs["weight_decay"] = config.WEIGHT_DECAY
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)

    previous_thresholded_sparsity = None
    sparsity_stall_count = 0


    epsilon_update_count = 0
    lambda_update_count = 0
    current_lambda_reg = config.LAMBDA_REG_END if not config.IS_LAMBDA_RISE else config.LAMBDA_REG_START
    current_epsilon = config.EPSILON_END if not config.IS_EPSILON_DECAY else config.EPSILON_START
    sparsity_delta_threshold = config.THRESHOLDED_SPARSITY_MIN_DELTA if sparsity_delta_threshold is None else sparsity_delta_threshold
    sparsity_patience = config.SPAR_PATIENCE if sparsity_patience is None else sparsity_patience
    config.EPSILON = config.EPSILON_END if not config.IS_EPSILON_DECAY else config.EPSILON_START
    if not config.IS_EPSILON_DECAY:
        current_epsilon = config.EPSILON_END

    if is_new_run:
        run = _init_wandb_run(config.REG_TYPE, run_name, is_new_run)

    total_num_batches = len(train_loader)
    update_batches = _regularization_update_batches(total_num_batches)

    for epoch in range(config.MAX_EPOCHS):
        model.train()
        # advance lambda once per epoch (slower growth)
        if config.IS_LAMBDA_RISE:
            lambda_update_count += 1
        total_train_loss = 0

        for batch_index, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            outputs = model(inputs)
            data_loss = criterion(input=outputs, target=targets)

            match(config.REG_TYPE):
                case "WL1":
                    batch_number = batch_index + 1

                    if batch_number in update_batches:
                        if config.IS_EPSILON_DECAY:
                            epsilon_update_count += 1
                            current_epsilon = _advance_epsilon(epsilon_update_count, epsilon_schedule_fn)
                        else:
                            current_epsilon = config.EPSILON_END

                        # Always update WL1 penalties at the scheduled update batches,
                        # even if epsilon decay is disabled — use the current_epsilon value.
                        L1_penalty_update(model, current_epsilon)

                    reg_loss = current_lambda_reg * calculate_WL1_norm(model)
                    loss = data_loss + reg_loss
                case "L1":
                    reg_loss = current_lambda_reg * calculate_L1_norm(model)
                    loss = data_loss + reg_loss
                case "L0":
                    loss = data_loss
                case "None":
                    loss = data_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            total_train_loss += data_loss.item()
            optimizer.step()

        if config.IS_LAMBDA_RISE:
            current_lambda_reg = _advance_lambda(lambda_update_count, lambda_schedule_fn)
        else:
            current_lambda_reg = config.LAMBDA_REG_END
        avg_train_loss = total_train_loss / total_num_batches
        current_val_loss, val_acc = validate(model, val_loader)
        current_thresholded_sparsity = calculate_thresholded_sparsity(model)
        current_real_sparsity = calculate_real_sparsity(model)
        sparsity_delta = None if previous_thresholded_sparsity is None else abs(current_thresholded_sparsity - previous_thresholded_sparsity)

        if run is not None:
            run.log({
                "train/loss": avg_train_loss,
                "val/loss": current_val_loss,
                "val/acc": val_acc,
                "train/thresholded_sparsity": current_thresholded_sparsity,
                "train/real_sparsity": current_real_sparsity,
                "reg/epsilon": current_epsilon,
                "reg/lambda_reg": current_lambda_reg,
            })

        sparsity_stall_count, should_stop, _ = _regularized_sparsity_stop(
            previous_thresholded_sparsity,
            current_thresholded_sparsity,
            sparsity_stall_count,
            sparsity_delta_threshold,
            sparsity_patience,
            current_epsilon,
        )

        previous_thresholded_sparsity = current_thresholded_sparsity

        if should_stop:
            break

    num_epochs = epoch + 1
    print(f"Ended regularization after {num_epochs} epochs")

    return model, run, num_epochs


def train( model, train_loader, val_loader, optimizer=None, is_new_run=True, run=None, run_name="default", weight_decay=False,
          rewind=False, rewind_checkpoint_path=None, rewind_epoch=None):
    criterion = torch.nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer_kwargs = {"lr": config.LEARNING_RATE}
        if weight_decay:
            optimizer_kwargs["weight_decay"] = config.WEIGHT_DECAY
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)

    best_params = None
    best_optimizer_state = None
    best_val_loss = float("inf")
    patience = config.LOSS_PATIENCE
    stalled_epochs = 0

    if is_new_run:
        run = _init_wandb_run(config.REG_TYPE, run_name, is_new_run)

    total_num_batches = len(train_loader)
    rewind_epoch = config.REWIND_EPOCH if rewind_epoch is None else rewind_epoch
    rewind_checkpoint_saved = False

    for epoch in range(config.MAX_EPOCHS):
        model.train()
        total_train_loss = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            outputs = model(inputs)
            loss = criterion(input=outputs, target=targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            total_train_loss += loss.item()
            optimizer.step()

        avg_train_loss = total_train_loss / max(1, total_num_batches)
        current_val_loss, val_acc = validate(model, val_loader)
        current_thresholded_sparsity = calculate_thresholded_sparsity(model)
        current_real_sparsity = calculate_real_sparsity(model)

        if rewind and (not rewind_checkpoint_saved) and ((epoch + 1) == rewind_epoch) and (rewind_checkpoint_path is not None):
            _save_rewind_checkpoint(model, rewind_checkpoint_path, epoch + 1, optimizer=optimizer)
            rewind_checkpoint_saved = True
            print(f"Saved rewind checkpoint at epoch {epoch + 1} to {rewind_checkpoint_path}")

        if run is not None:
            run.log({
                "train/loss": avg_train_loss,
                "val/loss": current_val_loss,
                "val/acc": val_acc,
                "train/thresholded_sparsity": current_thresholded_sparsity,
                "train/real_sparsity": current_real_sparsity,
                "reg/epsilon": 0,
                "reg/lambda_reg": 0,
            })
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

    num_epochs = epoch + 1
    print(f"Ended training after {num_epochs} epochs")

    if rewind and not rewind_checkpoint_saved:
        print(f"Warning: rewind checkpoint was not saved because training ended before epoch {rewind_epoch}")

    return model, run, num_epochs


def _evaluate_loader(model, data_loader):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    total_correct = 0
    total_examples = 0
    num_batches = len(data_loader)

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            outputs = model(inputs)
            loss = criterion(input=outputs, target=targets)
            total_loss += loss.item()

            _, predicted_classes = torch.max(outputs, 1)
            total_correct += (predicted_classes == targets).sum().item()
            total_examples += targets.shape[0]

    avg_loss = total_loss / num_batches
    accuracy = 100 * total_correct / total_examples

    return avg_loss, accuracy


def validate_in_memory(model, val_image_tensor, val_label_tensor):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total = val_label_tensor.shape[0]
    num_batches = round(total / config.BATCH_SIZE)
    with torch.no_grad():
        outputs = model(val_image_tensor)
        loss = criterion(input=outputs, target=val_label_tensor)
        total_val_loss = loss.item()
        max_values, predicted_classes = torch.max(outputs, 1)
        total_correct = (predicted_classes == val_label_tensor).sum().item()

    avg_val_loss = total_val_loss / num_batches
    val_acc = 100 * total_correct / total

    return avg_val_loss, val_acc


def validate(model, val_loader):
    return _evaluate_loader(model, val_loader)


def test_in_memory(model, test_image_tensor, test_label_tensor):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total = test_label_tensor.shape[0]
    num_batches = round(total / config.BATCH_SIZE)
    with torch.no_grad():
        outputs = model(test_image_tensor)
        loss = criterion(input=outputs, target=test_label_tensor)
        total_test_loss = loss.item()
        max_values, predicted_classes = torch.max(outputs, 1)
        total_correct = (predicted_classes == test_label_tensor).sum().item()

    avg_test_loss = total_test_loss / num_batches
    test_acc = 100 * total_correct / total

    return avg_test_loss, test_acc


def test(model, test_loader):
    return _evaluate_loader(model, test_loader)


def init_mask(model):
    for name, layer in model.named_modules():
        if _should_skip_module(name):
            continue

        if type(layer) == nn.Conv2d:
            weight = layer.weight
            with torch.no_grad():
                match(config.MODE):
                    case "weight-wise":
                        layer.register_buffer("mask", torch.ones(weight.shape, dtype=torch.float32, device=weight.device))
                        weight.register_hook(lambda grad, l=layer: grad * l.mask)
                    case "kernel-wise":
                        layer.register_buffer("mask", torch.ones(weight.shape[:2], dtype=torch.float32, device=weight.device))
                        weight.register_hook(lambda grad, l=layer: grad * l.mask[:, :, None, None])
                    case "channel-wise":
                        layer.register_buffer("mask", torch.ones(weight.shape[0], dtype=torch.float32, device=weight.device))
                        weight.register_hook(lambda grad, l=layer: grad * l.mask[:, None, None, None])

        elif (type(layer) == nn.Linear) and (config.MODE == "weight-wise"):
            weight = layer.weight
            with torch.no_grad():
                layer.register_buffer("mask", torch.ones(weight.shape, dtype=torch.float32, device=weight.device))
                weight.register_hook(lambda grad, l=layer: grad * l.mask)


# using global threshold for pruning
def global_pruning(model, masking=False):
    for name, layer in model.named_modules():
        if _should_skip_module(name):
            continue

        if type(layer) == nn.Conv2d:
            with torch.no_grad():
                match(config.MODE):
                    case "weight-wise":
                        threshold = config.WEIGHT_PRUNING_THRESHOLD
                        weight_amplitude = torch.abs(layer.weight)
                        pruned_weight = layer.weight * (weight_amplitude >= threshold).to(layer.weight.dtype)
                        layer.weight.copy_(pruned_weight)

                        if masking:
                            layer.mask.copy_((weight_amplitude >= threshold).to(layer.weight.dtype))

                    case "kernel-wise":
                        threshold = config.KERNEL_PRUNING_THRESHOLD
                        kernel_amplitude = torch.sqrt(torch.pow(layer.weight, 2).sum(dim=[2, 3]))
                        pruned_weight = layer.weight * (kernel_amplitude >= threshold).to(layer.weight.dtype)[:, :, None, None]
                        layer.weight.copy_(pruned_weight)

                        if masking:
                            layer.mask.copy_((kernel_amplitude >= threshold).to(layer.weight.dtype)[:, :, None, None])

                    case "channel-wise":
                        threshold = config.CHANNEL_PRUNING_THRESHOLD
                        channel_amplitude = torch.sqrt(torch.pow(layer.weight, 2).sum(dim=[1, 2, 3]))
                        pruned_weight = layer.weight * (channel_amplitude >= threshold).to(layer.weight.dtype)[:, None, None, None]
                        layer.weight.copy_(pruned_weight)
                        if masking:
                            layer.mask.copy_((channel_amplitude >= threshold).to(layer.weight.dtype)[:, None, None, None])

        elif (type(layer) == nn.Linear) and (config.MODE == "weight-wise"):
            weight = layer.weight
            with torch.no_grad():
                weight_amplitude = torch.abs(weight)
                pruned_weight = weight * (weight_amplitude >= threshold).to(weight.dtype)
                layer.weight.copy_(pruned_weight)

                if masking:
                    layer.mask.copy_((weight_amplitude >= threshold).to(weight.dtype))


def zero_pruned_optimizer_state(model, optimizer):
    if optimizer is None:
        return

    for name, layer in model.named_modules():
        if _should_skip_module(name):
            continue

        if not hasattr(layer, "mask"):
            continue

        weight = getattr(layer, "weight", None)
        if weight is None:
            continue

        state = optimizer.state.get(weight)
        if not state:
            continue

        mask = layer.mask.to(weight.dtype)
        with torch.no_grad():
            for state_value in state.values():
                if torch.is_tensor(state_value) and state_value.shape == weight.shape:
                    state_value.mul_(mask)


def save_sparacc_curve(spar_cp, acc_cp, path=config.CURVE_PATH):
    num_records = len(spar_cp)
    df = pandas.read_csv(path, index_col=False)
    row = {"model": config.MODEL, "reg_type": [config.REG_TYPE] * num_records, "mode": [config.MODE] * num_records, "sparsity": spar_cp,
            "accuracy": acc_cp, "lambda": [config.LAMBDA_REG] * num_records, "threshold": [config.WEIGHT_PRUNING_THRESHOLD] * num_records,
            "update_per_epoch": [config.UPDATE_PER_EPOCH] * num_records, "epsilon": [config.EPSILON] * num_records,
             "weight_decay": [config.WEIGHT_DECAY] * num_records
    }

    df = pandas.concat([df, pandas.DataFrame(row)], ignore_index=True)

    df.to_csv(path, index=False)


