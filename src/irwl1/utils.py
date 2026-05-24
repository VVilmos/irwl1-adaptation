import torch
import torch.nn as nn
import wandb
from copy import deepcopy

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


def train_in_memory(model, train_image_tensor, train_label_tensor, val_image_tensor, val_label_tensor, optimizer=None, apply_reg=False, is_new_run=True, run=None, run_name="default"):
    criterion = torch.nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val_loss = float('inf')
    current_patience = config.PATIENCE
    best_params = None

    if is_new_run:
        if config.REG_TYPE == "L1":
            run = wandb.init(project=f"pruning_{config.MODEL}_{config.REG_TYPE}", name=f"{config.MODE[:-5]}_lambda{round(config.LAMBDA_REG, 3)}", mode=config.WANDB_MODE)
        elif config.REG_TYPE == "WL1":
            run = wandb.init(project=f"pruning_{config.MODEL}_{config.REG_TYPE}", name=f"{config.MODE[:-5]}_threshold{config.WEIGHT_PRUNING_THRESHOLD}_updatefreq{config.UPDATE_PER_EPOCH}_eps{config.EPSILON}", mode=config.WANDB_MODE)
        elif config.REG_TYPE == "None":
            run = wandb.init(project=f"pre_pruning_tests_{config.MODEL}", name=run_name, mode=config.WANDB_MODE)

    train_size = train_image_tensor.shape[0]
    total_num_batches = round(train_size / config.BATCH_SIZE)

    update_interval = max(1, total_num_batches / config.UPDATE_PER_EPOCH)

    for epoch in range(config.EPOCHS):
        model.train()
        total_train_loss = 0

        indices = torch.randperm(train_size)
        train_image_tensor = train_image_tensor[indices]
        train_label_tensor = train_label_tensor[indices]

        for i in range(0, train_size, config.BATCH_SIZE):
            outputs = model(train_image_tensor[i:i + config.BATCH_SIZE])
            data_loss = criterion(input=outputs, target=train_label_tensor[i:i + config.BATCH_SIZE])

            if apply_reg:
                match(config.REG_TYPE):
                    case "WL1":
                        reg_loss = config.LAMBDA_REG * calculate_WL1_norm(model)
                        loss = data_loss + reg_loss
                    case "L1":
                        reg_loss = config.LAMBDA_REG * calculate_L1_norm(model)
                        loss = data_loss + reg_loss
                    case "L0":
                        loss = data_loss
                    case "None":
                        loss = data_loss
            else:
                loss = data_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            total_train_loss += data_loss.item()
            optimizer.step()

            if apply_reg and config.REG_TYPE == "WL1" and ((i/config.BATCH_SIZE) % update_interval == update_interval - 1):
                L1_penalty_update(model)

        avg_train_loss = total_train_loss / total_num_batches
        current_val_loss, val_acc = validate_in_memory(model, val_image_tensor, val_label_tensor)
        current_thresholded_sparsity = calculate_thresholded_sparsity(model)
        current_real_sparsity = calculate_real_sparsity(model)

        run.log({"train/loss": avg_train_loss, "val/loss": current_val_loss, "val/acc": val_acc,  "train/thresholded_sparsity": current_thresholded_sparsity, "train/real_sparsity": current_real_sparsity})

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_params = deepcopy(model.state_dict())
            current_patience = config.PATIENCE
        else:
            current_patience -= 1

        if current_patience == 0:
            break

    if best_params is not None:
        model.load_state_dict(best_params)

    if apply_reg:
        print(f"Ended reg_train phase after {epoch}")
    else:
        print(f"Ended recovery phase after {epoch}")

    return model, run


def train(model, train_loader, val_loader, optimizer=None, apply_reg=False, is_new_run=True, run=None, run_name="default", weight_decay=False):
    criterion = torch.nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer_kwargs = {"lr": config.LEARNING_RATE}
        if not apply_reg and weight_decay:
            optimizer_kwargs["weight_decay"] = config.WEIGHT_DECAY
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)

    best_val_loss = float('inf')
    current_patience = config.PATIENCE
    best_params = None

    if is_new_run:
        if config.REG_TYPE == "L1":
            run = wandb.init(project=f"pruning_{config.MODEL}_{config.REG_TYPE}", name=f"{config.MODE[:-5]}_lambda{round(config.LAMBDA_REG, 3)}", mode=config.WANDB_MODE)
        elif config.REG_TYPE == "WL1":
            run = wandb.init(project=f"pruning_{config.MODEL}_{config.REG_TYPE}", name=f"{config.MODE[:-5]}_threshold{config.WEIGHT_PRUNING_THRESHOLD}_updatefreq{config.UPDATE_PER_EPOCH}_eps{config.EPSILON}", mode=config.WANDB_MODE)
        elif config.REG_TYPE == "None":
            run = wandb.init(project=f"pre_pruning_tests_{config.MODEL}", name=run_name, mode=config.WANDB_MODE)

    total_num_batches = len(train_loader)
    update_interval = max(1, round(total_num_batches / config.UPDATE_PER_EPOCH))

    for epoch in range(config.EPOCHS):
        model.train()
        total_train_loss = 0

        for batch_index, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            outputs = model(inputs)
            data_loss = criterion(input=outputs, target=targets)

            if apply_reg:
                match(config.REG_TYPE):
                    case "WL1":
                        reg_loss = config.LAMBDA_REG * calculate_WL1_norm(model)
                        loss = data_loss + reg_loss
                    case "L1":
                        reg_loss = config.LAMBDA_REG * calculate_L1_norm(model)
                        loss = data_loss + reg_loss
                    case "L0":
                        loss = data_loss
                    case "None":
                        loss = data_loss
            else:
                loss = data_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            total_train_loss += data_loss.item()
            optimizer.step()

            if apply_reg and config.REG_TYPE == "WL1" and ((batch_index + 1) % update_interval == 0):
                L1_penalty_update(model)

        avg_train_loss = total_train_loss / total_num_batches
        current_val_loss, val_acc = validate(model, val_loader)
        current_thresholded_sparsity = calculate_thresholded_sparsity(model)
        current_real_sparsity = calculate_real_sparsity(model)

        run.log({"train/loss": avg_train_loss, "val/loss": current_val_loss, "val/acc": val_acc, "train/thresholded_sparsity": current_thresholded_sparsity, "train/real_sparsity": current_real_sparsity})

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_params = deepcopy(model.state_dict())
            current_patience = config.PATIENCE
        else:
            current_patience -= 1

        if current_patience == 0:
            break

    if best_params is not None:
        model.load_state_dict(best_params)

    if apply_reg:
        print(f"Ended reg_train phase after {epoch}")
    else:
        print(f"Ended recovery phase after {epoch}")

    return model, run


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

        elif type(layer) == nn.Linear:
            weight = layer.weight
            with torch.no_grad():
                layer.register_buffer("mask", torch.ones(weight.shape, dtype=torch.float32, device=weight.device))
                weight.register_hook(lambda grad, l=layer: grad * l.mask)


# using global threshold for pruning
def global_pruning(model, masking=False):
    threshold = config.WEIGHT_PRUNING_THRESHOLD
    for name, layer in model.named_modules():
        if _should_skip_module(name):
            continue

        if type(layer) == nn.Conv2d:
            weight = layer.weight
            with torch.no_grad():
                match(config.MODE):
                    case "weight-wise":
                        weight_amplitude = torch.abs(weight)
                        pruned_weight = weight * (weight_amplitude >= threshold).to(weight.dtype)
                        layer.weight.copy_(pruned_weight)

                        if masking:
                            layer.mask.copy_((weight_amplitude >= threshold).to(weight.dtype))

                    case "kernel-wise":
                        pass
                    case "channel-wise":
                        pass

        elif type(layer) == nn.Linear:
            weight = layer.weight
            with torch.no_grad():
                weight_amplitude = torch.abs(weight)
                pruned_weight = weight * (weight_amplitude >= threshold).to(weight.dtype)
                layer.weight.copy_(pruned_weight)

                if masking:
                    layer.mask.copy_((weight_amplitude >= threshold).to(weight.dtype))


def save_sparacc_curve(spar_cp, acc_cp, pgd_norm_cp=None, corr_acc_cp=None, fab_norm_cp=None, path=config.CURVE_PATH):
    num_records = len(spar_cp)
    df = pandas.read_csv(path, index_col=False)
    row = {"model": config.MODEL, "reg_type": [config.REG_TYPE] * num_records, "mode": [config.MODE] * num_records, "sparsity": spar_cp, "accuracy": acc_cp, "pgd_norm": pgd_norm_cp, "fab_norm": fab_norm_cp, "corr_acc": corr_acc_cp,
           "lambda": [config.LAMBDA_REG] * num_records, "threshold": [config.WEIGHT_PRUNING_THRESHOLD] * num_records, "update_per_epoch": [config.UPDATE_PER_EPOCH] * num_records, "epsilon": [config.EPSILON] * num_records, "weight_decay": [config.WEIGHT_DECAY] * num_records}
    df = pandas.concat([df, pandas.DataFrame(row)], ignore_index=True)

    df.to_csv(path, index=False)


def save_cifar10c_row(sparsity: float, test_accuracy: float, corruption_accuracies: dict, path: str = "results/resnet20cifar10_corruptions.csv") -> None:
    """Save a single-row record with per-corruption accuracies.

    The CSV will have one row per call. Columns include standard metadata
    ('model','reg_type','mode','sparsity','test_accuracy','lambda','threshold',...)
    and one column per corruption (keys from `corruption_accuracies`). If the
    file already exists, new corruption columns will be appended to the right.
    """
    path = Path(path)
    base = {
        "model": config.MODEL,
        "reg_type": config.REG_TYPE,
        "mode": config.MODE,
        "sparsity": sparsity,
        "test_accuracy": test_accuracy,
        "lambda": config.LAMBDA_REG,
        "threshold": config.WEIGHT_PRUNING_THRESHOLD,
        "update_per_epoch": config.UPDATE_PER_EPOCH,
        "epsilon": config.EPSILON,
        "weight_decay": config.WEIGHT_DECAY,
    }

    # incorporate per-corruption accuracies as separate columns
    for corr, acc in (corruption_accuracies or {}).items():
        base[corr] = acc

    df = pandas.DataFrame([base])
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        df.to_csv(path, index=False)
        return

    # If file exists, preserve existing column order and append new columns at the end
    existing_cols = list(pandas.read_csv(path, nrows=0).columns)
    # ensure all existing cols appear in df (fill missing with NA)
    for col in existing_cols:
        if col not in df.columns:
            df[col] = pandas.NA

    # order columns: existing then any new ones
    ordered_cols = existing_cols + [c for c in df.columns if c not in existing_cols]
    df = df.reindex(columns=ordered_cols)
    df.to_csv(path, mode="a", header=False, index=False)
