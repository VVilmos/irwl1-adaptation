import torch
import torch.nn as nn
import wandb
from copy import deepcopy
import irwl1.config as config
from irwl1.regularization import calculate_L1_norm, calculate_WL1_norm, L1_penalty_update
import pandas

def calculate_sparsity(model, threshold=1e-5):
    num_total_weights, num_zero_weights = 0, 0 
    for name, layer in model.named_modules():
        num_total_weights += torch.numel(layer.weight) 
        num_zero_weights += torch.sum(torch.abs(layer.weight) < threshold).item()

    return num_zero_weights / num_total_weights  * 100

def calculate_real_sparsity(model):
    num_total_weights, num_nonzero_weights = 0, 0 
    for name, layer in model.named_modules():
        if type(layer) in [nn.Conv2d, nn.Linear]:
            num_total_weights += torch.numel(layer.weight)
            num_nonzero_weights += layer.weight.nonzero().shape[0] 
        
    return (num_total_weights - num_nonzero_weights) / num_total_weights  * 100

def train_in_memory(model, train_image_tensor, train_label_tensor, val_image_tensor, val_label_tensor, is_new_run=True, run=None, run_name="default"):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf') 
    current_patience = config.PATIENCE
    best_params = None

    if is_new_run:
        if config.REG_TYPE == "L1":
            run = wandb.init(project = f"pruning_{config.REG_TYPE}", name=f"{config.MODE[:-5]}_lambda{round(config.LAMBDA_REG, 3)}")
        elif config.REG_TYPE == "WL1":
            run = wandb.init(project = f"pruning_{config.REG_TYPE}", name=f"{config.MODE[:-5]}_lambda{config.LAMBDA_REG}_K{config.K}_eps{config.EPSILON}")
        elif config.REG_TYPE == "None":
            run = wandb.init(project = f"pre_pruning_tests", name=run_name)

    train_size = train_image_tensor.shape[0]
    num_batches = round(train_size/config.BATCH_SIZE)

    for epoch in range(config.EPOCHS):
        model.train()
        total_train_loss= 0

        indices = torch.randperm(train_size)  # shuffle training tensor for every epoch
        train_image_tensor = train_image_tensor[indices]
        train_label_tensor = train_label_tensor[indices]

        for i in range(0, train_size, config.BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(train_image_tensor[i:i+config.BATCH_SIZE])
            data_loss = criterion(input=outputs, target=train_label_tensor[i:i+config.BATCH_SIZE])

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

            loss.backward()
            total_train_loss += loss.item()
            optimizer.step()

            match(config.REG_TYPE):
                case "WL1":
                    if (i % config.K == config.K-1):
                        model.apply(L1_penalty_update)

        avg_train_loss = total_train_loss / num_batches
        current_val_loss, val_acc = validate_in_memory(model, val_image_tensor, val_label_tensor)
        current_sparsity = calculate_real_sparsity(model)

        run.log({"train/loss": avg_train_loss, "val/loss": current_val_loss, "val/acc": val_acc, "train/sparsity": current_sparsity})

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
      
    return model, run


def validate_in_memory(model, val_image_tensor, val_label_tensor):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total = val_label_tensor.shape[0]
    num_batches = round(total/config.BATCH_SIZE)
    with torch.no_grad():
          outputs = model(val_image_tensor)
          loss = criterion(input = outputs, target=val_label_tensor)
          total_val_loss = loss.item()
          max_values, predicted_classes = torch.max(outputs, 1)
          total_correct = (predicted_classes == val_label_tensor).sum().item()

    avg_val_loss = total_val_loss / num_batches
    val_acc = 100 * total_correct / total

    return avg_val_loss, val_acc


def test_in_memory(model, test_image_tensor, test_label_tensor):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total = test_label_tensor.shape[0]
    num_batches = round(total / config.BATCH_SIZE)
    with torch.no_grad():
          outputs = model(test_image_tensor)
          loss = criterion(input = outputs, target=test_label_tensor)
          total_test_loss = loss.item()
          max_values, predicted_classes = torch.max(outputs, 1)
          total_correct = (predicted_classes == test_label_tensor).sum().item()

    avg_test_loss = total_test_loss / num_batches
    test_acc = 100 * total_correct / total

    return avg_test_loss, test_acc

def init_mask(model):
    for name, layer in model.named_modules():
        if type(layer) == nn.Conv2d:
            weight = layer.weight
            with torch.no_grad():
                match(config.MODE):
                    case "weight-wise":
                        layer.register_buffer("mask", torch.ones(weight.shape, dtype=torch.float32, device=weight.device))
                    case "kernel-wise":
                        layer.register_buffer("mask", torch.ones(weight.shape[:2], dtype=torch.float32, device=weight.device))
                    case "channel-wise":
                        layer.register_buffer("mask", torch.ones(weight.shape[0], dtype=torch.float32, device=weight.device))

        elif (type(layer) == nn.Linear) and (name != "out"):
            weight = layer.weight
            with torch.no_grad():
                layer.register_buffer("mask", torch.ones(weight.shape, dtype=torch.float32, device=weight.device))

def prune_with_mask(model, compress_ratio=None):
    if compress_ratio is None:
        compress_ratio = config.COMPRESS_RATIO

    for name, layer in model.named_modules():
        if type(layer) == nn.Conv2d:
            weight = layer.weight
            with torch.no_grad():
                match(config.MODE):
                    case "weight-wise":
                        weight_amplitude = torch.abs(weight)
                        active = layer.mask > 0
                        active_values = weight_amplitude[active]
                        if active_values.numel() == 0:
                            continue
                        threshold = torch.quantile(active_values, q=compress_ratio)
                        layer.mask.copy_((active & (weight_amplitude >= threshold)).to(layer.mask.dtype))

                        weight.mul_(layer.mask)
                    case "kernel-wise":
                        kernel_amplitude = weight.pow(2).sum(dim=(2, 3)).sqrt()
                        active = layer.mask > 0
                        active_values = kernel_amplitude[active]
                        if active_values.numel() == 0:
                            continue
                        threshold = torch.quantile(active_values, q=compress_ratio)
                        layer.mask.copy_((active & (kernel_amplitude >= threshold)).to(layer.mask.dtype))

                        weight.mul_(layer.mask[:, :, None, None])
                    case "channel-wise":
                        channel_amplitude = weight.pow(2).sum(dim=(1, 2, 3)).sqrt()
                        active = layer.mask > 0
                        active_values = channel_amplitude[active]
                        if active_values.numel() == 0:
                            continue
                        threshold = torch.quantile(active_values, q=compress_ratio)
                        layer.mask.copy_((active & (channel_amplitude >= threshold)).to(layer.mask.dtype))

                        weight.mul_(layer.mask[:, None, None, None])
        elif (type(layer) == nn.Linear) and (name != "out"):
            weight = layer.weight
            with torch.no_grad():
                weight_amplitude = torch.abs(weight)
                active = layer.mask > 0
                active_values = weight_amplitude[active]
                if active_values.numel() == 0:
                    continue
                threshold = torch.quantile(active_values, q=compress_ratio)
                layer.mask.copy_((active & (weight_amplitude >= threshold)).to(layer.mask.dtype))
                weight.mul_(layer.mask)

def save_sparacc_curve(spar_cp, acc_cp):
    num_records = len(spar_cp)
    df = pandas.read_csv(config.CURVE_PATH, index_col=False)
    row = {"reg_type": [config.REG_TYPE]*num_records, "mode": [config.MODE]*num_records, "sparsity": spar_cp, "accuracy": acc_cp, 
           "lambda": [config.LAMBDA_REG]*num_records, "compress_ratio": [config.COMPRESS_RATIO]*num_records, "K": [config.K]*num_records, "epsilon": [config.EPSILON]*num_records}
    df = pandas.concat([df, pandas.DataFrame(row)], ignore_index=True)

    df.to_csv(config.CURVE_PATH, index=False)