import torch
import torch.nn as nn
import wandb
from copy import deepcopy
from irwl1.config import PATIENCE, REG_TYPE, MODE, LAMBDA_REG, EPOCHS, K, BATCH_SIZE, EPSILON, LEARNING_RATE, COMPRESS_RATIO
from irwl1.regularization import calculate_L1_norm, calculate_WL1_norm, L1_penalty_update

def calculate_sparsity(model, threshold=1e-5):
    num_total_weights, num_zero_weights = 0, 0 
    for name, layer in model.named_modules():
        num_total_weights += torch.numel(layer.weight) 
        num_zero_weights += torch.sum(torch.abs(layer.weight) < threshold).item()

    return num_zero_weights / num_total_weights  * 100

def calculate_real_sparsity(model):
    num_total_weights, num_zero_weights = 0, 0 
    for name, layer in model.named_modules():
        num_total_weights += torch.numel(layer.weight)
        num_nonzero_weights += layer.weight.nonzero().shape[0] 
        
    return (num_total_weights - num_nonzero_weights) / num_total_weights  * 100

def train_in_memory(model, train_image_tensor, train_label_tensor):
    optimizer = torch.optim.Adam(model.parameters, lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf') # Initialize to infinity
    current_patience = PATIENCE
    best_params = None # Initialize best_params
    if REG_TYPE == "L1":
      run = wandb.init(project = f"pruning/{REG_TYPE}", name=f"{MODE[:-5]}_lambda{round(LAMBDA_REG, 3)}")
    elif REG_TYPE == "RWL1":
      run = wandb.init(project = f"prunin/{REG_TYPE}", name=f"{MODE[:-5]}_lambda{LAMBDA_REG}_K{K}_eps{EPSILON}")
    train_size = train_image_tensor.shape[0]
    num_batches = round(train_size/BATCH_SIZE)

    for epoch in range(EPOCHS):
        total_train_loss= 0
        # shuffle training tensor for every epoch
        indices = torch.randperm(train_size)
        train_image_tensor = train_image_tensor[indices]
        train_label_tensor = train_label_tensor[indices]
        for i in range(0, train_size, BATCH_SIZE):
          optimizer.zero_grad(set_to_none=True)
          outputs = model(train_image_tensor[i:i+BATCH_SIZE])
          data_loss = criterion(input = outputs, target=train_label_tensor[i:i+BATCH_SIZE])

          match(REG_TYPE):
            case "RWL1":
              reg_loss = LAMBDA_REG * calculate_WL1_norm()
              loss = data_loss + reg_loss
            case "L1":
              reg_loss = LAMBDA_REG * calculate_L1_norm()
              loss = data_loss + reg_loss
            case "L0":
              loss = data_loss

          loss.backward()
          total_train_loss += loss.item()
          optimizer.step()

          match(REG_TYPE):
            case "RWL1":
              if (i % K == 0):
                model.apply(L1_penalty_update)


        match(REG_TYPE):
          case "L0":
              model.apply(prox_op_layerwise)

        avg_train_loss = total_train_loss / num_batches
        current_val_loss, val_acc = validate_in_memory()
        current_sparsity = calculate_sparsity()

        run.log({"train/loss": avg_train_loss, "val/loss": current_val_loss, "val/acc": val_acc, "sparsity": current_sparsity})

        # early stopping
        if current_val_loss < best_val_loss: # If validation loss improved
            best_val_loss = current_val_loss
            best_params = deepcopy(model.state_dict())
            current_patience = PATIENCE
        else:
            current_patience -= 1

        if current_patience == 0:
            break


    if best_params is not None:
      model.load_state_dict(best_params)
      
    return model


def validate_in_memory(model, val_image_tensor, val_label_tensor):
    criterion = torch.nn.CrossEntropyLoss()
    total = val_label_tensor.shape[0]
    num_batches = round(total/BATCH_SIZE)
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
    criterion = torch.nn.CrossEntropyLoss()
    total = test_label_tensor.shape[0]
    num_batches = round(total / BATCH_SIZE)
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
            filter, depth, height, width = weight.shape
            with torch.no_grad():
                match(MODE):
                    case "weight-wise":
                        weight_amplitude = torch.abs(weight)
                        threshold = torch.quantile(weight_amplitude, q=COMPRESS_RATIO)
                        layer.mask = torch.tensor(weight_amplitude >= threshold, dtype=torch.float32)
                        weight[weight_amplitude < threshold] = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=False)
                    case "kernel-wise":
                        kernel_amplitude = weight.pow(2).sum(dim=(2, 3)).sqrt()
                        threshold = torch.quantile(kernel_amplitude, q=COMPRESS_RATIO)
                        layer.mask = torch.tensor(kernel_amplitude >= threshold, dtype=torch.float32)
                        weight[kernel_amplitude < threshold] = torch.zeros(height, width, dtype=torch.float32, device=device, requires_grad=False)

                    case "channel-wise":
                        channel_amplitude = weight.pow(2).sum(dim=(1, 2, 3)).sqrt()
                        threshold = torch.quantile(channel_amplitude, q=COMPRESS_RATIO)
                        layer.mask = torch.tensor(channel_amplitude >= threshold, dtype=torch.float32)
                        weight[channel_amplitude < threshold] = torch.zeros(depth, height, width, dtype=torch.float32, device=device, requires_grad=False)

        elif name != "out":
            weight = layer.weight
            with torch.no_grad():
                weight_amplitude  = torch.abs(weight) 
                threshold = torch.quantile(weight_amplitude, q=COMPRESS_RATIO)
                layer.mask = torch.tensor(weight_amplitude >= threshold, dtype=torch.float32)

def update_mask(model):
    for name, layer in model.named_modules():
        if type(layer) == nn.Conv2d:
            weight = layer.weight
            filter, depth, height, width = weight.shape
            with torch.no_grad():
                match(MODE):
                    case "weight-wise":
                        weight_amplitude = torch.abs(weight)
                        threshold = torch.quantile(weight_amplitude, q=COMPRESS_RATIO)
                        layer.mask = torch.tensor(weight_amplitude >= threshold, dtype=torch.float32)
                        weight[weight_amplitude < threshold] = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=False)
                    case "kernel-wise":
                        kernel_amplitude = weight.pow(2).sum(dim=(2, 3)).sqrt()
                        threshold = torch.quantile(kernel_amplitude, q=COMPRESS_RATIO)
                        layer.mask = torch.tensor(kernel_amplitude >= threshold, dtype=torch.float32)
                        weight[kernel_amplitude < threshold] = torch.zeros(height, width, dtype=torch.float32, device=device, requires_grad=False)

                    case "channel-wise":
                        channel_amplitude = weight.pow(2).sum(dim=(1, 2, 3)).sqrt()
                        threshold = torch.quantile(channel_amplitude, q=COMPRESS_RATIO)
                        layer.mask = torch.tensor(channel_amplitude >= threshold, dtype=torch.float32)
                        weight[channel_amplitude < threshold] = torch.zeros(depth, height, width, dtype=torch.float32, device=device, requires_grad=False)

        elif name != "out":
            weight = layer.weight
            with torch.no_grad():
                weight_amplitude  = torch.abs(weight) 
                threshold = torch.quantile(weight_amplitude, q=COMPRESS_RATIO)
                layer.mask = torch.tensor(weight_amplitude >= threshold, dtype=torch.float32)
