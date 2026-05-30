import torch
import torch.nn as nn
import wandb
from copy import deepcopy
import math
import numpy

import pandas
from pathlib import Path

import irwl1.config as config
from irwl1.regularization import calculate_L1_norm, calculate_WL1_norm, L1_penalty_update
from torch.func import jacrev, vmap
from irwl1.model import ResNet20
from irwl1.regularization import L1_penalty_init
import h5py

def _should_skip_module(name):
    return any(part in {"out", "fc", "downsample"} for part in name.split("."))

def configure_evaluation() -> None:
	config.WEIGHT_PRUNING_THRESHOLD = 1e-5
	config.EPSILON = 1e-6
	config.UPDATE_INTERVAL = 1
	config.MODE = "weight-wise"
	config.REG_TYPE = "WL1"
	config.MODEL = "ResNet20"
	config.WANDB_MODE = "online"
	config.WEIGHT_DECAY = 0.0


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
	model = ResNet20().to(device)
	L1_penalty_init(model)
	init_mask(model)
	checkpoint = torch.load(checkpoint_path, map_location=device)
	state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
	model.load_state_dict(state_dict)
	return model, checkpoint if isinstance(checkpoint, dict) else {}

import torch
import torch

def log_cross_section_gradient_variance(model, history_dict):
    """
    Calculates the gradient variance for the bottom 10% of NONZERO weights.
    Run this immediately AFTER loss.backward() and BEFORE optimizer.step()
    """
    target_layers = history_dict.keys()
    
    with torch.no_grad():
        for name, module in model.named_modules():
            if name in target_layers:
                if module.weight is not None and module.weight.grad is not None:
                    w = module.weight.detach()
                    g = module.weight.grad.detach()
                    
                    w_abs = torch.abs(w).view(-1)
                    g_flat = g.view(-1)
                    
                    # 1. Isolate the nonzero weights 
                    # (Using > 1e-12 instead of > 0 to safely handle floating-point underflow)
                    nonzero_w_abs = w_abs[w_abs > 1e-12]
                    
                    # 2. Ensure there are enough active weights to calculate a meaningful quantile
                    if len(nonzero_w_abs) > 1:
                        threshold = torch.quantile(nonzero_w_abs.to(torch.float32), 0.10)
                        
                        # 3. Create a compound mask: strictly active AND below the threshold
                        bottom_10_mask = (w_abs > 1e-12) & (w_abs <= threshold)
                        bottom_10_grads = g_flat[bottom_10_mask]
                        
                        # 4. Calculate variance
                        if len(bottom_10_grads) > 1:
                            variance = torch.var(bottom_10_grads).item()
                            history_dict[name].append(variance)


def compute_layer_condition_numbers(model, zero_threshold=1e-5):
    condition_numbers = {}
    
    layer_idx = 0
    # Ensure no gradients are tracked for this diagnostic
    with torch.no_grad():
        for name, module in model.named_modules():
            if _should_skip_module(name):
                continue
            # Target only layers with learnable weight matrices
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                
                w = module.weight.detach()
                
                # Unfold 4D Conv weights into 2D matrices
                if isinstance(module, torch.nn.Conv2d):
                    w = w.view(w.shape[0], -1) 
                    
                # Calculate singular values (returns them sorted descending)
                singular_values = torch.linalg.svdvals(w)
                
                # Isolate the active spectrum to handle sparsity zeros
                active_svs = singular_values[singular_values > zero_threshold]
                
                if len(active_svs) > 0:
                    sigma_max = active_svs[0].item()
                    sigma_min = active_svs[-1].item()
                    condition_numbers[f"layer_{layer_idx}"] = sigma_max / sigma_min
                else:
                    # Matrix is entirely pruned or dead
                    condition_numbers[f"layer_{layer_idx}"] = float('inf') 

                layer_idx += 1
                    
    return condition_numbers
    

def compute_jacobian_norm(model, test_loader, device='cuda'):
    model.eval()
    total_norm = 0.0
    num_samples = 0
    
    # We define a wrapper function for the model that takes a single input 
    # and returns the logits. jacrev expects a function.
    def fnet_single(x):
        return model(x.unsqueeze(0)).squeeze(0)
    
    # vmap vectorizes the jacobian computation across the batch dimension
    compute_batch_jacobian = vmap(jacrev(fnet_single))

    # Disable gradient tracking for the weights, we only need it for inputs
    with torch.no_grad(): 
        for images, _ in test_loader:
            images = images.to(device)
            
            # compute_batch_jacobian requires inputs to have requires_grad=False 
            # in the outer context, as functorch handles the internal autodiff.
            
            # J shape: (Batch, 10, 3, 32, 32)
            J = compute_batch_jacobian(images)
            
            # Flatten the spatial/channel dimensions: (Batch, 10, 3072)
            J_flat = J.view(J.shape[0], J.shape[1], -1)
            
            # Compute Frobenius norm for each sample in the batch
            # norm shape: (Batch,)
            frob_norms = torch.linalg.matrix_norm(J_flat, ord='fro')
            
            total_norm += frob_norms.sum().item()
            num_samples += images.size(0)
            
    # Return the average Jacobian Frobenius norm across the dataset
    return total_norm / num_samples


def append_weights_to_hdf5(model, filepath, epsilon, lambd, layer_name='layer2.0.conv1'):
    """
    Extracts mid-layer weights and appends them to an HDF5 file.
    The file is structured hierarchically: epsilon -> lambda -> step_X.
    """
    # 1. Extract the raw weights
    with torch.no_grad():
        for name, module in model.named_modules():
            if name == layer_name:
                weights = module.weight.detach().cpu().numpy().flatten()
                break
        else:
            raise ValueError(f"Layer '{layer_name}' not found in the model.")

    # 2. Open HDF5 file in 'append' mode ('a' creates it if it doesn't exist)
    with h5py.File(filepath, 'a') as f:
        # Create a clean string path for the group (e.g., "eps_1e-06/lam_0.01")
        # We format epsilon in scientific notation to avoid messy strings
        group_path = f"eps_{epsilon:.1e}/lam_{lambd:.4f}"
        
        # 3. Get or create the hierarchical group
        if group_path in f:
            grp = f[group_path]
        else:
            grp = f.create_group(group_path)
            
        # 4. Determine the next step index by counting existing datasets
        step_idx = len(grp.keys())
        
        # 5. Save the numpy array directly into the file
        grp.create_dataset(f"step_{step_idx}", data=weights)


def calculate_thresholded_sparsity(model):
    match(config.MODE):
        case "weight-wise":
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

        case "kernel-wise":
            threhold = config.KERNEL_PRUNING_THRESHOLD
            num_total_kernels, num_zero_kernels = 0, 0
            for name, layer in model.named_modules():
                if _should_skip_module(name):
                    continue

                if type(layer) == nn.Conv2d:
                    kernel_amplitude = torch.sqrt(torch.pow(layer.weight, 2).sum(dim=[2, 3]))
                    num_total_kernels += kernel_amplitude.numel()
                    num_zero_kernels += torch.sum(kernel_amplitude < threhold).item()
            if num_total_kernels == 0:
                return 0.0
            return num_zero_kernels / num_total_kernels * 100  

        case "channel-wise":
            threhold = config.CHANNEL_PRUNING_THRESHOLD
            num_total_channels, num_zero_channels = 0, 0
            for name, layer in model.named_modules():
                if _should_skip_module(name):
                    continue

                if type(layer) == nn.Conv2d:
                    channel_amplitude = torch.sqrt(torch.pow(layer.weight, 2).sum(dim=[1, 2, 3]))
                    num_total_channels += channel_amplitude.numel()
                    num_zero_channels += torch.sum(channel_amplitude < threhold).item()
            if num_total_channels == 0:
                return 0.0 
            return num_zero_channels / num_total_channels * 100


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


def _regularization_update_batches(num_batches, num_updates):
    update_batches = set()

    for update_index in range(1, num_updates + 1):
        batch_number = math.ceil(update_index * num_batches / (num_updates + 1))
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
        return wandb.init(project=f"pruning_{config.MODEL}_{config.REG_TYPE}", name=f"{config.MODE[:-5]}_lambda{round(config.LAMBDA_REG_END, 3)}", mode=config.WANDB_MODE)
    if reg_type == "WL1":
        return wandb.init(project=f"pruning_{config.MODEL}_{config.REG_TYPE}", name=f"{config.MODE[:-5]}_threshold{config.WEIGHT_PRUNING_THRESHOLD}_updateint{config.UPDATE_INTERVAL}_eps{config.EPSILON}", mode=config.WANDB_MODE)
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


def _mask_for_weight(weight, mask):
    if config.MODE == "weight-wise":
        return mask.to(weight.dtype)

    if config.MODE == "kernel-wise":
        return mask.to(weight.dtype)[:, :, None, None]

    if config.MODE == "channel-wise":
        return mask.to(weight.dtype)[:, None, None, None]

    return mask.to(weight.dtype)


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
                layer.weight.mul_(_mask_for_weight(layer.weight, layer.mask))

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

    epoch_variance_history = {
    'conv1': [], 
    'layer2.0.conv1': [], 
    'layer3.2.conv2': []
}

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
    update_interval = max(float(config.UPDATE_INTERVAL), 1e-12)
    epoch_update_period = update_interval if update_interval > 1 else None
    batch_update_count = max(1, math.ceil(1.0 / update_interval)) if update_interval <= 1 else 0
    update_batches = _regularization_update_batches(total_num_batches, batch_update_count) if batch_update_count else set()
    next_epoch_update = epoch_update_period if epoch_update_period is not None else None

    for epoch in range(config.MAX_EPOCHS):
        model.train()
        # advance lambda once per epoch (slower growth)
        if config.IS_LAMBDA_RISE:
            lambda_update_count += 1
        total_train_loss = 0

        batch_variances = {layer: [] for layer in epoch_variance_history.keys()}
        for batch_index, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            outputs = model(inputs)
            data_loss = criterion(input=outputs, target=targets)

            match(config.REG_TYPE):
                case "WL1":
                    batch_number = batch_index + 1

                    if update_interval <= 1 and (batch_number in update_batches):
                        if config.IS_EPSILON_DECAY:
                            epsilon_update_count += 1
                            current_epsilon = _advance_epsilon(epsilon_update_count, epsilon_schedule_fn)
                        else:
                            current_epsilon = config.EPSILON_END

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

            log_cross_section_gradient_variance(model, batch_variances)  # Call the logging function here
            total_train_loss += data_loss.item()
            optimizer.step()

        #if (epoch % 5 == 0):
        #    append_weights_to_hdf5(model, "results/waterfall_weight.h5", current_epsilon, current_lambda_reg)

        if (next_epoch_update is not None) and ((epoch + 1) >= next_epoch_update - 1e-12):
            if config.IS_EPSILON_DECAY:
                epsilon_update_count += 1
                current_epsilon = _advance_epsilon(epsilon_update_count, epsilon_schedule_fn)
            else:
                current_epsilon = config.EPSILON_END

            L1_penalty_update(model, current_epsilon)
            next_epoch_update += update_interval


        avg_train_loss = total_train_loss / total_num_batches
        current_val_loss, val_acc = validate(model, val_loader)
        current_thresholded_sparsity = calculate_thresholded_sparsity(model)
        current_real_sparsity = calculate_real_sparsity(model)
        sparsity_delta = None if previous_thresholded_sparsity is None else abs(current_thresholded_sparsity - previous_thresholded_sparsity)

        for layer_name in epoch_variance_history.keys():
            if len(batch_variances[layer_name]) > 0:
                avg_epoch_var = numpy.mean(batch_variances[layer_name])
                epoch_variance_history[layer_name].append(avg_epoch_var)

        if run is not None:
            run.log({
                "train/loss": avg_train_loss,
                "val/loss": current_val_loss,
                "val/acc": val_acc,
                "train/thresholded_sparsity": current_thresholded_sparsity,
                "train/real_sparsity": current_real_sparsity,
                "reg/epsilon": current_epsilon,
                "reg/lambda_reg": current_lambda_reg,
                "train/early_layer_variance": epoch_variance_history['conv1'][-1] if len(epoch_variance_history['conv1']) > 0 else None,
                "train/middle_layer_variance": epoch_variance_history['layer2.0.conv1'][-1] if len(epoch_variance_history['layer2.0.conv1']) > 0 else None,
                "train/end_layer_variance": epoch_variance_history['layer3.2.conv2'][-1] if len(epoch_variance_history['layer3.2.conv2']) > 0 else None,
            })
        if config.IS_LAMBDA_RISE:
            current_lambda_reg = _advance_lambda(lambda_update_count, lambda_schedule_fn)
        else:
            current_lambda_reg = config.LAMBDA_REG_END

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
    epoch_variance_history = {
        'conv1': [],
        'layer2.0.conv1': [],
        'layer3.2.conv2': []   
    }
    if is_new_run:
        run = _init_wandb_run(config.REG_TYPE, run_name, is_new_run)

    total_num_batches = len(train_loader)
    rewind_epoch = config.REWIND_EPOCH if rewind_epoch is None else rewind_epoch
    rewind_checkpoint_saved = False

    for epoch in range(config.MAX_EPOCHS):
        model.train()
        total_train_loss = 0

        batch_variances = {layer: [] for layer in epoch_variance_history.keys()}
        for inputs, targets in train_loader:
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            outputs = model(inputs)
            loss = criterion(input=outputs, target=targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            log_cross_section_gradient_variance(model, batch_variances)
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


        for layer_name in epoch_variance_history.keys():
            if len(batch_variances[layer_name]) > 0:
                avg_epoch_var = numpy.mean(batch_variances[layer_name])
                epoch_variance_history[layer_name].append(avg_epoch_var)

        if run is not None:
            run.log({
                "train/loss": avg_train_loss,
                "val/loss": current_val_loss,
                "val/acc": val_acc,
                "train/thresholded_sparsity": current_thresholded_sparsity,
                "train/real_sparsity": current_real_sparsity,
                "reg/epsilon": 0,
                "reg/lambda_reg": 0,
                "train/early_layer_variance": epoch_variance_history['conv1'][-1] if len(epoch_variance_history['conv1']) > 0 else None,
                "train/middle_layer_variance": epoch_variance_history['layer2.0.conv1'][-1] if len(epoch_variance_history['layer2.0.conv1']) > 0 else None,
                "train/end_layer_variance": epoch_variance_history['layer3.2.conv2'][-1] if len(epoch_variance_history['layer3.2.conv2']) > 0 else None,
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


def _regularization_parameters(model):
    regularization_parameters = []

    for name, layer in model.named_modules():
        if _should_skip_module(name):
            continue

        if type(layer) in [nn.Conv2d, nn.Linear]:
            weight = getattr(layer, "weight", None)
            if weight is not None and weight.requires_grad:
                regularization_parameters.append(weight)

    return regularization_parameters


def _effective_v2_lambda(current_lambda_reg, lambda_floor):
    lambda_floor = 1e-3 if lambda_floor is None else lambda_floor
    return max(current_lambda_reg, lambda_floor)


def _apply_regularization_update(model, optimizer, regularization_loss):
    regularization_parameters = _regularization_parameters(model)
    if not regularization_parameters:
        return

    param_lrs = {
        id(param): group.get("lr", config.LEARNING_RATE)
        for group in optimizer.param_groups
        for param in group["params"]
    }
    regularization_grads = torch.autograd.grad(
        regularization_loss,
        regularization_parameters,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    with torch.no_grad():
        for param, grad in zip(regularization_parameters, regularization_grads):
            if grad is None:
                continue

            param.add_(grad, alpha=-param_lrs.get(id(param), config.LEARNING_RATE))


def train_regularized_v2(
    model,
    train_loader,
    val_loader,
    optimizer=None,
    is_new_run=True,
    run=None,
    run_name="default",
    weight_decay=False,
    sparsity_delta_threshold=None,
    sparsity_patience=None,
    epsilon_schedule_fn=None,
    lambda_schedule_fn=None,
    lambda_floor=1e-3,
):

    criterion = torch.nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer_kwargs = {"lr": config.LEARNING_RATE}
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    else:
        for group in optimizer.param_groups:
            group["weight_decay"] = 0.0

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
    update_interval = max(float(config.UPDATE_INTERVAL), 1e-12)
    epoch_update_period = update_interval if update_interval > 1 else None
    batch_update_count = max(1, math.ceil(1.0 / update_interval)) if update_interval <= 1 else 0
    update_batches = _regularization_update_batches(total_num_batches, batch_update_count) if batch_update_count else set()
    next_epoch_update = epoch_update_period if epoch_update_period is not None else None

    for epoch in range(config.MAX_EPOCHS):
        model.train()
        if config.IS_LAMBDA_RISE:
            lambda_update_count += 1

        total_train_loss = 0

        for batch_index, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            outputs = model(inputs)
            data_loss = criterion(input=outputs, target=targets)

            optimizer.zero_grad(set_to_none=True)
            data_loss.backward()
            total_train_loss += data_loss.item()
            optimizer.step()

            if config.REG_TYPE == "WL1":
                batch_number = batch_index + 1
                if update_interval <= 1 and (batch_number in update_batches):
                    if config.IS_EPSILON_DECAY:
                        epsilon_update_count += 1
                        current_epsilon = _advance_epsilon(epsilon_update_count, epsilon_schedule_fn)
                    else:
                        current_epsilon = config.EPSILON_END

                    L1_penalty_update(model, current_epsilon)

                effective_lambda_reg = _effective_v2_lambda(current_lambda_reg, lambda_floor)
                regularization_loss = effective_lambda_reg * calculate_WL1_norm(model)

                optimizer.zero_grad(set_to_none=True)
                _apply_regularization_update(model, optimizer, regularization_loss)

            elif config.REG_TYPE == "L1":
                effective_lambda_reg = _effective_v2_lambda(current_lambda_reg, lambda_floor)
                regularization_loss = effective_lambda_reg * calculate_L1_norm(model)

                optimizer.zero_grad(set_to_none=True)
                _apply_regularization_update(model, optimizer, regularization_loss)

        if (next_epoch_update is not None) and ((epoch + 1) >= next_epoch_update - 1e-12):
            if config.IS_EPSILON_DECAY:
                epsilon_update_count += 1
                current_epsilon = _advance_epsilon(epsilon_update_count, epsilon_schedule_fn)
            else:
                current_epsilon = config.EPSILON_END

            L1_penalty_update(model, current_epsilon)
            next_epoch_update += update_interval

        if config.IS_LAMBDA_RISE:
            current_lambda_reg = _advance_lambda(lambda_update_count, lambda_schedule_fn)
        else:
            current_lambda_reg = config.LAMBDA_REG_END

        avg_train_loss = total_train_loss / max(1, total_num_batches)
        current_val_loss, val_acc = validate(model, val_loader)
        current_thresholded_sparsity = calculate_thresholded_sparsity(model)
        current_real_sparsity = calculate_real_sparsity(model)

        if run is not None:
            run.log({
                "train/loss": avg_train_loss,
                "val/loss": current_val_loss,
                "val/acc": val_acc,
                "train/thresholded_sparsity": current_thresholded_sparsity,
                "train/real_sparsity": current_real_sparsity,
                "reg/epsilon": current_epsilon,
                "reg/lambda_reg": _effective_v2_lambda(current_lambda_reg, lambda_floor),
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
                            layer.mask.copy_((kernel_amplitude >= threshold).to(layer.weight.dtype))

                    case "channel-wise":
                        threshold = config.CHANNEL_PRUNING_THRESHOLD
                        channel_amplitude = torch.sqrt(torch.pow(layer.weight, 2).sum(dim=[1, 2, 3]))
                        pruned_weight = layer.weight * (channel_amplitude >= threshold).to(layer.weight.dtype)[:, None, None, None]
                        layer.weight.copy_(pruned_weight)
                        if masking:
                            layer.mask.copy_((channel_amplitude >= threshold).to(layer.weight.dtype))

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
            "update_interval": [config.UPDATE_INTERVAL] * num_records, "epsilon": [config.EPSILON] * num_records,
             "weight_decay": [config.WEIGHT_DECAY] * num_records
    }

    df = pandas.concat([df, pandas.DataFrame(row)], ignore_index=True)

    df.to_csv(path, index=False)


