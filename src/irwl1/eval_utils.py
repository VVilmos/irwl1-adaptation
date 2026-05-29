from __future__ import annotations

from pathlib import Path

import torch
import os
import h5py

import irwl1.config as config
from irwl1.model import ResNet20
from irwl1.regularization import L1_penalty_init
from irwl1.utils import init_mask, _should_skip_module


def configure_evaluation() -> None:
	config.WEIGHT_PRUNING_THRESHOLD = 1e-5
	config.EPSILON = 1e-6
	config.UPDATE_INTERVAL = 2
	config.MODE = "weight-wise"
	config.REG_TYPE = "WL1"
	config.MODEL = "ResNet20"
	config.WANDB_MODE = "offline"
	config.FAB_STEPS = 10
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
from torch.func import jacrev, vmap
import torch

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
