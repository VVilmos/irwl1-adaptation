import torch
import torch.nn as nn
from irwl1.config import MODE, COMPRESS_RATIO, DEVICE, EPSILON

# L0 proximal operator
def prox_op_layerwise(layer):
  if type(layer) in [nn.Conv2d]: # TODO: update with linear layers
    weight = layer.weight
    filter, depth, height, width = weight.shape
    with torch.no_grad():
      match(MODE):
        case "weight-wise":
          amp_weight = torch.abs(weight)
          layer.hard_threshold = torch.quantile(amp_weight, COMPRESS_RATIO/100)
          weight[amp_weight < layer.hard_threshold] = torch.tensor(0., device=DEVICE, dtype=weight.dtype)

        case "kernel-wise":
          kernel_norms = torch.linalg.vector_norm(weight, ord=2, dim=(2, 3))
          layer.hard_threshold = torch.quantile(kernel_norms, COMPRESS_RATIO / 100)
          weight[kernel_norms < layer.hard_threshold] = torch.zeros(height, width, device=DEVICE)

        case "channel-wise":
          channel_norms = torch.linalg.vector_norm(weight, ord=2, dim=(1, 2,3)) # flattened vector norm of the channel weight tensor
          layer.hard_threshold = torch.quantile(channel_norms, COMPRESS_RATIO/100)
          weight[channel_norms < layer.hard_threshold] = torch.zeros(depth, height, width, device=DEVICE)


def calculate_L1_norm(model):
  l1_norm = 0
  for layer in model.modules():
    if type(layer) in [nn.Conv2d]:
      match(MODE):
        case "weight-wise":
          l1_norm += layer.weight.abs().sum()
        case "kernel-wise":
          kernel_amplitudes= layer.weight.pow(2).sum(dim=(2, 3)).sqrt()
          l1_norm += kernel_amplitudes.sum()
        case "channel-wise":
          channel_amplitudes = layer.weight.pow(2).sum(dim=(1, 2, 3)).sqrt()
          l1_norm += channel_amplitudes.sum()
    else:
      l1_norm += layer.weight.abs().sum()

  return l1_norm


# Reweighted L1
def L1_penalty_init(layer):
        if type(layer) in [nn.Conv2d]:
            match(MODE):
              case "weight-wise":
                layer.penalty = torch.ones_like(layer.weight, device=DEVICE, requires_grad=False) # every weight has a l1_weight/alpha
              case "kernel-wise":
                layer.penalty = torch.ones(layer.weight.shape[:2], device=DEVICE, requires_grad=False) # every kernel
              case "channel-wise":
                layer.penalty = torch.ones(layer.weight.shape[0], device=DEVICE,requires_grad=False)

def L1_penalty_update(model):
    for name, layer in model.named_modules():
        if type(layer)  == nn.Conv2d:
          with torch.no_grad():
            match(MODE):
              case "weight-wise":
                layer.penalty = 1 / (layer.weight.abs() + EPSILON)
              case "kernel-wise":
                layer.penalty = 1 / (layer.weight.pow(2).sum(dim=(2, 3)).sqrt() + EPSILON)
              case "channel-wise":
                layer.penalty = 1 / (layer.weight.pow(2).sum(dim=(1, 2, 3)).sqrt() + EPSILON)
        elif name != "out":
          layer.penalty = 1 / (layer.weight.abs() + EPSILON)

def calculate_WL1_norm(model):
  l1_norm = 0
  for name, layer in model.named_modules():
    if type(layer)  == nn.Conv2d:
        match(MODE):
          case "weight-wise":
            l1_norm += (layer.penalty * layer.weight.abs()).sum()
          case "kernel-wise":
            l1_norm += (layer.penalty * layer.weight.pow(2).sum(dim=(2, 3)).sqrt()).sum()
          case "channel-wise":
            l1_norm += (layer.penalty * layer.weight.pow(2).sum(dim=(1, 2, 3)).sqrt()).sum()
    elif name != "out":
      l1_norm += (layer.penalty * layer.weight.abs()).sum() 

  return l1_norm
