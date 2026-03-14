import torch
import torch.nn as nn
import irwl1.config as config

# L0 proximal operator
def prox_op_layerwise(layer):
  if type(layer) in [nn.Conv2d]: # TODO: update with linear layers
    weight = layer.weight
    filter, depth, height, width = weight.shape
    with torch.no_grad():
      match(config.MODE):
        case "weight-wise":
          amp_weight = torch.abs(weight)
          layer.hard_threshold = torch.quantile(amp_weight, config.COMPRESS_RATIO/100)
          weight[amp_weight < layer.hard_threshold] = torch.tensor(0., device=config.DEVICE, dtype=weight.dtype)

        case "kernel-wise":
          kernel_norms = torch.linalg.vector_norm(weight, ord=2, dim=(2, 3))
          layer.hard_threshold = torch.quantile(kernel_norms, config.COMPRESS_RATIO / 100)
          weight[kernel_norms < layer.hard_threshold] = torch.zeros(height, width, device=config.DEVICE)

        case "channel-wise":
          channel_norms = torch.linalg.vector_norm(weight, ord=2, dim=(1, 2,3)) # flattened vector norm of the channel weight tensor
          layer.hard_threshold = torch.quantile(channel_norms, config.COMPRESS_RATIO/100)
          weight[channel_norms < layer.hard_threshold] = torch.zeros(depth, height, width, device=config.DEVICE)

# L1
def calculate_L1_norm(model):
  l1_norm = 0
  for name, layer in model.named_modules():
    if type(layer) in [nn.Conv2d]:
      match(config.MODE):
        case "weight-wise":
          weight_abs = layer.weight.abs()
          if hasattr(layer, "mask"):
            active = layer.mask > 0
            l1_norm += weight_abs[active].sum()
          else:
            l1_norm += weight_abs.sum()
        case "kernel-wise":
          kernel_amplitudes = (layer.weight.pow(2).sum(dim=(2, 3)) + config.DELTA).sqrt()
          if hasattr(layer, "mask"):
            active = layer.mask > 0
            l1_norm += kernel_amplitudes[active].sum()
          else:
            l1_norm += kernel_amplitudes.sum()
        case "channel-wise":
          channel_amplitudes = (layer.weight.pow(2).sum(dim=(1, 2, 3)) + config.DELTA).sqrt()
          if hasattr(layer, "mask"):
            active = layer.mask > 0
            l1_norm += channel_amplitudes[active].sum()
          else:
            l1_norm += channel_amplitudes.sum()
    elif (type(layer) == nn.Linear) and (name != 'out'):
      weight_abs = layer.weight.abs()
      if hasattr(layer, "mask"):
        active = layer.mask > 0
        l1_norm += weight_abs[active].sum()
      else:
        l1_norm += weight_abs.sum()

  return l1_norm


# Reweighted L1
def L1_penalty_init(model):
    for name, layer in model.named_modules():
        if type(layer) == nn.Conv2d:
            match(config.MODE):
              case "weight-wise":
                layer.penalty = torch.ones(layer.weight.shape, device=config.DEVICE, requires_grad=False) # every weight has a l1_weight/alpha
              case "kernel-wise":
                layer.penalty = torch.ones(layer.weight.shape[:2], device=config.DEVICE, requires_grad=False) # every kernel
              case "channel-wise":
                layer.penalty = torch.ones(layer.weight.shape[0], device=config.DEVICE,requires_grad=False)
        elif (type(layer) == nn.Linear) and (name != "out"):
          layer.penalty = torch.ones(layer.weight.shape, device=config.DEVICE, requires_grad=False)

def L1_penalty_update(model):
    for name, layer in model.named_modules():
        if type(layer)  == nn.Conv2d:
          with torch.no_grad():
            match(config.MODE):
              case "weight-wise":
                layer.penalty = 1 / (layer.weight.abs() + config.EPSILON)
              case "kernel-wise":
                layer.penalty = 1 / (layer.weight.pow(2).sum(dim=(2, 3)).sqrt() + config.EPSILON)
              case "channel-wise":
                layer.penalty = 1 / (layer.weight.pow(2).sum(dim=(1, 2, 3)).sqrt() + config.EPSILON)
        elif (type(layer) == nn.Linear) and (name != "out"):
          layer.penalty = 1 / (layer.weight.abs() + config.EPSILON)
          if hasattr(layer, "mask"):
            layer.penalty = layer.penalty * (layer.mask > 0)

def calculate_WL1_norm(model):
  l1_norm = 0
  for name, layer in model.named_modules():
    if type(layer)  == nn.Conv2d:
        match(config.MODE):
          case "weight-wise":
            wl1 = layer.penalty * layer.weight.abs()
            if hasattr(layer, "mask"):
              active = layer.mask > 0
              l1_norm += wl1[active].sum()
            else:
              l1_norm += wl1.sum()
          case "kernel-wise":
            wl1 = layer.penalty * (layer.weight.pow(2).sum(dim=(2, 3)) + config.DELTA).sqrt()
            if hasattr(layer, "mask"):
              active = layer.mask > 0
              l1_norm += wl1[active].sum()
            else:
              l1_norm += wl1.sum()
          case "channel-wise":
            wl1 = layer.penalty * (layer.weight.pow(2).sum(dim=(1, 2, 3)) + config.DELTA).sqrt()
            if hasattr(layer, "mask"):
              active = layer.mask > 0
              l1_norm += wl1[active].sum()
            else:
              l1_norm += wl1.sum()
    elif (type(layer) == nn.Linear) and (name != "out"):
      wl1 = layer.penalty * layer.weight.abs()
      if hasattr(layer, "mask"):
        active = layer.mask > 0
        l1_norm += wl1[active].sum()
      else:
        l1_norm += wl1.sum()

  return l1_norm
