import torch 
from torch import nn 
from .hook import Hook
# Reference: https://arxiv.org/abs/1610.02391
def get_heatmap(
    model: nn.Module, input: torch.Tensor, target: torch.Tensor, layer: nn.Module
) -> torch.Tensor:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()

    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)

    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)

    with torch.cuda.amp.autocast(), torch.autograd.set_detect_anomaly(True), torch.set_grad_enabled(True), Hook(layer) as hook:
        # Do a forward and backward pass.
        output = model(input)
        output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()

        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        heatmap = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        heatmap = torch.clamp(heatmap, min=0)
        # Normalize the heatmap
        heatmap /= torch.max(heatmap)

    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])

    return heatmap
