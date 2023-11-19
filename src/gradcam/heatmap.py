import torch
from open_clip.transformer import VisionTransformer
from torch import nn

from .hook import Hook


# https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md
def reshape_transform(tensor, height=14, width=14):
    tensor.squeeze()
    result = tensor[1:, :].reshape(tensor.size(1), height, width, tensor.size(2))

    # Bring the channels to the first dimension, like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def get_gradient(model, hook):
    if isinstance(model.visual, VisionTransformer):
        return reshape_transform(hook.gradient.float())
    return hook.gradient.float()


def get_activation(model, hook):
    if isinstance(model.visual, VisionTransformer):
        return reshape_transform(hook.activation.float())
    return hook.activation.float()


# https://arxiv.org/abs/1610.02391
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

    with torch.cuda.amp.autocast(), torch.autograd.set_detect_anomaly(
        True
    ), torch.set_grad_enabled(True), Hook(layer) as hook:
        # Do a forward and backward pass.
        output = model.visual(input)
        output.backward(model.encode_text(target))

        # image_features = model.encode_image(input)
        # text_features = model.encode_text(target)

        # normalized_image_features = image_features / torch.linalg.norm(
        #     image_features, dim=-1, keepdim=True
        # )
        # normalized_text_features = text_features / torch.linalg.norm(
        #     text_features, dim=-1, keepdim=True
        # )
        # text_probs = 100.0 * normalized_image_features @ normalized_text_features.T
        # text_probs[:, 0].backward()

        grad = get_gradient(model, hook)
        act = get_activation(model, hook)

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
