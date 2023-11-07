# https://colab.research.google.com/github/kevinzakka/clip_playground/blob/main/CLIP_GradCAM_Visualization.ipynb

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import filters

model_name = "convnext_base"
pretrain_tag = "laion400m_s13b_b51k"
image_name = "test.jpg"

model, _, preprocess = open_clip.create_model_and_transforms(
    model_name, pretrained=pretrain_tag
)
tokenizer = open_clip.get_tokenizer(model_name)

image = preprocess(Image.open(image_name)).unsqueeze(0)
caption = tokenizer(["a dog"])


def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x


# Modified from: https://github.com/salesforce/ALBEF/blob/main/visualization.ipynb
def getAttMap(img, attn_map, blur=True):
    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02 * max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap("jet")
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = (
        1 * (1 - attn_map**0.7).reshape(attn_map.shape + (1,)) * img
        + (attn_map**0.7).reshape(attn_map.shape + (1,)) * attn_map_c
    )
    return attn_map


def viz_attn(img, attn_map, blur=True):
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[1].imshow(getAttMap(img, attn_map, blur))
    for ax in axes:
        ax.axis("off")
    plt.show()


def load_image(img_path, resize=None):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize(resize)
    return np.asarray(image).astype(np.float32) / 255.0


class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)

    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()

    @property
    def activation(self) -> torch.Tensor:
        return self.data

    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad


# Reference: https://arxiv.org/abs/1610.02391
def gradCAM(
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
    with Hook(layer) as hook:
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
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(
        gradcam, input.shape[2:], mode="bicubic", align_corners=False
    )

    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])

    return gradcam


with torch.cuda.amp.autocast(), torch.autograd.set_detect_anomaly(
    True
), torch.set_grad_enabled(True):
    # image_features = model.encode_image(image)
    # text_features = model.encode_text(text)
    # image_features /= image_ftexteatures.norm(dim=-1, keepdim=True)
    # text_features /= text_features.norm(dim=-1, keepdim=True)

    # text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    image_np = load_image(image_name, model.visual.image_size)

    attn_map = gradCAM(
        model.visual,
        image,
        model.encode_text(caption).float(),
        model.visual.trunk.stages[-1],
    )
    attn_map = attn_map.squeeze().detach().cpu().numpy()

    viz_attn(image_np, attn_map, True)

#  print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
#  open_clip.list_pretrained()
