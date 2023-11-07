# https://colab.research.google.com/github/kevinzakka/clip_playground/blob/main/CLIP_GradCAM_Visualization.ipynb

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import torch.nn as nn
from PIL import Image

model_name = "RN50"
pretrain_tag = "openai"
image_name = "test.jpg"
caption_text = "a cat"

model, _, preprocess = open_clip.create_model_and_transforms(
    model_name, pretrained=pretrain_tag
)
tokenizer = open_clip.get_tokenizer(model_name)

image = preprocess(Image.open(image_name)).unsqueeze(0)
caption = tokenizer([caption_text])


def load_image(img_path, resize=None):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.0


def show_attention_map(heatmap):
    _, axes = plt.subplots(1, 3, figsize=(15, 8))
    axes[0].matshow(heatmap.squeeze())
    img = cv2.imread(image_name)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    result_img = (heatmap * 0.4 + img).astype(np.float32)
    axes[1].imshow(img[..., ::-1])
    axes[2].imshow((result_img / 255)[..., ::-1])
    for ax in axes:
        ax.axis("off")
    plt.show()
    # cv2.imwrite("./map.jpg", superimposed_img)


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


with torch.cuda.amp.autocast(), torch.autograd.set_detect_anomaly(
    True
), torch.set_grad_enabled(True):
    image_np = load_image(image_name, model.visual.image_size)
    heatmap = get_heatmap(
        model.visual,
        image,
        model.encode_text(caption).float(),
        getattr(model.visual, "layer4"),
    )
    heatmap = heatmap.squeeze().detach().cpu().numpy()
    show_attention_map(heatmap)
