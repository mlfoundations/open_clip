# https://colab.research.google.com/github/kevinzakka/clip_playground/blob/main/CLIP_GradCAM_Visualization.ipynb

import open_clip
from open_clip.modified_resnet import ModifiedResNet
from open_clip.timm_model import TimmModel
from open_clip.transformer import VisionTransformer
from PIL import Image

from .heatmap import get_heatmap
from .utils import show_attention_map


def get_layer(model):
    if isinstance(model.visual, ModifiedResNet):
        return model.visual.layer4
    if isinstance(model.visual, TimmModel):
        return model.visual.trunk.stages[-1]
    if isinstance(model.visual, VisionTransformer):
        return model.visual.transformer.resblocks[-2].ls_2
    return None


def grad_cam(model_name, pretrain_tag, image_name, caption_text):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrain_tag
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    image = preprocess(Image.open(image_name)).unsqueeze(0)
    caption = tokenizer([caption_text])

    heatmap = get_heatmap(
        model.visual,
        image,
        model.encode_text(caption).float(),
        get_layer(model),
    )
    heatmap = heatmap.squeeze().detach().cpu().numpy()
    show_attention_map(heatmap, image_name)
