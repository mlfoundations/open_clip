# https://colab.research.google.com/github/kevinzakka/clip_playground/blob/main/CLIP_GradCAM_Visualization.ipynb

import open_clip
from PIL import Image
from .heatmap import get_heatmap
from .utils import show_attention_map

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
            getattr(model.visual, "layer4"),
        )
    heatmap = heatmap.squeeze().detach().cpu().numpy()
    show_attention_map(heatmap, image_name)




