import open_clip
import torch
from PIL import Image

model_name = "RN50x16"
pretrain_tag = "openai"

model, _, preprocess = open_clip.create_model_and_transforms(
    model_name, pretrained=pretrain_tag
)
tokenizer = open_clip.get_tokenizer(model_name)

image = preprocess(Image.open("test.jpg")).unsqueeze(0)
text = tokenizer(
    [
        "a dog",
        "a cat",
    ]
)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

#  open_clip.list_pretrained()
