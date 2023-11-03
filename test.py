import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('convnext_base', pretrained='laion400m_s13b_b51k')
tokenizer = open_clip.get_tokenizer('convnext_base')

image = preprocess(Image.open("test.jpg")).unsqueeze(0)
text = tokenizer(["a dog", "a cat",])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

#  open_clip.list_pretrained()