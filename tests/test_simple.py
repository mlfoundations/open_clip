
import torch
from PIL import Image
from open_clip import tokenizer
import open_clip
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def test_inference():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')

    current_dir = os.path.dirname(os.path.realpath(__file__))

    image = preprocess(Image.open(current_dir + "/../docs/CLIP.png")).unsqueeze(0)
    text = tokenizer.tokenize(["a diagram", "a dog", "a cat"])

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    assert text_probs.cpu().numpy()[0].tolist() == [1.0, 0.0, 0.0]