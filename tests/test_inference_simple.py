
import torch
from PIL import Image
from open_clip.factory import get_tokenizer
import pytest
import open_clip
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

@pytest.mark.parametrize("model_type,pretrained", [("ViT-B-32-quickgelu", "laion400m_e32"), ("roberta-ViT-B-32", "laion2b_s12b_b32k")])
def test_inference_simple(model_type, pretrained):
    model, _, preprocess = open_clip.create_model_and_transforms(model_type, pretrained=pretrained, jit=False)
    tokenizer = get_tokenizer(model_type)
    tokenizer = get_tokenizer(model_type)

    current_dir = os.path.dirname(os.path.realpath(__file__))

    image = preprocess(Image.open(current_dir + "/../docs/CLIP.png")).unsqueeze(0)
    text = tokenizer(["a diagram", "a dog", "a cat"])

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    assert text_probs.cpu().numpy()[0].tolist() == [1.0, 0.0, 0.0]
    

@pytest.mark.parametrize("model_type,pretrained", [("ViT-B-32-quickgelu", "laion400m_e32"), ("roberta-ViT-B-32", "laion2b_s12b_b32k")])
def test_output_tokens(model_type, pretrained):
    model, _, preprocess = open_clip.create_model_and_transforms(model_type, pretrained=pretrained, jit=False)
    tokenizer = get_tokenizer(model_type)
    viz_transformer = model.visual
    text_transformer = model.text if hasattr(model, "text") else None
    tokenizer = get_tokenizer(model_type)
    
    current_dir = os.path.dirname(os.path.realpath(__file__))

    image = preprocess(Image.open(current_dir + "/../docs/CLIP.png")).unsqueeze(0)
    text = tokenizer(["a diagram", "a dog", "a cat"])

    with torch.no_grad():
        if text_transformer is not None:
            text_output = text_transformer(text)
            text_output_tokens = text_transformer(text, output_tokens=True)
        viz_output = viz_transformer(image)
        viz_output_tokens = viz_transformer(image, output_tokens=True)

    if text_transformer is not None:
        assert (text_output == text_output_tokens[0]).all()
    assert (viz_output == viz_output_tokens[0]).all()

@pytest.mark.parametrize("model_type,pretrained", [("ViT-B-32-quickgelu", "laion400m_e32"), ("roberta-ViT-B-32", "laion2b_s12b_b32k")])
def test_output_dict(model_type, pretrained):
    model, _, preprocess = open_clip.create_model_and_transforms(model_type, pretrained=pretrained, jit=False)
    tokenizer = get_tokenizer(model_type)
    tokenizer = get_tokenizer(model_type)
    loss = (
        open_clip.factory.CoCaLoss()
        if model_type.startswith("coca") 
        else open_clip.factory.ClipLoss()
    )
    current_dir = os.path.dirname(os.path.realpath(__file__))

    image = preprocess(Image.open(current_dir + "/../docs/CLIP.png")).unsqueeze(0)
    text = tokenizer(["a diagram"])

    with torch.no_grad():
        image_features, text_features, logit_scale = model(image, text)
        output_dict = model(image, text, output_dict=True)
        loss_value = loss(image_features, text_features, logit_scale)
        loss_value_dict = loss(**output_dict, output_dict=True)
    
    assert (image_features == output_dict["image_features"]).all()
    assert (text_features == output_dict["text_features"]).all()
    assert (logit_scale == output_dict["logit_scale"]).all()
    assert sum(loss_value_dict.values()) == loss_value
        
