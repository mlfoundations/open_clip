import torch
from PIL import Image
from open_clip.factory import get_tokenizer
import pytest
import open_clip
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


if hasattr(torch._C, '_jit_set_profiling_executor'):
    # legacy executor is too slow to compile large models for unit tests
    # no need for the fusion performance here
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(False)


test_simple_models = [
    # model, pretrained, jit, force_custom_text
    ("ViT-B-32", "laion2b_s34b_b79k", False, False),
    # TODO: Add test coca
]


@pytest.mark.parametrize("model_type,pretrained,jit,force_custom_text", test_simple_models)
def test_inference_simple(
        model_type,
        pretrained,
        jit,
        force_custom_text,
):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_type,
        pretrained=pretrained,
        jit=jit,
        force_custom_text=force_custom_text,
    )
    tokenizer = get_tokenizer(model_type)

    current_dir = os.path.dirname(os.path.realpath(__file__))

    image = preprocess(Image.open(current_dir + "/../docs/CLIP.png")).unsqueeze(0)
    text = tokenizer(["a diagram", "a dog", "a cat"])

    with torch.no_grad():
        image_result = model.encode_image(image, output_hidden_states=True)
        text_result = model.encode_text(text, output_hidden_states=True)

        image_features, image_hidden_states = image_result
        text_features, text_hidden_states = text_result

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # shape of hidden states: [bs, n_hidden_states, 1, seq_len, hidden_size]
        # take first elem
        print(f"Length of text hidden states: {len(text_hidden_states[0])}")
        print(f"Length of image hidden states: {len(image_hidden_states)}")

        # TODO: Write hidden state shapes assertions

    assert text_probs.cpu().numpy()[0].tolist() == [1.0, 0.0, 0.0]
