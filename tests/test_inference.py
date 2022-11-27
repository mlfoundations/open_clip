
import os
import random
import pytest
import numpy
import torch
from PIL import Image
import open_clip
import util_test

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# test all model with some exceptions
models_to_test = set(open_clip.list_models()).difference({
        # not available with timm yet
        # see https://github.com/mlfoundations/open_clip/issues/219
        'timm-convnext_xlarge',
        'timm-vit_medium_patch16_gap_256',
        # exceeds GH runner memory limit
        'ViT-bigG-14',
        'ViT-e-14',
        'mt5-xl-ViT-H-14',
})

@pytest.mark.parametrize('model_name', models_to_test)
def test_inference_with_data(
        model_name,
        pretrained = None,
        precision = 'fp32',
        jit = False,
        force_quick_gelu = False,
        # experimentally determined between author machine and GH runner
        tolerance = torch.finfo(torch.float32).resolution * 4
):
    util_test.seed_all()
    model, _, preprocess_val = open_clip.create_model_and_transforms(
            model_name,
            pretrained = pretrained,
            precision = precision,
            jit = jit,
            force_quick_gelu = force_quick_gelu,
            pretrained_hf = False
    )
    model_id = f'{model_name}_{pretrained}_{precision}'
    input_dir, output_dir = util_test.get_data_dirs()
    # text
    input_text_path = os.path.join(input_dir, 'random_text.pt')
    gt_text_path = os.path.join(output_dir, f'{model_id}_random_text.pt')
    assert os.path.isfile(input_text_path), f"missing test data, expected at {input_text_path}"
    assert os.path.isfile(gt_text_path), f"missing test data, expected at {gt_text_path}"
    input_text = torch.load(input_text_path)
    gt_text = torch.load(gt_text_path)
    y_text = util_test.inference_text(model, model_name, input_text)
    assert torch.allclose(y_text, gt_text, atol=tolerance), f"text output differs @ {input_text_path}"
    # image
    image_size = model.visual.image_size
    if not isinstance(image_size, tuple):
        image_size = (image_size, image_size)
    input_image_path = os.path.join(input_dir, f'random_image_{image_size[0]}_{image_size[1]}.pt')
    gt_image_path = os.path.join(output_dir, f'{model_id}_random_image.pt')
    assert os.path.isfile(input_image_path), f"missing test data, expected at {input_image_path}"
    assert os.path.isfile(gt_image_path), f"missing test data, expected at {gt_image_path}"
    input_image = torch.load(input_image_path)
    gt_image = torch.load(gt_image_path)
    y_image = util_test.inference_image(model, preprocess_val, input_image)
    assert torch.allclose(y_image, gt_image, atol=tolerance), f"image output differs @ {input_image_path}"


