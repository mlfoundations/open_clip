
import os
import pytest
import torch
import open_clip
import util_test
import PIL

os.environ['CUDA_VISIBLE_DEVICES'] = ''

torch.serialization.add_safe_globals([PIL.Image.Image])

if hasattr(torch._C, '_jit_set_profiling_executor'):
    # legacy executor is too slow to compile large models for unit tests
    # no need for the fusion performance here
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(False)

models_to_test = set(open_clip.list_models())

# testing excemptions
models_to_test = models_to_test.difference({
        # not available with timm yet
        # see https://github.com/mlfoundations/open_clip/issues/219
        'convnext_xlarge',
        'convnext_xxlarge',
        'convnext_xxlarge_320',
        'vit_medium_patch16_gap_256',
        # exceeds GH runner memory limit
        'ViT-bigG-14',
        'ViT-e-14',
        'mt5-xl-ViT-H-14',
        'coca_base',
        'coca_ViT-B-32',
        'coca_roberta-ViT-B-32'
})

if 'OPEN_CLIP_TEST_REG_MODELS' in os.environ:
    external_model_list = os.environ['OPEN_CLIP_TEST_REG_MODELS']
    with open(external_model_list, 'r') as f:
        models_to_test = set(f.read().splitlines()).intersection(models_to_test)
    print(f"Selected models from {external_model_list}: {models_to_test}")

# TODO: add "coca_ViT-B-32" onece https://github.com/pytorch/pytorch/issues/92073 gets fixed
models_to_test = list(models_to_test)
models_to_test.sort()
models_to_test = [(model_name, False) for model_name in models_to_test]

models_to_jit_test = {"ViT-B-32"}
models_to_jit_test = list(models_to_jit_test)
models_to_jit_test = [(model_name, True) for model_name in models_to_jit_test]
models_to_test_fully = models_to_test + models_to_jit_test


@pytest.mark.regression_test
@pytest.mark.parametrize("model_name,jit", models_to_test_fully)
def test_inference_with_data(
        model_name,
        jit,
        pretrained = None,
        pretrained_hf = False,
        precision = 'fp32',
        force_quick_gelu = False,
):
    util_test.seed_all()
    model, _, preprocess_val = open_clip.create_model_and_transforms(
            model_name,
            pretrained = pretrained,
            precision = precision,
            jit = jit,
            force_quick_gelu = force_quick_gelu,
            pretrained_hf = pretrained_hf
    )
    model_id = f'{model_name}_{pretrained or pretrained_hf}_{precision}'
    input_dir, output_dir = util_test.get_data_dirs()
    # text
    input_text_path = os.path.join(input_dir, 'random_text.pt')
    gt_text_path = os.path.join(output_dir, f'{model_id}_random_text.pt')
    if not os.path.isfile(input_text_path):
        pytest.skip(reason = f"missing test data, expected at {input_text_path}")
    if not os.path.isfile(gt_text_path):
        pytest.skip(reason = f"missing test data, expected at {gt_text_path}")
    input_text = torch.load(input_text_path)
    gt_text = torch.load(gt_text_path)
    y_text = util_test.inference_text(model, model_name, input_text)
    assert (y_text == gt_text).all(), f"text output differs @ {input_text_path}"
    # image
    image_size = model.visual.image_size
    if not isinstance(image_size, tuple):
        image_size = (image_size, image_size)
    input_image_path = os.path.join(input_dir, f'random_image_{image_size[0]}_{image_size[1]}.pt')
    gt_image_path = os.path.join(output_dir, f'{model_id}_random_image.pt')
    if not os.path.isfile(input_image_path):
        pytest.skip(reason = f"missing test data, expected at {input_image_path}")
    if not os.path.isfile(gt_image_path):
        pytest.skip(reason = f"missing test data, expected at {gt_image_path}")
    input_image = torch.load(input_image_path)
    gt_image = torch.load(gt_image_path)
    y_image = util_test.inference_image(model, preprocess_val, input_image)
    assert (y_image == gt_image).all(), f"image output differs @ {input_image_path}"
    
    if not jit:
        model.eval()
        model_out = util_test.forward_model(model, model_name, preprocess_val, input_image, input_text)
        if type(model) not in [open_clip.CLIP, open_clip.CustomTextCLIP]:
            assert type(model_out) == dict
        else:
            model.output_dict = True
            model_out_dict = util_test.forward_model(model, model_name, preprocess_val, input_image, input_text)
            assert (model_out_dict["image_features"] == model_out[0]).all()
            assert (model_out_dict["text_features"] == model_out[1]).all()
            assert (model_out_dict["logit_scale"] == model_out[2]).all()
            model.output_dict = None
    else:
        model, _, preprocess_val = open_clip.create_model_and_transforms(
            model_name,
            pretrained = pretrained,
            precision = precision,
            jit = False,
            force_quick_gelu = force_quick_gelu,
            pretrained_hf = pretrained_hf
        )
        
        test_model = util_test.TestWrapper(model, model_name, output_dict=False)
        test_model = torch.jit.script(test_model)
        model_out = util_test.forward_model(test_model, model_name, preprocess_val, input_image, input_text)
        assert model_out["test_output"].shape[-1] == 2

        test_model = util_test.TestWrapper(model, model_name, output_dict=True)
        test_model = torch.jit.script(test_model)
        model_out = util_test.forward_model(test_model, model_name, preprocess_val, input_image, input_text)
        assert model_out["test_output"].shape[-1] == 2
        
    


