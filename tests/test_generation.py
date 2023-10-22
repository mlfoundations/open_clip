import os
import pytest
import torch
import open_clip
from PIL import Image
import util_test

os.environ["CUDA_VISIBLE_DEVICES"] = ""

if hasattr(torch._C, "_jit_set_profiling_executor"):
    # legacy executor is too slow to compile large models for unit tests
    # no need for the fusion performance here
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(False)

models_to_test = open_clip.list_generative_models().difference(
    {"coca_roberta-ViT-B-32", "coca_base"}
)


@pytest.mark.generative_regression_test
@pytest.mark.parametrize("model_name", models_to_test)
def test_generate_with_data(
    model_name,
    # jit, currently omitted because no generative model supports jit
    pretrained=None,
    pretrained_hf=False,
    precision="fp32",
    force_quick_gelu=False,
):
    util_test.seed_all()
    model_id = f"{model_name}_{pretrained or pretrained_hf}_{precision}"
    input_dir, output_dir = util_test.get_data_dirs(generative=True)
    # text
    gt_image_path = os.path.join(output_dir, f"generative_{model_id}_random_image.pt")
    gt_text_path = os.path.join(output_dir, f"generative_{model_id}_random_text.pt")
    gt_logits_path = os.path.join(output_dir, f"generative_{model_id}_logits.pt")
    if not os.path.isfile(gt_image_path):
        pytest.skip(reason=f"missing test data, expected at {gt_image_path}")
    if not os.path.isfile(gt_text_path):
        pytest.skip(reason=f"missing test data, expected at {gt_text_path}")
    if not os.path.isfile(gt_logits_path):
        pytest.skip(reason=f"missing test data, expected at {gt_logits_path}")
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        precision=precision,
        # jit = jit, currently omitted because no generative model supports jit
        force_quick_gelu=force_quick_gelu,
        pretrained_hf=pretrained_hf,
    )
    input_image = torch.load(gt_image_path)
    gt_text = torch.load(gt_text_path)
    with torch.no_grad(), torch.cuda.amp.autocast():
        y_text = util_test.model_generate(model, preprocess_val, input_image)
    assert y_text == gt_text, f"text output differs @ {gt_text_path}"
    # logits
    y_logits = util_test.forward_model(model, model_name, preprocess_val, input_image, gt_text)[
        "logits"
    ]
    gt_logits = torch.load(gt_logits_path)
    assert (y_logits == gt_logits).all(), f"logits output differs @ {gt_image_path}"
