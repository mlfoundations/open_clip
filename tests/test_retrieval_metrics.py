import numpy as np
import pytest
import torch
import torch.nn.functional as F


def _full_matrix_metrics(image_features, text_features, logit_scale, image_key="image", text_key="text"):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {f"{image_key}_to_{text_key}": logits_per_image, f"{text_key}_to_{image_key}": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def _features(num_items=11, dim=7):
    generator = torch.Generator().manual_seed(0)
    image_features = F.normalize(torch.randn(num_items, dim, generator=generator), dim=-1)
    text_features = F.normalize(torch.randn(num_items, dim, generator=generator), dim=-1)
    return image_features, text_features


def _assert_metrics_close(actual, expected):
    assert set(actual) == set(expected)
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value)


def test_chunked_clip_metrics_match_full_matrix_metrics():
    from open_clip_train.metrics import get_clip_metrics

    image_features, text_features = _features()
    logit_scale = torch.tensor(3.5)

    actual = get_clip_metrics(
        image_features=image_features,
        text_features=text_features,
        logit_scale=logit_scale,
        image_key="scan",
        text_key="report",
        retrieval_chunk_size=3,
    )
    expected = _full_matrix_metrics(
        image_features,
        text_features,
        logit_scale,
        image_key="scan",
        text_key="report",
    )

    _assert_metrics_close(actual, expected)


def test_default_chunked_clip_metrics_match_full_matrix_metrics():
    from open_clip_train.metrics import get_clip_metrics

    image_features, text_features = _features(num_items=13, dim=5)
    logit_scale = torch.tensor(8.0)

    actual = get_clip_metrics(
        image_features=image_features,
        text_features=text_features,
        logit_scale=logit_scale,
        retrieval_chunk_size=4,
    )
    expected = _full_matrix_metrics(image_features, text_features, logit_scale)

    _assert_metrics_close(actual, expected)


def test_feature_list_chunked_metrics_match_full_matrix_metrics():
    from open_clip_train.metrics import get_clip_metrics

    image_features, text_features = _features(num_items=19, dim=8)
    logit_scale = torch.tensor(4.0)
    image_chunks = [image_features[:3], image_features[3:10], image_features[10:]]
    text_chunks = [text_features[:5], text_features[5:11], text_features[11:]]

    actual = get_clip_metrics(
        image_features=image_chunks,
        text_features=text_chunks,
        logit_scale=logit_scale,
        retrieval_chunk_size=4,
    )
    expected = _full_matrix_metrics(image_features, text_features, logit_scale)

    _assert_metrics_close(actual, expected)


def test_zero_chunk_size_matches_chunked_metrics():
    from open_clip_train.metrics import get_clip_metrics

    image_features, text_features = _features(num_items=17, dim=9)
    logit_scale = torch.tensor(5.0)

    chunked = get_clip_metrics(
        image_features=image_features,
        text_features=text_features,
        logit_scale=logit_scale,
        retrieval_chunk_size=4,
    )
    one_block = get_clip_metrics(
        image_features=image_features,
        text_features=text_features,
        logit_scale=logit_scale,
        retrieval_chunk_size=0,
    )

    _assert_metrics_close(one_block, chunked)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_retrieval_device_cuda_matches_cpu_metrics():
    from open_clip_train.metrics import get_clip_metrics

    image_features, text_features = _features(num_items=10, dim=6)
    logit_scale = torch.tensor(2.0)

    cpu_metrics = get_clip_metrics(
        image_features=image_features,
        text_features=text_features,
        logit_scale=logit_scale,
        retrieval_chunk_size=3,
    )
    cuda_metrics = get_clip_metrics(
        image_features=image_features,
        text_features=text_features,
        logit_scale=logit_scale,
        retrieval_chunk_size=3,
        retrieval_device=torch.device("cuda"),
    )

    _assert_metrics_close(cuda_metrics, cpu_metrics)
