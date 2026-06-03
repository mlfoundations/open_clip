import numpy as np
import torch


DEFAULT_RETRIEVAL_CHUNK_SIZE = 4096


def _resolve_chunk_size(chunk_size, num_items):
    if chunk_size is None or chunk_size <= 0:
        return num_items
    return min(int(chunk_size), num_items)


def _resolve_retrieval_dtype(retrieval_dtype, feature_dtype):
    if retrieval_dtype is None or retrieval_dtype == "model":
        return feature_dtype
    if retrieval_dtype == "fp32":
        return torch.float32
    if isinstance(retrieval_dtype, torch.dtype):
        return retrieval_dtype
    raise ValueError(f"Unsupported retrieval dtype: {retrieval_dtype!r}")


def _to_device(tensor, device, dtype):
    if tensor.device == device and tensor.dtype == dtype:
        return tensor
    return tensor.to(device=device, dtype=dtype, non_blocking=True)


def _feature_shape(features):
    if isinstance(features, torch.Tensor):
        if features.ndim != 2:
            raise ValueError("Retrieval metrics expect 2D feature tensors.")
        return features.shape

    num_items = 0
    feature_dim = None
    for feature in features:
        if feature.ndim != 2:
            raise ValueError("Retrieval metrics expect 2D feature tensors.")
        if feature_dim is None:
            feature_dim = feature.shape[1]
        elif feature.shape[1] != feature_dim:
            raise ValueError("Retrieval feature tensors must have a consistent feature dimension.")
        num_items += feature.shape[0]

    return (num_items, 0 if feature_dim is None else feature_dim)


def _cat_chunk(parts):
    if len(parts) == 1:
        return parts[0]
    return torch.cat(parts, dim=0)


def _first_feature_tensor(features):
    if isinstance(features, torch.Tensor):
        return features
    for feature in features:
        if feature.shape[0] > 0:
            return feature
    return None


def _iter_feature_chunks(features, chunk_size):
    if isinstance(features, torch.Tensor):
        for start in range(0, features.shape[0], chunk_size):
            end = min(start + chunk_size, features.shape[0])
            yield start, end, features[start:end]
        return

    parts = []
    part_size = 0
    chunk_start = 0
    global_offset = 0
    for feature in features:
        feature_offset = 0
        while feature_offset < feature.shape[0]:
            if part_size == 0:
                chunk_start = global_offset + feature_offset
            take = min(feature.shape[0] - feature_offset, chunk_size - part_size)
            parts.append(feature[feature_offset:feature_offset + take])
            part_size += take
            feature_offset += take
            if part_size == chunk_size:
                yield chunk_start, chunk_start + part_size, _cat_chunk(parts)
                parts = []
                part_size = 0
        global_offset += feature.shape[0]

    if part_size:
        yield chunk_start, chunk_start + part_size, _cat_chunk(parts)


def _paired_retrieval_ranks(
        image_features,
        text_features,
        logit_scale,
        chunk_size,
        device=None,
        retrieval_dtype=torch.float32,
):
    image_shape = _feature_shape(image_features)
    text_shape = _feature_shape(text_features)
    if image_shape != text_shape:
        raise ValueError(
            "Paired retrieval metrics require image and text features with matching shape."
        )

    num_items = image_shape[0]
    if num_items == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty

    first_feature = _first_feature_tensor(image_features)
    assert first_feature is not None
    if device is None:
        device = first_feature.device
    else:
        device = torch.device(device)
    retrieval_dtype = _resolve_retrieval_dtype(retrieval_dtype, first_feature.dtype)
    chunk_size = _resolve_chunk_size(chunk_size, num_items)
    image_to_text_ranks = torch.empty(num_items, dtype=torch.long)
    text_to_image_ranks = torch.zeros(num_items, device=device, dtype=torch.long)

    scale = torch.as_tensor(
        logit_scale,
        device=device,
        dtype=retrieval_dtype,
    )
    targets = torch.empty(num_items, device=device, dtype=retrieval_dtype)

    paired_chunks = zip(
        _iter_feature_chunks(image_features, chunk_size),
        _iter_feature_chunks(text_features, chunk_size),
    )
    for (image_start, image_end, image_chunk), (text_start, text_end, text_chunk) in paired_chunks:
        assert image_start == text_start and image_end == text_end
        image = _to_device(image_chunk, device, retrieval_dtype)
        text = _to_device(text_chunk, device, retrieval_dtype)
        paired_scores = scale * (image @ text.T)
        targets[image_start:image_end] = paired_scores.diagonal()

    for image_start, image_end, image_chunk in _iter_feature_chunks(image_features, chunk_size):
        image = _to_device(image_chunk, device, retrieval_dtype)
        image_target = targets[image_start:image_end]
        image_rank = torch.zeros(image.shape[0], device=device, dtype=torch.long)
        image_query_idx = torch.arange(image_start, image_end, device=device)[:, None]

        for text_start, text_end, text_chunk in _iter_feature_chunks(text_features, chunk_size):
            text = _to_device(text_chunk, device, retrieval_dtype)
            text_query_idx = torch.arange(text_start, text_end, device=device)[None, :]
            scores = scale * (image @ text.T)
            text_candidate_idx = torch.arange(text_start, text_end, device=device)[None, :]
            image_candidate_idx = torch.arange(image_start, image_end, device=device)[:, None]
            greater_image = (
                (scores > image_target[:, None]) |
                ((scores == image_target[:, None]) & (text_candidate_idx < image_query_idx))
            )
            greater_text = (
                (scores > targets[text_start:text_end][None, :]) |
                ((scores == targets[text_start:text_end][None, :]) & (image_candidate_idx < text_query_idx))
            )
            image_rank += greater_image.sum(dim=1)
            text_to_image_ranks[text_start:text_end] += greater_text.sum(dim=0)

        image_to_text_ranks[image_start:image_end] = image_rank.cpu()

    return image_to_text_ranks.numpy(), text_to_image_ranks.cpu().numpy()


def _add_rank_metrics(metrics, name, ranks):
    metrics[f"{name}_mean_rank"] = ranks.mean() + 1
    metrics[f"{name}_median_rank"] = np.floor(np.median(ranks)) + 1
    for k in [1, 5, 10]:
        metrics[f"{name}_R@{k}"] = np.mean(ranks < k)


def get_clip_metrics(
        image_features,
        text_features,
        logit_scale,
        image_key="image",
        text_key="text",
        retrieval_chunk_size=DEFAULT_RETRIEVAL_CHUNK_SIZE,
        retrieval_device=None,
        retrieval_dtype=torch.float32,
):
    metrics = {}

    image_to_text_ranks, text_to_image_ranks = _paired_retrieval_ranks(
        image_features,
        text_features,
        logit_scale,
        retrieval_chunk_size,
        device=retrieval_device,
        retrieval_dtype=retrieval_dtype,
    )

    _add_rank_metrics(metrics, f"{image_key}_to_{text_key}", image_to_text_ranks)
    _add_rank_metrics(metrics, f"{text_key}_to_{image_key}", text_to_image_ranks)
    return metrics
