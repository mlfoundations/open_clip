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


def _multi_caption_targets(image_features, text_features, image_ids, chunk_size, device, dtype, scale):
    # Keep the first embedding of each image in dataset order, including when
    # captions for the same image fall in different loader batches.
    _, first, inverse = np.unique(image_ids, return_index=True, return_inverse=True)
    order = np.argsort(first)
    owners = np.argsort(order)[inverse]
    keep = np.zeros(len(image_ids), dtype=bool)
    keep[first] = True
    images = [
        chunk[torch.as_tensor(keep[start:end], device=chunk.device)]
        for start, end, chunk in _iter_feature_chunks(image_features, chunk_size)
        if keep[start:end].any()
    ]

    # Score positives with the same tiles as the ranking pass. Computing them
    # as individual dot products can break ties under reduced precision.
    targets = torch.empty(len(owners), dtype=dtype, device=device)
    for start, end, chunk in _iter_feature_chunks(images, chunk_size):
        image = _to_device(chunk, device, dtype)
        for t0, t1, text in _iter_feature_chunks(text_features, chunk_size):
            local = owners[t0:t1]
            cols = np.flatnonzero((local >= start) & (local < end))
            if cols.size:
                scores = scale * (image @ _to_device(text, device, dtype).T)
                col_idx = torch.as_tensor(cols, device=device)
                row_idx = torch.as_tensor(local[cols] - start, device=device)
                targets[t0 + col_idx] = scores[row_idx, col_idx]

    owner = torch.as_tensor(owners, device=device)
    best = torch.full((len(first),), -torch.inf, dtype=dtype, device=device)
    best.scatter_reduce_(0, owner, targets, reduce="amax")
    # Ties use the earliest relevant caption, just like stable descending sort.
    positions = torch.arange(len(owners), device=device)
    first_best = torch.full_like(best, len(owners), dtype=torch.long)
    first_best.scatter_reduce_(0, owner, torch.where(targets == best[owner], positions, len(owners)), reduce="amin")
    return images, best, first_best, targets, owner


def _paired_retrieval_ranks(
        image_features,
        text_features,
        logit_scale,
        chunk_size,
        device=None,
        retrieval_dtype=torch.float32,
        image_ids=None,
):
    image_shape = _feature_shape(image_features)
    text_shape = _feature_shape(text_features)
    if image_shape != text_shape:
        raise ValueError(
            "Paired retrieval metrics require image and text features with matching shape."
        )

    num_items = image_shape[0]
    if image_ids is not None:
        image_ids = np.asarray(image_ids)
        if image_ids.shape != (num_items,):
            raise ValueError("Retrieval needs one image ID per image-caption pair.")
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
    text_to_image_ranks = torch.zeros(num_items, device=device, dtype=torch.long)

    scale = torch.as_tensor(
        logit_scale,
        device=device,
        dtype=retrieval_dtype,
    )
    if image_ids is None:
        targets = torch.empty(num_items, device=device, dtype=retrieval_dtype)
        paired_chunks = zip(
            _iter_feature_chunks(image_features, chunk_size),
            _iter_feature_chunks(text_features, chunk_size),
        )
        for (start, end, image_chunk), (_, _, text_chunk) in paired_chunks:
            image = _to_device(image_chunk, device, retrieval_dtype)
            text = _to_device(text_chunk, device, retrieval_dtype)
            targets[start:end] = (scale * (image @ text.T)).diagonal()
        best = targets
        first_best = owners = torch.arange(num_items, device=device)
    else:
        image_features, best, first_best, targets, owners = _multi_caption_targets(
            image_features, text_features, image_ids, chunk_size, device, retrieval_dtype, scale,
        )
    image_to_text_ranks = torch.empty(len(best), dtype=torch.long)

    for image_start, image_end, image_chunk in _iter_feature_chunks(image_features, chunk_size):
        image = _to_device(image_chunk, device, retrieval_dtype)
        image_target = best[image_start:image_end]
        image_rank = torch.zeros(image.shape[0], device=device, dtype=torch.long)
        image_query_idx = first_best[image_start:image_end, None]

        for text_start, text_end, text_chunk in _iter_feature_chunks(text_features, chunk_size):
            text = _to_device(text_chunk, device, retrieval_dtype)
            text_query_idx = owners[None, text_start:text_end]
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
        image_ids=None,
):
    metrics = {}

    image_to_text_ranks, text_to_image_ranks = _paired_retrieval_ranks(
        image_features,
        text_features,
        logit_scale,
        retrieval_chunk_size,
        device=retrieval_device,
        retrieval_dtype=retrieval_dtype,
        image_ids=image_ids,
    )

    _add_rank_metrics(metrics, f"{image_key}_to_{text_key}", image_to_text_ranks)
    _add_rank_metrics(metrics, f"{text_key}_to_{image_key}", text_to_image_ranks)
    return metrics
