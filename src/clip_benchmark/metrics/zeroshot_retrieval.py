from contextlib import suppress
from typing import Any, Optional, Union

import torch
import torch.nn.functional as f
import torch.utils.data
from tqdm import tqdm


def _dataloader_with_indices(dataloader):
    start = 0
    for x, y in dataloader:
        end = start + len(x)
        inds = torch.arange(start, end)
        yield x, y, inds
        start = end


def _recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings
        (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(
        topk_indices, num_classes=nb_images
    )
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1, 2))
    # compute recall at k
    return nb_true_positive / nb_positive


def _batchify(func, x, y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(x), batch_size):
        end = start + batch_size
        _x = x[start:end].to(device)
        _y = y[start:end].to(device)
        result = func(_x, _y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    tokenizer: Any,
    device: Union[str, torch.device],
    amp: bool = True,
    recall_k_list: Optional[list[int]] = None,
):
    """
    Evaluate the model on the given dataset

    Parameters
    ----------
    model: torch.nn,Module
        CLIP-like model with `encode_image` and `encode_text`
    dataloader: torch.utils.data.Dataloader
        dataloader to use for evaluation
    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    device: cpu/cuda
    amp: whether to use automatic mixed precision
    recall_k_list: list of int
        recall@k k's to use
    Returns
    -------
        Dict of retrieval metrics
    """
    # list of batch of images embedding
    batch_images_emb_list = []
    # list of batch of text embedding
    batch_texts_emb_list = []
    # for each text, we collect the corresponding image index, as each image can
    # have multiple corresponding texts
    texts_image_index = []
    dataloader = _dataloader_with_indices(dataloader)
    autocast = torch.cuda.amp.autocast if amp else suppress

    for batch_images, batch_texts, inds in tqdm(dataloader):
        batch_images = batch_images.to(device)
        # tokenize all texts in the batch
        batch_texts_tok = tokenizer(
            [text for i, texts in enumerate(batch_texts) for text in texts]
        ).to(device)

        # store the index of image for each text
        batch_texts_image_index = [
            ind for ind, texts in zip(inds, batch_texts) for _ in texts
        ]

        # compute the embedding of images and texts
        with torch.no_grad(), autocast():
            batch_images_emb = f.normalize(model.encode_image(batch_images), dim=-1)
            batch_texts_emb = f.normalize(model.encode_text(batch_texts_tok), dim=-1)

        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        texts_image_index.extend(batch_texts_image_index)

    batch_size = len(batch_images_emb_list[0])

    # concatenate all embeddings
    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)

    # get the score for each text and image pair
    scores = texts_emb @ images_emb.t()

    # construct a the positive pair matrix, which tells whether each text-image pair
    # is a positive or not
    positive_pairs = torch.zeros_like(scores, dtype=torch.bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}

    recall_k_list = recall_k_list or [5]

    for recall_k in recall_k_list:
        # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_
        # positives, where the number
        # of true positives, e.g. for text retrieval, is, for each image,  the number
        # of retrieved texts matching that image among the top-k.
        # Also, the number of positives are the total number of texts matching the
        # image in the dataset, as we have a set of captions for each image, that
        # number will be greater than 1 for text retrieval.
        # However, image/text retrieval recall@k, the way it is done in CLIP-like
        # papers, is a bit different. recall@k, in CLIP-like papers, is, for each image,
        # either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        # So we can easily compute that using the actual recall, by checking whether
        # there is at least one true positive, which would be the case if the recall
        # is greater than 0. One we compute the recal for each image (or text), we
        # average it over the dataset.

        metrics[f'image_retrieval_recall@{recall_k}'] = (
            (
                _batchify(
                    _recall_at_k, scores, positive_pairs, batch_size, device, k=recall_k
                )
                > 0
            )
            .float()
            .mean()
            .item()
        )
        metrics[f'text_retrieval_recall@{recall_k}'] = (
            (
                _batchify(
                    _recall_at_k,
                    scores.T,
                    positive_pairs.T,
                    batch_size,
                    device,
                    k=recall_k,
                )
                > 0
            )
            .float()
            .mean()
            .item()
        )

    return metrics
