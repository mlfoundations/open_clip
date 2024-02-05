from contextlib import suppress
from typing import Any, Union

import torch
import torch.nn.functional as f
import torch.utils.data
from tqdm import tqdm


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    tokenizer: Any,
    device: Union[str, torch.device],
    amp: bool = True,
):
    """
    Evaluate the model on the given dataset.
    The task has N instances, each instance has I images and C captions.
    For each instance, the goal is to find the correct image for each caption and the
    correct caption for each image.
    This is done by computing the similarities between each image and each caption.
    This procedure is used to evaluate the models on Winoground and SugarCrepe.

    Parameters
    ----------
    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    dataloader: torch.utils.data.DataLoader
        Dataloader to use for evaluation
    tokenizer:
        Text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    device: cpu/cuda
    amp: Whether to use automatic mixed precision

    Returns
    -------
        Dict of accuracy metrics
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    image_score = []
    text_score = []
    score = []

    for batch_images, batch_texts in tqdm(dataloader):
        if len(batch_images.shape) == 4:
            b, c, h, w = batch_images.shape
            batch_images = batch_images.view(b, 1, c, h, w)

        # batch_images: B, nb_images_per_instance, C, H, W
        # batch_texts: B, nb_captions_per_instance

        b, nim, c, h, w = batch_images.shape

        nt = len(batch_texts[0])
        batch_images = batch_images.to(device)
        batch_images_ = batch_images.view(b * nim, c, h, w)  # B*nim, C, H, W

        # tokenize all texts in the batch
        batch_texts_tok_ = tokenizer(
            [text for i, texts in enumerate(batch_texts) for text in texts]
        ).to(device)

        # compute the embedding of images and texts
        with torch.no_grad(), autocast():
            batch_images_emb = f.normalize(
                model.encode_image(batch_images_), dim=-1
            ).view(b, nim, -1)
            batch_texts_emb = f.normalize(
                model.encode_text(batch_texts_tok_), dim=-1
            ).view(b, nt, -1)

        gt = torch.arange(min(nim, nt)).to(device)

        for i in range(b):
            images_emb = batch_images_emb[i]
            texts_emb = batch_texts_emb[i]
            scores = images_emb @ texts_emb.t()
            image_closest_text = scores.argmax(dim=1)[: len(gt)]
            text_closest_image = scores.argmax(dim=0)[: len(gt)]

            pred_text_is_correct = (image_closest_text == gt).all().item()
            pred_image_is_correct = (text_closest_image == gt).all().item()
            all_correct = pred_text_is_correct and pred_image_is_correct

            image_score.append(pred_image_is_correct)
            text_score.append(pred_text_is_correct)
            score.append(all_correct)

    return {
        'image_acc': torch.Tensor(image_score).float().mean().item(),
        'text_acc': torch.Tensor(text_score).float().mean().item(),
        'acc': torch.Tensor(score).float().mean().item(),
    }
