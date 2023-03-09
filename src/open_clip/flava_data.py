import torch
from transformers import DataCollatorForLanguageModeling

from .tokenizer import HFTokenizer


def get_flava_collate(hf_tokenizer, mlm_prob=0.15, itm_prob=0.1):
    assert isinstance(hf_tokenizer, HFTokenizer), 'tokenizer must be HFTokenizer'
    tokenizer = hf_tokenizer.tokenizer
    mlm_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_prob,
    )

    def collate(example_list):
        image_list, text_list = [], []
        for example in example_list:
            image_list.append(example["image"])
            text_list.append(example["text"])
        image = torch.stack(image_list)
        text = torch.stack(text_list)
        batch = {
            "image": image,
            "text": text,
        }

        # ITM
        bs = image.shape[0]
        itm_labels = torch.bernoulli(torch.ones(bs, dtype=image.dtype) * (1 - itm_prob))
        itm_idx_mask = itm_labels.byte()
        batch_idx = torch.arange(bs)
        neg_batch_idx = (batch_idx + 1) % bs
        negative_text_idx = itm_idx_mask * batch_idx + (1 - itm_idx_mask) * neg_batch_idx
        batch.update({
            "itm_neg_text_idx": negative_text_idx,
            "itm_labels": itm_labels,
        })

        # MLM
        mlm_input = mlm_collator(text_list)  # TODO: add special_tokens_mask (improves efficiency)
        batch.update({
            "text_masked": mlm_input["input_ids"],
            "mlm_labels": mlm_input["labels"],
        })

        return batch

    return collate


def get_mlm_collate(hf_tokenizer, mlm_prob=0.15):
    assert isinstance(hf_tokenizer, HFTokenizer), 'tokenizer must be HFTokenizer'
    tokenizer = hf_tokenizer.tokenizer
    mlm_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_prob,
    )

    def collate(example_list):
        mlm_input = mlm_collator([example["text"] for example in example_list])
        return {
            "text_masked": mlm_input["input_ids"],
            "mlm_labels": mlm_input["labels"],
        }

    return collate
