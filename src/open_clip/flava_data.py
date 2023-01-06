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
    pad_token_id = tokenizer.pad_token_id

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
        itm_labels = torch.bernoulli(torch.ones(len(image)) * (1 - itm_prob))
        negative_text_idx = (torch.where(itm_labels == 0)[0] + 1) % len(text)

        itm_text = text.clone()
        itm_text[torch.where(itm_labels == 0)] = text[negative_text_idx]
        batch.update({
            "itm_text": itm_text,
            "itm_labels": itm_labels,
        })

        # MLM
        mlm_input = mlm_collator(text_list)
        batch.update({
            "text_masked": mlm_input["input_ids"],
            "text_masked_labels": mlm_input["labels"],
        })

        return batch

    return collate
