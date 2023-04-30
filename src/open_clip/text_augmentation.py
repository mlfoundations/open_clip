import torch
import random
import numpy as np


'''
Data augmentation scripts from the original contriever implementation: https://github.com/facebookresearch/contriever/blob/main/src/data.py
'''

def randomcrop(x, ratio_min, ratio_max):

    ratio = random.uniform(ratio_min, ratio_max)
    length = int(len(x) * ratio)
    start = random.randint(0, len(x) - length)
    end = start + length
    crop = x[start:end].clone()
    return crop


def add_token(x, token):
    x = torch.cat((torch.tensor([token]), x))
    return x


def deleteword(x, p=0.1):
    mask = np.random.rand(len(x))
    x = [e for e, m in zip(x, mask) if m > p]
    return x


def replaceword(x, min_random, max_random, p=0.1):
    mask = np.random.rand(len(x))
    x = [e if m > p else random.randint(min_random, max_random) for e, m in zip(x, mask)]
    return x


def maskword(x, mask_id, p=0.1):
    mask = np.random.rand(len(x))
    x = [e if m > p else mask_id for e, m in zip(x, mask)]
    return x


def shuffleword(x, p=0.1):
    count = (np.random.rand(len(x)) < p).sum()
    """Shuffles any n number of values in a list"""
    indices_to_shuffle = random.sample(range(len(x)), k=count)
    to_shuffle = [x[i] for i in indices_to_shuffle]
    random.shuffle(to_shuffle)
    for index, value in enumerate(to_shuffle):
        old_index = indices_to_shuffle[index]
        x[old_index] = value
    return x


def apply_augmentation(x, tokenizer,args):
    if args.text_augmentation == "mask":
        return torch.tensor(maskword(x, mask_id=tokenizer.mask_token_id, p=args.text_prob_augmentation))
    elif args.text_augmentation == "replace":
        return torch.tensor(
            replaceword(x, min_random=args.start_id, max_random=len(tokenizer) - 1, p=args.text_prob_augmentation)
        )
    elif args.text_augmentation == "delete":
        return torch.tensor(deleteword(x, p=args.text_prob_augmentation))
    elif args.text_augmentation == "shuffle":
        return torch.tensor(shuffleword(x, p=args.text_prob_augmentation))
    else:
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        return x
    

def add_bos_eos(x, bos_token_id, eos_token_id):
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)
    if bos_token_id is None and eos_token_id is not None:
        x = torch.cat([x.clone().detach(), torch.tensor([eos_token_id])])
    elif bos_token_id is not None and eos_token_id is None:
        x = torch.cat([torch.tensor([bos_token_id]), x.clone().detach()])
    elif bos_token_id is None and eos_token_id is None:
        pass
    else:
        x = torch.cat([torch.tensor([bos_token_id]), x.clone().detach(), torch.tensor([eos_token_id])])
    return x


def create_unsupervised_sample(text,tokenizer,args):
    # remove padding
    text = text[text!=tokenizer.pad_token_id]
    if text[0] == tokenizer.bos_token_id:
        text == text[1:]
    if text[-1] == tokenizer.eos_token_id:
        text = text[:-1]
    # perform data augmentation
    text = randomcrop(text, args.text_aug_ratio_min, args.text_aug_ratio_max)
    text = apply_augmentation(text, tokenizer, args)
    # add bos/eos tokens back
    text = add_bos_eos(text, tokenizer.bos_token_id, tokenizer.eos_token_id)
    # padding back to max length
    text = tokenizer(
            text,
            return_tensors='pt',
            max_length=args.context_length,
            padding='max_length',
            truncation=True,
        ).input_ids
    
    return text