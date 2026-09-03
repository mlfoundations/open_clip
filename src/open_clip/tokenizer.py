""" CLIP tokenizer

Copied from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import gzip
import html
import base64
import json
import os
import random
import string
from functools import lru_cache, partial
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, Union
import warnings

import ftfy
import numpy as np
import regex as re
import torch

# https://stackoverflow.com/q/62691279
os.environ["TOKENIZERS_PARALLELISM"] = "false"
_nltk_init = False

DEFAULT_CONTEXT_LENGTH = 77  # default context length for OpenAI CLIP

TokenizerInput = Union[str, Sequence[str]]
TokenIds = Union[Sequence[int], np.ndarray, torch.Tensor]
BatchTokenIds = Union[Iterable[TokenIds], np.ndarray, torch.Tensor]
TokenizerOutput = Union[
    torch.Tensor,
    List[torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]


class Tokenizer(Protocol):
    """Structural interface shared by OpenCLIP tokenizer implementations."""

    context_length: Optional[int]
    vocab_size: int
    bos_token_id: Optional[int]
    eos_token_id: Optional[int]
    pad_token_id: Optional[int]
    sot_token_id: Optional[int]
    eot_token_id: Optional[int]
    all_special_ids: List[int]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]: ...

    def decode(
            self,
            tokens: TokenIds,
            skip_special_tokens: bool = False,
            stop_at_eos: bool = True,
    ) -> str: ...

    def batch_decode(
            self,
            batch_tokens: BatchTokenIds,
            skip_special_tokens: bool = False,
            stop_at_eos: bool = True,
    ) -> List[str]: ...

    def __call__(
            self,
            texts: TokenizerInput,
            context_length: Optional[int] = None,
            pad: bool = True,
            output_mask: bool = False,
            add_special_tokens: bool = True,
    ) -> TokenizerOutput: ...


def _to_token_list(tokens: TokenIds) -> List[int]:
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.detach().cpu().tolist()
    elif isinstance(tokens, np.ndarray):
        tokens = tokens.tolist()
    return list(tokens)


def _to_token_batch(batch_tokens: BatchTokenIds) -> Iterable[TokenIds]:
    if isinstance(batch_tokens, torch.Tensor):
        return batch_tokens.detach().cpu().tolist()
    if isinstance(batch_tokens, np.ndarray):
        return batch_tokens.tolist()
    return batch_tokens


def _truncate_at_eos(tokens: TokenIds, eos_token_id: Optional[int], stop_at_eos: bool) -> List[int]:
    tokens = _to_token_list(tokens)
    if stop_at_eos and eos_token_id is not None:
        try:
            tokens = tokens[:tokens.index(eos_token_id) + 1]
        except ValueError:
            pass
    return tokens


def _decode_with_backend(
        backend,
        tokens: TokenIds,
        eos_token_id: Optional[int],
        skip_special_tokens: bool,
        stop_at_eos: bool,
) -> str:
    tokens = _truncate_at_eos(tokens, eos_token_id, stop_at_eos)
    return backend.decode(tokens, skip_special_tokens=skip_special_tokens)


def _batch_decode_with_backend(
        backend,
        batch_tokens: BatchTokenIds,
        eos_token_id: Optional[int],
        skip_special_tokens: bool,
        stop_at_eos: bool,
) -> List[str]:
    batch_tokens = _to_token_batch(batch_tokens)
    batch_tokens = [
        _truncate_at_eos(tokens, eos_token_id, stop_at_eos)
        for tokens in batch_tokens
    ]
    return backend.batch_decode(batch_tokens, skip_special_tokens=skip_special_tokens)


def _get_pad_fill_id(pad_token_id: Optional[int]) -> int:
    """Use the reserved pad id when present, otherwise preserve the historical id-0 fill."""
    return 0 if pad_token_id is None else pad_token_id


def _pad_token_sequences(
        all_tokens: List[List[int]],
        context_length: int,
        pad_token_id: int = 0,
        output_mask: bool = False,
) -> TokenizerOutput:
    result = torch.full((len(all_tokens), context_length), pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(result, dtype=torch.bool) if output_mask else None
    for i, tokens in enumerate(all_tokens):
        result[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        if mask is not None:
            mask[i, :len(tokens)] = True
    return (result, mask) if mask is not None else result


@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = " ".join(text.split())
    text = text.strip()
    return text


def _clean_canonicalize(x):
    # basic, remove whitespace, remove punctuation, lower case
    return canonicalize_text(basic_clean(x))


def _clean_lower(x):
    # basic, remove whitespace, lower case
    return whitespace_clean(basic_clean(x)).lower()


def _clean_whitespace(x):
    # basic, remove whitespace
    return whitespace_clean(basic_clean(x))


def _clean_whitespace_underscore(x):
    # case- and punctuation-preserving 'whitespace' clean, plus snake_case separators -> spaces. Useful for
    # verbatim-trained models fed machine-formatted labels (e.g. 'sea_waves' -> 'sea waves') without the
    # lowercasing/punctuation-stripping of 'canonicalize'. Unicode normalization is inherited from basic_clean.
    return whitespace_clean(basic_clean(x).replace("_", " "))


def get_clean_fn(type: str):
    if type == 'canonicalize':
        return _clean_canonicalize
    elif type == 'lower':
        return _clean_lower
    elif type == 'whitespace':
        return _clean_whitespace
    elif type == 'whitespace_underscore':
        return _clean_whitespace_underscore
    else:
        assert False, f"Invalid clean function ({type})."


def canonicalize_text(
    text,
    *,
    keep_punctuation_exact_string=None,
    trans_punctuation: dict = str.maketrans("", "", string.punctuation),
):
    """Returns canonicalized `text` (lowercase and punctuation removed).

    From: https://github.com/google-research/big_vision/blob/53f18caf27a9419231bbf08d3388b07671616d3d/big_vision/evaluators/proj/image_text/prompt_engineering.py#L94

    Args:
      text: string to be canonicalized.
      keep_punctuation_exact_string: If provided, then this exact string kept.
        For example providing '{}' will keep any occurrences of '{}' (but will
        still remove '{' and '}' that appear separately).
    """
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(trans_punctuation)
            for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(trans_punctuation)
    text = text.lower()
    text = " ".join(text.split())
    return text.strip()


class SimpleTokenizer(object):
    def __init__(
            self,
            bpe_path: str = default_bpe(),
            additional_special_tokens: Optional[List[str]] = None,
            context_length: Optional[int] = DEFAULT_CONTEXT_LENGTH,
            clean: str = 'lower',
            reduction_mask: str = ''
    ):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        special_tokens = ['<start_of_text>', '<end_of_text>']
        if additional_special_tokens:
            special_tokens += additional_special_tokens
        vocab.extend(special_tokens)
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {t:t for t in special_tokens}
        special = "|".join(special_tokens)
        self.pat = re.compile(
            special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )
        self.vocab_size = len(self.encoder)
        self.all_special_ids = [self.encoder[t] for t in special_tokens]
        self.sot_token_id = self.all_special_ids[0]
        self.eot_token_id = self.all_special_ids[1]
        self.bos_token_id = self.sot_token_id
        self.eos_token_id = self.eot_token_id
        self.pad_token_id = None
        self.context_length = context_length
        self.clean_fn = get_clean_fn(clean)
        self.reduction_fn = get_reduction_mask_fn(reduction_mask) if reduction_mask else None

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        bpe_tokens = []
        text = self.clean_fn(text)
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        if add_special_tokens:
            bpe_tokens = [self.sot_token_id] + bpe_tokens + [self.eot_token_id]
        return bpe_tokens

    def decode(
            self,
            tokens: TokenIds,
            skip_special_tokens: bool = False,
            stop_at_eos: bool = True,
    ) -> str:
        tokens = _truncate_at_eos(tokens, self.eot_token_id, stop_at_eos)
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.all_special_ids]
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

    def batch_decode(
            self,
            batch_tokens: BatchTokenIds,
            skip_special_tokens: bool = False,
            stop_at_eos: bool = True,
    ) -> List[str]:
        batch_tokens = _to_token_batch(batch_tokens)
        return [
            self.decode(tokens, skip_special_tokens=skip_special_tokens, stop_at_eos=stop_at_eos)
            for tokens in batch_tokens
        ]

    def __call__(
            self,
            texts: TokenizerInput,
            context_length: Optional[int] = None,
            pad: bool = True,
            output_mask: bool = False,
            add_special_tokens: bool = True,
    ) -> TokenizerOutput:
        """ Returns the tokenized representation of given input string(s)

        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize
        context_length : int
            The context length to use; all CLIP models use 77 as the context length
        output_mask : bool
            Also return a [B, L] bool attention mask (True = real token, HF polarity). Length-derived,
            so it stays exact even though this tokenizer pads with 0, a real vocab token.
        add_special_tokens : bool
            Add the start- and end-of-text tokens. Defaults to True for model-ready tokenization.

        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length],
        plus the attention mask when ``output_mask`` is set.
        """
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, 'Please set a valid context length'

        if not pad:
            raise ValueError(
                "SimpleTokenizer does not support variable-length tokenization because token id 0 "
                "is part of the BPE vocabulary. Use TikTokenTokenizer or an HF tokenizer with a "
                "reserved pad token for variable_text=True."
            )

        if self.reduction_fn is not None:
            # use reduction strategy for tokenize if set, otherwise default to truncation below
            result = self.reduction_fn(
                texts,
                context_length=context_length,
                sot_token_id=self.sot_token_id,
                eot_token_id=self.eot_token_id,
                encode_fn=self.encode,
                add_special_tokens=add_special_tokens,
                output_mask=output_mask,
            )
            return result

        all_tokens = [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]
        truncated = []
        for tokens in all_tokens:
            if len(tokens) > context_length:
                tokens = tokens[:context_length]  # Truncate
                if add_special_tokens:
                    tokens[-1] = self.eot_token_id
            truncated.append(tokens)
        all_tokens = truncated
        # The length-derived mask remains exact even though id 0 is both fill and a valid body token.
        return _pad_token_sequences(all_tokens, context_length, output_mask=output_mask)


_tokenizer = SimpleTokenizer()


def decode(
        output_ids: TokenIds,
        skip_special_tokens: bool = False,
        stop_at_eos: bool = True,
) -> str:
    return _tokenizer.decode(
        output_ids,
        skip_special_tokens=skip_special_tokens,
        stop_at_eos=stop_at_eos,
    )


def batch_decode(
        output_ids: BatchTokenIds,
        skip_special_tokens: bool = False,
        stop_at_eos: bool = True,
) -> List[str]:
    return _tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=skip_special_tokens,
        stop_at_eos=stop_at_eos,
    )


def tokenize(
        texts: TokenizerInput,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        add_special_tokens: bool = True,
) -> torch.LongTensor:
    return _tokenizer(
        texts,
        context_length=context_length,
        add_special_tokens=add_special_tokens,
    )


def random_mask_tokenize(
        texts: Union[str, List[str]],
        context_length: int,
        sot_token_id: int,
        eot_token_id: int,
        encode_fn: Callable,
        shuffle: bool = False,
        add_special_tokens: bool = True,
        output_mask: bool = False,
):
    all_tokens = [encode_fn(text) for text in texts]
    reduced_tokens = []
    num_special_tokens = 2 if add_special_tokens else 0
    num_keep = context_length - num_special_tokens

    for tokens in all_tokens:
        tokens = torch.tensor(tokens)
        num_tokens = len(tokens)
        if num_tokens > num_keep:
            indices = torch.randperm(len(tokens))
            indices = indices[:num_keep]
            if not shuffle:
                indices = indices.msort()
            tokens = tokens[indices]
        tokens = tokens.tolist()
        if add_special_tokens:
            tokens = [sot_token_id] + tokens + [eot_token_id]
        reduced_tokens.append(tokens)

    return _pad_token_sequences(reduced_tokens, context_length, output_mask=output_mask)


def simple_mask_tokenize(
        texts: Union[str, List[str]],
        context_length: int,
        sot_token_id: int,
        eot_token_id: int,
        encode_fn: Callable,
        add_special_tokens: bool = True,
        output_mask: bool = False,
):
    all_tokens = [encode_fn(text) for text in texts]
    reduced_tokens = []
    num_special_tokens = 2 if add_special_tokens else 0
    num_keep = context_length - num_special_tokens

    for tokens in all_tokens:
        num_tokens = len(tokens)
        if num_tokens > num_keep:
            start_index = random.randint(0, num_tokens - num_keep)  # high is incl
            tokens = tokens[start_index: start_index + num_keep]
        if add_special_tokens:
            tokens = [sot_token_id] + tokens + [eot_token_id]
        reduced_tokens.append(tokens)

    return _pad_token_sequences(reduced_tokens, context_length, output_mask=output_mask)


def syntax_mask_tokenize(
        texts: Union[str, List[str]],
        context_length: int,
        sot_token_id: int,
        eot_token_id: int,
        encode_fn: Callable,
        add_special_tokens: bool = True,
        output_mask: bool = False,
) -> Union[torch.LongTensor, Tuple[torch.LongTensor, torch.Tensor]]:
    """ Returns the tokenized representation of given input string(s).
    Apply syntax masking before tokenize.
    """
    import nltk
    global _nltk_init
    if not _nltk_init:
        # run them for the first time
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        _nltk_init = True

    def get_order(x):
        if x.startswith('NN'):
            return 1
        elif x.startswith('JJ'):
            return 2
        elif x.startswith('VB'):
            return 3
        else:
            return 4

    # syntax masking
    new_texts = []
    num_special_tokens = 2 if add_special_tokens else 0
    for text in texts:
        list_tokens = nltk.tokenize.word_tokenize(text)
        pos_tags = nltk.pos_tag(list_tokens)
        #  sample the words by get_order method
        order_list = [get_order(tag) for _, tag in pos_tags]
        sorted_ids = np.argsort(np.array(order_list))
        sampled_ids = sorted(sorted_ids[:context_length - num_special_tokens])
        sampled_tokens = np.take(np.array(list_tokens), sampled_ids, axis=0)  # sample the tokens

        new_text = ''
        for token in sampled_tokens:
            new_text = new_text + str(token) + ' '
        new_text = new_text.strip()
        new_texts.append(new_text)
    texts = new_texts

    all_tokens = [encode_fn(text) for text in texts]
    truncated = []

    for tokens in all_tokens:
        if add_special_tokens:
            tokens = [sot_token_id] + tokens + [eot_token_id]
        # still need first truncate because some words produces two tokens
        if len(tokens) > context_length:
            tokens = tokens[:context_length]  # Truncate
            if add_special_tokens:
                tokens[-1] = eot_token_id
        truncated.append(tokens)

    return _pad_token_sequences(truncated, context_length, output_mask=output_mask)


def get_reduction_mask_fn(type: str):
    """ Choose strategy for dropping (masking) tokens to achieve target context length"""
    assert type in ('simple', 'random', 'shuffle', 'syntax')
    if type == 'simple':
        return simple_mask_tokenize  # randomly select block [start:end]
    elif type == 'random':
        return random_mask_tokenize  # randomly drop tokens (keep order)
    elif type == 'shuffle':
        return partial(random_mask_tokenize, shuffle=True)  # randomly drop tokens (shuffle order)
    elif type == 'syntax':
        return syntax_mask_tokenize  # randomly drop prioritized by syntax
    else:
        assert False, F'Unknown type {type}.'


class HFTokenizer:
    """HuggingFace tokenizer wrapper with support for custom tokenization modes"""

    def __init__(
            self,
            tokenizer_name: str,
            context_length: Optional[int] = DEFAULT_CONTEXT_LENGTH,
            clean: str = 'whitespace',
            strip_sep_token: bool = False,
            language: Optional[str] = None,
            cache_dir: Optional[str] = None,
            tokenizer_mode: Optional[str] = None,  # None, 'clips'
            **kwargs
    ):
        self.tokenizer_mode = tokenizer_mode or ''
        self.context_length = context_length
        self.clean_fn = get_clean_fn(clean)
        self.strip_sep_token = strip_sep_token

        # NOTE: Left as example of loading custom tokenizer from file for experimentation
        # if self.tokenizer_mode == 'bert_clips':
        #     self.special_tokens = {
        #         "bos_token": 1,
        #         "eos_token": 2,
        #         "cls_token": 101,
        #         "pad_token": 0
        #     }
        #
        #     # For BERT CLIPS mode with vocab file
        #     from tokenizers import BertWordPieceTokenizer
        #     if tokenizer_name.startswith('hf-hub:'):
        #         from huggingface_hub import hf_hub_download
        #         # Format: hf-hub:repo_id/filename
        #         repo_url = tokenizer_name[7:]
        #         parts = repo_url.split('/')
        #         filename = parts[-1]
        #         repo_id = '/'.join(parts[:-1])
        #         vocab_file = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
        #         self.tokenizer = BertWordPieceTokenizer(lowercase=True)
        #         self.tokenizer = self.tokenizer.from_file(vocab_file)
        #     else:
        #         # Assume tokenizer_name is a local path to a vocab file
        #         self.tokenizer = BertWordPieceTokenizer(lowercase=True)
        #         self.tokenizer = self.tokenizer.from_file(tokenizer_name)

        # Standard HuggingFace tokenizer initialization
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir,
            **kwargs
        )
        # Keep pad_token_id as None when the underlying tokenizer has no reserved pad token: a 0 fallback is a
        # real vocab id in most BPE vocabs, and downstream variable-text setup (get_text_pad_id) relies on None
        # to reject tokenizers that cannot pad safely.
        self.pad_token_id = self.tokenizer.pad_token_id
        # open_clip pooling and masking assume right-padded sequences (causal no-mask path, last/eos index math,
        # cls at position 0); force it for tokenizers that default to left padding (decoder-family).
        self.tokenizer.padding_side = 'right'
        self.eot_token_id = self.tokenizer.eos_token_id
        if self.eot_token_id is None:
            self.eot_token_id = self.tokenizer.sep_token_id
        self.sot_token_id = self.tokenizer.bos_token_id
        if self.sot_token_id is None:
            self.sot_token_id = self.tokenizer.cls_token_id
        self.eos_token_id = self.eot_token_id
        self.bos_token_id = self.sot_token_id
        self.all_special_ids = self.tokenizer.all_special_ids
        self.vocab_size = len(self.tokenizer)

        # Set language function if available
        set_lang_fn = getattr(self.tokenizer, 'set_src_lang_special_tokens', None)
        if callable(set_lang_fn):
            self.set_lang_fn = set_lang_fn
        if language is not None:
            self.set_language(language)

    def save_pretrained(self, dest):
        self.tokenizer.save_pretrained(dest)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        text = self.clean_fn(text)
        if self.tokenizer_mode == 'clips':
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if add_special_tokens:
                tokens = [self.tokenizer.bos_token_id] + tokens + [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.cls_token_id,
                ]
        else:
            tokens = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

        if self.strip_sep_token and self.tokenizer.sep_token_id in tokens:
            fill_id = _get_pad_fill_id(self.pad_token_id)
            tokens = [fill_id if token == self.tokenizer.sep_token_id else token for token in tokens]
        return tokens

    def decode(
            self,
            tokens: TokenIds,
            skip_special_tokens: bool = False,
            stop_at_eos: bool = True,
    ) -> str:
        return _decode_with_backend(
            self.tokenizer,
            tokens,
            self.eot_token_id,
            skip_special_tokens,
            stop_at_eos,
        )

    def batch_decode(
            self,
            batch_tokens: BatchTokenIds,
            skip_special_tokens: bool = False,
            stop_at_eos: bool = True,
    ) -> List[str]:
        return _batch_decode_with_backend(
            self.tokenizer,
            batch_tokens,
            self.eot_token_id,
            skip_special_tokens,
            stop_at_eos,
        )

    def __call__(
            self,
            texts: TokenizerInput,
            context_length: Optional[int] = None,
            pad: bool = True,
            output_mask: bool = False,
            add_special_tokens: bool = True,
    ) -> TokenizerOutput:
        # same cleaning as for default tokenizer, except lowercasing
        # adding lower (for case-sensitive tokenizers) will make it more robust but less sensitive to nuance
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, 'Please set a valid context length in class init or call.'

        texts = [self.clean_fn(text) for text in texts]

        if output_mask and (not pad or self.tokenizer_mode == 'clips'):
            raise ValueError(
                "output_mask=True requires pad=True and the standard tokenizer mode "
                "(variable-length collation derives its own validity)."
            )

        # Handle different tokenization modes
        if self.tokenizer_mode == 'clips':
            return self._clips_tokenize(
                texts,
                context_length,
                pad=pad,
                add_special_tokens=add_special_tokens,
            )
        else:
            # Standard tokenization
            encoded = self.tokenizer(
                texts,
                max_length=context_length,
                padding='max_length' if pad else False,
                truncation=True,
                return_tensors='pt' if pad else None,
                add_special_tokens=add_special_tokens,
            )
            input_ids = encoded.input_ids if pad else encoded["input_ids"]
            attn_mask = encoded.attention_mask.bool() if (pad and output_mask) else None

            if self.strip_sep_token:
                fill_id = _get_pad_fill_id(self.pad_token_id)
                if pad:
                    sep_positions = input_ids == self.tokenizer.sep_token_id
                    input_ids = torch.where(
                        sep_positions,
                        torch.full_like(input_ids, fill_id),
                        input_ids,
                    )
                    if attn_mask is not None:
                        # stripped sep positions carry fill, not content
                        attn_mask = attn_mask & ~sep_positions
                else:
                    input_ids = [
                        [fill_id if token == self.tokenizer.sep_token_id else token for token in tokens]
                        for tokens in input_ids
                    ]

            if not pad:
                return [torch.tensor(tokens, dtype=torch.long) for tokens in input_ids]

            if attn_mask is not None:
                return input_ids, attn_mask

            return input_ids

    def set_language(self, src_lang):
        if hasattr(self, 'set_lang_fn'):
            self.set_lang_fn(src_lang)
        else:
            warnings.warn('Cannot set language for the tokenizer.')

    def _clips_tokenize(
            self,
            texts: List[str],
            context_length: int,
            pad: bool = True,
            add_special_tokens: bool = True,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Use standard HF tokenizer but apply custom post-processing"""
        # Use standard tokenizer without special tokens - we'll add our own
        encoded_outputs = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors=None
        )

        encoded = []
        num_special_tokens = 3 if add_special_tokens else 0
        for tokens in encoded_outputs["input_ids"]:
            tokens = tokens[:context_length - num_special_tokens]
            if add_special_tokens:
                tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
            encoded.append(tokens)

        if not pad:
            # Match the padded contract: the class token terminates the sequence. The body is truncated to
            # context_length - 3 above, so [bos] + body + [eos] + [cls] always fits within context_length.
            if add_special_tokens:
                encoded = [tokens + [self.tokenizer.cls_token_id] for tokens in encoded]
            return [torch.tensor(tokens, dtype=torch.long) for tokens in encoded]

        if not add_special_tokens:
            fill_id = _get_pad_fill_id(self.pad_token_id)
            return _pad_token_sequences(encoded, context_length, pad_token_id=fill_id)

        # Create result tensor and handle padding + class token
        result = torch.zeros(len(encoded), context_length, dtype=torch.long)
        for i, tokens in enumerate(encoded):
            padded_tokens = self._pad_and_add_class_token(
                tokens,
                max_length=context_length,
                pad_token_id=self.tokenizer.pad_token_id,
                cls_token_id=self.tokenizer.cls_token_id,
            )
            result[i, :len(padded_tokens)] = torch.tensor(padded_tokens)

        return result

    def _pad_and_add_class_token(
            self,
            tokens: List[int],
            max_length: int,
            pad_token_id: int = 0,
            cls_token_id: int = 101,
    ) -> List[int]:
        """ Add padding with class token at the end """
        if len(tokens) > max_length - 1:
            tokens = tokens[:max_length - 1]

        # Add padding to reach max_length-1
        if len(tokens) < max_length - 1:
            tokens = tokens + [pad_token_id] * (max_length - 1 - len(tokens))

        # Add class token at the end
        tokens = tokens + [cls_token_id]
        return tokens


class SigLipTokenizer:
    """HuggingFace tokenizer wrapper for SigLIP T5 compatible sentencepiece vocabs

    NOTE: this is not needed in normal library use, but is used to import new sentencepiece tokenizers
    into OpenCLIP. Leaving code here in case future models use new tokenizers.
    """
    VOCAB_FILES = {
        # english, vocab_size=32_000
        "c4-en": "http://storage.googleapis.com/t5-data/vocabs/cc_en.32000/sentencepiece.model",
        # used in multilingual models (mT5, PaLI), vocab_size=250_000
        "mc4": "http://storage.googleapis.com/t5-data/vocabs/mc4.250000.100extra/sentencepiece.model",
        # used in SigLIP2 models, vocab_size=256000
        "gemma": "http://storage.googleapis.com/big_vision/gemma_tokenizer.model",
    }

    def __init__(
            self,
            tokenizer_name: str,
            context_length: Optional[int] = 64,
    ):
        if 'gemma' in tokenizer_name:
            from transformers import GemmaTokenizerFast
            tokenizer_cls = partial(
                GemmaTokenizerFast, padding_side='right', add_bos_token=False, add_eos_token=True)
        else:
            from transformers import T5TokenizerFast
            tokenizer_cls = partial(T5TokenizerFast, extra_ids=0)

        if tokenizer_name in self.VOCAB_FILES:
            # FIXME temporary hack?
            import tempfile
            import fsspec
            vocab_file = self.VOCAB_FILES[tokenizer_name]
            with tempfile.NamedTemporaryFile('wb') as dst:
                with fsspec.open(vocab_file, 'rb') as src:
                    dst.write(src.read())
                self.tokenizer = tokenizer_cls(dst.name, legacy=False)
        else:
            self.tokenizer = tokenizer_cls(tokenizer_name, legacy=False)

        self.tokenizer.pad_token_id = 0 if 'gemma' in tokenizer_name else 1
        self.tokenizer.eos_token_id = 1
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eot_token_id = self.tokenizer.eos_token_id
        self.eos_token_id = self.eot_token_id
        self.sot_token_id = self.tokenizer.bos_token_id
        self.bos_token_id = self.sot_token_id
        self.all_special_ids = self.tokenizer.all_special_ids
        self.vocab_size = len(self.tokenizer)
        self.context_length = context_length

    def save_pretrained(self, dest):
        self.tokenizer.save_pretrained(dest)

    def _clean(self, text: str) -> str:
        return canonicalize_text(basic_clean(text))

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return self.tokenizer.encode(
            self._clean(text),
            add_special_tokens=add_special_tokens,
        )

    def decode(
            self,
            tokens: TokenIds,
            skip_special_tokens: bool = False,
            stop_at_eos: bool = True,
    ) -> str:
        return _decode_with_backend(
            self.tokenizer,
            tokens,
            self.eot_token_id,
            skip_special_tokens,
            stop_at_eos,
        )

    def batch_decode(
            self,
            batch_tokens: BatchTokenIds,
            skip_special_tokens: bool = False,
            stop_at_eos: bool = True,
    ) -> List[str]:
        return _batch_decode_with_backend(
            self.tokenizer,
            batch_tokens,
            self.eot_token_id,
            skip_special_tokens,
            stop_at_eos,
        )

    def __call__(
            self,
            texts: TokenizerInput,
            context_length: Optional[int] = None,
            pad: bool = True,
            output_mask: bool = False,
            add_special_tokens: bool = True,
    ) -> TokenizerOutput:
        # same cleaning as for default tokenizer, except lowercasing
        # adding lower (for case-sensitive tokenizers) will make it more robust but less sensitive to nuance
        if output_mask:
            # SigLIP pads with the eos id (pad == eos), so a value-derived mask cannot separate the
            # terminal eos from padding; no generative config uses this tokenizer.
            raise NotImplementedError("SigLipTokenizer does not support output_mask.")
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, 'Please set a valid context length in class init or call.'

        texts = [self._clean(text) for text in texts]
        output = self.tokenizer(
            texts,
            return_tensors='pt' if pad else None,
            max_length=context_length,
            padding='max_length' if pad else False,
            truncation=True,
            add_special_tokens=add_special_tokens,
        )
        if not pad:
            return [torch.tensor(tokens, dtype=torch.long) for tokens in output.input_ids]
        return output.input_ids


class TikTokenTokenizer:
    """tiktoken-based tokenizer for generative (autoregressive) captioning.

    Wraps an OpenAI ``tiktoken`` BPE encoding (default ``cl100k_base``) for English-priority, fast tokenization.
    tiktoken handles only the caption body; the control ids (EOS, PAD, BOS) are reserved *above* the base
    vocabulary so they never collide with body tokens. Two output modes are supported:

    - ``pad=True`` (default): a fixed ``[N, context_length]`` tensor padded with ``pad_id`` (CLIP-style contract).
    - ``pad=False``: a list of variable-length 1-D tensors ``[BOS] + body + [EOS]`` for per-sample batching
      (used by the NaFlex GenLIP "rows" data path, which pads within the batch).
    """

    def __init__(
            self,
            encoding_name: str = 'cl100k_base',
            context_length: Optional[int] = 256,
            add_bos: bool = True,
            add_eos: bool = True,
            clean: Optional[str] = None,
            bpe_path: Optional[Union[str, os.PathLike]] = None,
            encoding_config_path: Optional[Union[str, os.PathLike]] = None,
    ):
        try:
            import tiktoken
        except ImportError as e:
            raise ImportError("Please install tiktoken to use TikTokenTokenizer (`pip install tiktoken`).") from e

        self.encoding_name = encoding_name
        if bpe_path is not None:
            cfg = self._load_encoding_config(encoding_name, encoding_config_path)
            mergeable_ranks = self._load_tiktoken_bpe(bpe_path)
            special_tokens = {str(k): int(v) for k, v in cfg.get("special_tokens", {}).items()}
            explicit_n_vocab = cfg.get("explicit_n_vocab")
            if explicit_n_vocab is not None:
                max_token_value = max(
                    max(mergeable_ranks.values(), default=0),
                    max(special_tokens.values(), default=0),
                )
                if (
                        len(mergeable_ranks) + len(special_tokens) != explicit_n_vocab or
                        max_token_value != explicit_n_vocab - 1
                ):
                    warnings.warn(
                        f"Ignoring explicit_n_vocab={explicit_n_vocab} in tiktoken encoding config for "
                        f"{encoding_name!r}: it does not match the loaded vocab ({len(mergeable_ranks)} ranks "
                        f"+ {len(special_tokens)} special tokens, max token id {max_token_value}). Expected for "
                        f"gapped-vocab assets exported by older open_clip versions, but can also indicate a "
                        f"truncated or mismatched bpe file.",
                        UserWarning,
                    )
                    explicit_n_vocab = None
            self.enc = tiktoken.Encoding(
                cfg.get("name", encoding_name),
                pat_str=cfg["pat_str"],
                mergeable_ranks=mergeable_ranks,
                special_tokens=special_tokens,
                explicit_n_vocab=explicit_n_vocab,
            )
        else:
            self.enc = tiktoken.get_encoding(encoding_name)
        self.encoding_name = self.enc.name
        self.context_length = context_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        # Optional text cleaning ('canonicalize' / 'lower' / 'whitespace' / 'whitespace_underscore'); default None =
        # verbatim. Verbatim is required for the generative captioning path (cleaning would strip case/punctuation
        # it must reproduce). Contrastive configs can opt in via tokenizer_kwargs: 'canonicalize' (SigLIP-style
        # lowercase + punctuation strip) or 'whitespace_underscore' (case/punctuation-preserving, only snake_case
        # -> spaces -- best for a verbatim-trained model fed machine-formatted labels).
        self.clean_fn = get_clean_fn(clean) if clean else None

        # Reserve control ids above the base vocabulary so they never collide with body tokens.
        base = self.enc.n_vocab
        self.eot_token_id = base  # end-of-text / EOS
        self.pad_token_id = base + 1
        self.bos_token_id = base + 2
        self.sot_token_id = self.bos_token_id  # alias for CLIP-style callers
        self.eos_token_id = self.eot_token_id
        self.vocab_size = base + 3
        self._special_token_text = {
            self.bos_token_id: '<|bos|>',
            self.eot_token_id: '<|eos|>',
            self.pad_token_id: '<|pad|>',
        }
        # tiktoken's registered specials (e.g. <|endoftext|>) sit *below* n_vocab: encode_ordinary never emits
        # them, but the LM head spans them, so decode has to know them to honour skip_special_tokens.
        self._native_special_ids = frozenset(
            self.enc.encode_single_token(name) for name in self.enc.special_tokens_set
        )
        self.all_special_ids = [
            self.eot_token_id, self.pad_token_id, self.bos_token_id, *sorted(self._native_special_ids),
        ]

    @staticmethod
    def _load_tiktoken_bpe(path: Union[str, os.PathLike]) -> Dict[bytes, int]:
        ranks = {}
        with open(path, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                token, rank = line.split()
                ranks[base64.b64decode(token)] = int(rank)
        return ranks

    @staticmethod
    def _dump_tiktoken_bpe(ranks: Dict[bytes, int], path: Union[str, os.PathLike]) -> None:
        with open(path, "wb") as f:
            for token, rank in sorted(ranks.items(), key=lambda x: x[1]):
                f.write(base64.b64encode(token) + b" " + str(rank).encode() + b"\n")

    @staticmethod
    def _load_encoding_config(encoding_name: str, path: Optional[Union[str, os.PathLike]]) -> Dict[str, object]:
        if path is None:
            raise ValueError(
                f"encoding_config_path is required when loading tiktoken encoding {encoding_name!r} from bpe_path."
            )
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_pretrained(self, dest: Union[str, os.PathLike]) -> Dict[str, str]:
        dest = Path(dest)
        asset_dir = dest / "open_clip_tiktoken"
        asset_dir.mkdir(parents=True, exist_ok=True)

        stem = self.encoding_name.replace("/", "_")
        bpe_path = asset_dir / f"{stem}.tiktoken"
        config_path = asset_dir / f"{stem}.json"

        self._dump_tiktoken_bpe(self.enc._mergeable_ranks, bpe_path)
        config = {
            "name": self.enc.name,
            "pat_str": self.enc._pat_str,
            "special_tokens": self.enc._special_tokens,
        }
        # n_vocab is max_token_value + 1 by construction, so tiktoken's explicit_n_vocab entry-count assertion
        # only holds for dense id spaces; omit it for gapped vocabs (e.g. cl100k_base) or reload would fail.
        if len(self.enc._mergeable_ranks) + len(self.enc._special_tokens) == self.enc.n_vocab:
            config["explicit_n_vocab"] = self.enc.n_vocab
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        return {
            "tiktoken_bpe_path": str(bpe_path.relative_to(dest)),
            "tiktoken_config_path": str(config_path.relative_to(dest)),
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        # encode_ordinary ignores any special-token markup in the text, treating it as plain bytes.
        if self.clean_fn is not None:
            text = self.clean_fn(text)
        tokens = self.enc.encode_ordinary(text)
        return self._wrap(tokens) if add_special_tokens else tokens

    def decode(
            self,
            tokens: TokenIds,
            skip_special_tokens: bool = False,
            stop_at_eos: bool = True,
    ) -> str:
        tokens = _truncate_at_eos(tokens, self.eot_token_id, stop_at_eos)
        n_vocab = self.enc.n_vocab
        if skip_special_tokens:
            # Fast path: one filter + one decode. `< n_vocab` drops the reserved ids (above the vocab) and unknown
            # ids (legacy tolerance); the native set drops tiktoken's own specials, which sit below n_vocab.
            native = self._native_special_ids
            return self._decode_body([token for token in tokens if token < n_vocab and token not in native])
        parts = []
        body = []
        for token in tokens:
            if token in self._special_token_text:
                if body:
                    parts.append(self._decode_body(body))
                    body = []
                parts.append(self._special_token_text[token])
            elif token < n_vocab:
                # Native specials stay in the body: tiktoken renders their own text (e.g. <|endoftext|>).
                body.append(token)
            # Preserve the legacy tolerance for unknown ids above the tiktoken vocabulary.
        if body:
            parts.append(self._decode_body(body))
        return ''.join(parts)

    def _decode_body(self, body: List[int]) -> str:
        try:
            return self.enc.decode(body)
        except KeyError:
            # Only reached when a gap id (unused id below n_vocab; cl100k/o200k have a few) slipped in:
            # drop what tiktoken cannot decode and keep the rest.
            chunks = []
            for token in body:
                try:
                    chunks.append(self.enc.decode_single_token_bytes(token))
                except KeyError:
                    pass
            return b''.join(chunks).decode('utf-8', errors='replace')

    def batch_decode(
            self,
            batch_tokens: BatchTokenIds,
            skip_special_tokens: bool = False,
            stop_at_eos: bool = True,
    ) -> List[str]:
        batch_tokens = _to_token_batch(batch_tokens)
        return [
            self.decode(tokens, skip_special_tokens=skip_special_tokens, stop_at_eos=stop_at_eos)
            for tokens in batch_tokens
        ]

    def _wrap(self, ids: List[int]) -> List[int]:
        if self.add_bos:
            ids = [self.bos_token_id] + ids
        if self.add_eos:
            ids = ids + [self.eot_token_id]
        return ids

    def __call__(
            self,
            texts: TokenizerInput,
            context_length: Optional[int] = None,
            pad: bool = True,
            output_mask: bool = False,
            add_special_tokens: bool = True,
    ) -> TokenizerOutput:
        """Tokenize text(s).

        Args:
            texts: A string or list of strings.
            context_length: Max length (including control tokens). Defaults to ``self.context_length``.
                Used for truncation in both modes and for padding in fixed mode.
            pad: When True return a padded ``[N, context_length]`` tensor; when False return a list of
                variable-length 1-D tensors.
            output_mask: Also return a [N, context_length] bool attention mask (True = real token,
                HF polarity). Requires ``pad=True``. Exact: the pad id is reserved above the vocab.
            add_special_tokens: Apply the constructor-configured BOS/EOS template. Defaults to True.
        """
        if isinstance(texts, str):
            texts = [texts]
        context_length = context_length or self.context_length

        if output_mask and not pad:
            raise ValueError("output_mask=True requires pad=True (variable-length collation derives its own validity).")

        all_tokens = [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]
        if context_length is not None:
            truncated = []
            for tokens in all_tokens:
                if len(tokens) > context_length:
                    tokens = tokens[:context_length]
                    if add_special_tokens and self.add_eos:
                        tokens[-1] = self.eot_token_id
                truncated.append(tokens)
            all_tokens = truncated

        if not pad:
            return [torch.tensor(tokens, dtype=torch.long) for tokens in all_tokens]

        assert context_length, 'A context_length is required for padded (pad=True) tokenization.'
        result = torch.full((len(all_tokens), context_length), self.pad_token_id, dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            result[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)

        if output_mask:
            return result, result != self.pad_token_id

        return result
