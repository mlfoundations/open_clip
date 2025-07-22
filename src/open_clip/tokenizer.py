""" CLIP tokenizer

Copied from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import gzip
import html
import os
import random
import string
from functools import lru_cache, partial
from typing import Callable, List, Optional, Union, Dict
import warnings

import ftfy
import numpy as np
import regex as re
import torch

# https://stackoverflow.com/q/62691279
os.environ["TOKENIZERS_PARALLELISM"] = "false"
_nltk_init = False

DEFAULT_CONTEXT_LENGTH = 77  # default context length for OpenAI CLIP


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


def get_clean_fn(type: str):
    if type == 'canonicalize':
        return _clean_canonicalize
    elif type == 'lower':
        return _clean_lower
    elif type == 'whitespace':
        return _clean_whitespace
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

    def encode(self, text):
        bpe_tokens = []
        text = self.clean_fn(text)
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

    def __call__(self, texts: Union[str, List[str]], context_length: Optional[int] = None) -> torch.LongTensor:
        """ Returns the tokenized representation of given input string(s)

        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize
        context_length : int
            The context length to use; all CLIP models use 77 as the context length

        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
        """
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, 'Please set a valid context length'

        if self.reduction_fn is not None:
            # use reduction strategy for tokenize if set, otherwise default to truncation below
            return self.reduction_fn(
                texts,
                context_length=context_length,
                sot_token_id=self.sot_token_id,
                eot_token_id=self.eot_token_id,
                encode_fn=self.encode,
            )

        all_tokens = [[self.sot_token_id] + self.encode(text) + [self.eot_token_id] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]  # Truncate
                tokens[-1] = self.eot_token_id
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result


_tokenizer = SimpleTokenizer()


def decode(output_ids: torch.Tensor):
    output_ids = output_ids.cpu().numpy()
    return _tokenizer.decode(output_ids)


def tokenize(texts: Union[str, List[str]], context_length: int = DEFAULT_CONTEXT_LENGTH) -> torch.LongTensor:
    return _tokenizer(texts, context_length=context_length)


def random_mask_tokenize(
        texts: Union[str, List[str]],
        context_length: int,
        sot_token_id: int,
        eot_token_id: int,
        encode_fn: Callable,
        shuffle: bool = False,
):
    all_tokens = [encode_fn(text) for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        tokens = torch.tensor(tokens)
        num_tokens = len(tokens)
        if num_tokens > context_length - 2:  # 2 for sot and eot token
            num_keep = context_length - 2
            indices = torch.randperm(len(tokens))
            indices = indices[:num_keep]
            if not shuffle:
                indices = indices.msort()
            tokens = tokens[indices]
            num_tokens = num_keep
        result[i, 0] = sot_token_id
        result[i, 1:num_tokens + 1] = tokens
        result[i, num_tokens + 1] = eot_token_id

    return result


def simple_mask_tokenize(
        texts: Union[str, List[str]],
        context_length: int,
        sot_token_id: int,
        eot_token_id: int,
        encode_fn: Callable,
):
    all_tokens = [encode_fn(text) for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        num_tokens = len(tokens)
        if num_tokens > context_length - 2:  # 2 for sot and eot token
            num_keep = context_length - 2
            start_index = random.randint(0, num_tokens - num_keep)  # high is incl
            tokens = tokens[start_index: start_index + num_keep]
        tokens = [sot_token_id] + tokens + [eot_token_id]
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def syntax_mask_tokenize(
        texts: Union[str, List[str]],
        context_length: int,
        sot_token_id: int,
        eot_token_id: int,
        encode_fn: Callable,
) -> torch.LongTensor:
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
    for text in texts:
        list_tokens = nltk.tokenize.word_tokenize(text)
        pos_tags = nltk.pos_tag(list_tokens)
        #  sample the words by get_order method
        order_list = [get_order(tag) for _, tag in pos_tags]
        sorted_ids = np.argsort(np.array(order_list))
        sampled_ids = sorted(sorted_ids[:context_length - 2]) # need 2 slots for sot and eot tokens
        sampled_tokens = np.take(np.array(list_tokens), sampled_ids, axis=0)  # sample the tokens

        new_text = ''
        for token in sampled_tokens:
            new_text = new_text + str(token) + ' '
        new_text = new_text.strip()
        new_texts.append(new_text)
    texts = new_texts

    all_tokens = [[sot_token_id] + encode_fn(text) + [eot_token_id] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        # still need first truncate because some words produces two tokens
        if len(tokens) > context_length:
            tokens = tokens[:context_length]  # Truncate
            tokens[-1] = eot_token_id
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


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

        # Set language function if available
        set_lang_fn = getattr(self.tokenizer, 'set_src_lang_special_tokens', None)
        if callable(set_lang_fn):
            self.set_lang_fn = set_lang_fn
        if language is not None:
            self.set_language(language)

    def save_pretrained(self, dest):
        self.tokenizer.save_pretrained(dest)

    def __call__(self, texts: Union[str, List[str]], context_length: Optional[int] = None) -> torch.Tensor:
        # same cleaning as for default tokenizer, except lowercasing
        # adding lower (for case-sensitive tokenizers) will make it more robust but less sensitive to nuance
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, 'Please set a valid context length in class init or call.'

        texts = [self.clean_fn(text) for text in texts]

        # Handle different tokenization modes
        if self.tokenizer_mode == 'clips':
            return self._clips_tokenize(texts, context_length)
        else:
            # Standard tokenization
            input_ids = self.tokenizer.batch_encode_plus(
                texts,
                return_tensors='pt',
                max_length=context_length,
                padding='max_length',
                truncation=True,
            ).input_ids

            if self.strip_sep_token:
                input_ids = torch.where(
                    input_ids == self.tokenizer.sep_token_id,
                    torch.zeros_like(input_ids),
                    input_ids,
                )

            return input_ids

    def set_language(self, src_lang):
        if hasattr(self, 'set_lang_fn'):
            self.set_lang_fn(src_lang)
        else:
            warnings.warn('Cannot set language for the tokenizer.')

    def _clips_tokenize(self, texts: List[str], context_length: int) -> torch.Tensor:
        """Use standard HF tokenizer but apply custom post-processing"""
        # Use standard tokenizer without special tokens - we'll add our own
        encoded_outputs = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors=None
        )

        encoded = []
        for tokens in encoded_outputs["input_ids"]:
            tokens = tokens[:context_length - 3]  # Leave room for special tokens
            tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
            encoded.append(tokens)

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
        self.context_length = context_length

    def save_pretrained(self, dest):
        self.tokenizer.save_pretrained(dest)

    def __call__(self, texts: Union[str, List[str]], context_length: Optional[int] = None) -> torch.Tensor:
        # same cleaning as for default tokenizer, except lowercasing
        # adding lower (for case-sensitive tokenizers) will make it more robust but less sensitive to nuance
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, 'Please set a valid context length in class init or call.'

        texts = [canonicalize_text(basic_clean(text)) for text in texts]
        output = self.tokenizer(
            texts,
            return_tensors='pt',
            max_length=context_length,
            padding='max_length',
            truncation=True,
        )
        return output.input_ids
