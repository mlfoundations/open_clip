""" CLIP tokenizer

Copied from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import gzip
import html
import os
import string
from functools import lru_cache
from typing import Optional, List, Union

import ftfy
import numpy as np
import regex as re
import torch

# https://stackoverflow.com/q/62691279
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
    text = re.sub(r'\s+', ' ', text)
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


def canonicalize_text(text, *, keep_punctuation_exact_string=None):
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
            part.translate(str.maketrans("", "", string.punctuation))
            for part in text.split(keep_punctuation_exact_string))
    else:
        text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class SimpleTokenizer(object):
    def __init__(
            self,
            bpe_path: str = default_bpe(),
            special_tokens=None,
            clean: str = 'lower',
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
        if not special_tokens:
            special_tokens = ['<start_of_text>', '<end_of_text>']
        else:
            special_tokens = ['<start_of_text>', '<end_of_text>'] + special_tokens
        vocab.extend(special_tokens)
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {t:t for t in special_tokens}
        special = "|".join(special_tokens)
        self.pat = re.compile(special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)
        self.clean_fn = get_clean_fn(clean)
        self.vocab_size = len(self.encoder)
        self.all_special_ids = [self.encoder[t] for t in special_tokens]

    @staticmethod
    def create(text_mask='', **kwargs) -> 'SimpleTokenizer':
        if text_mask == 'simple':
            return SimpleMaskTokenizer(**kwargs)
        elif text_mask == 'random':
            return RandomMaskTokenizer(**kwargs)
        elif text_mask == 'syntax':
            return SyntaxMaskTokenizer(**kwargs)
        else:
            return SimpleTokenizer(**kwargs)

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
                except:
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

    def __call__(self, texts: Union[str, List[str]], context_length: int = 77) -> torch.LongTensor:
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

        sot_token = self.encoder["<start_of_text>"]
        eot_token = self.encoder["<end_of_text>"]
        all_tokens = [[sot_token] + self.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]  # Truncate
                tokens[-1] = eot_token
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result


_tokenizer = SimpleTokenizer()


def decode(output_ids: torch.Tensor):
    output_ids = output_ids.cpu().numpy()
    return _tokenizer.decode(output_ids)


def tokenize(texts: Union[str, List[str]], context_length: int = 77) -> torch.LongTensor:
    return _tokenizer(texts, context_length=context_length)


class HFTokenizer:
    """HuggingFace tokenizer wrapper"""

    def __init__(
            self,
            tokenizer_name: str,
            clean: str = 'whitespace',
            strip_sep_token=False,
    ):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.clean_fn = get_clean_fn(clean)
        self.strip_sep_token = strip_sep_token

    def save_pretrained(self, dest):
        self.tokenizer.save_pretrained(dest)

    def __call__(self, texts: Union[str, List[str]], context_length: int = 77) -> torch.Tensor:
        # same cleaning as for default tokenizer, except lowercasing
        # adding lower (for case-sensitive tokenizers) will make it more robust but less sensitive to nuance
        if isinstance(texts, str):
            texts = [texts]

        texts = [self.clean_fn(text) for text in texts]
        input_ids = self.tokenizer(
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


class RandomMaskTokenizer(SimpleTokenizer):

    def __call__(self, texts: Union[str, List[str]], context_length: int = 77) -> torch.LongTensor:
        """
        Returns the tokenized representation of given input string(s)

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

        sot_token = self.encoder["<start_of_text>"]
        eot_token = self.encoder["<end_of_text>"]
        all_tokens = [self.encode(text) for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length - 2:  # 2 for sot and eot token
                indices = np.random.permutation(len(tokens)).tolist()
                indices = indices[:context_length - 2]
                tokens = tokens[indices]
            tokens = [sot_token] + tokens + [eot_token]
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result


class SimpleMaskTokenizer(SimpleTokenizer):
    def __call__(self, texts: Union[str, List[str]], context_length: int = 77) -> torch.LongTensor:
        """
        Returns the tokenized representation of given input string(s)

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

        sot_token = self.encoder["<start_of_text>"]
        eot_token = self.encoder["<end_of_text>"]
        all_tokens = [self.encode(text) for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length - 2: # 2 for sot and eot token
                start_index = np.random.randint(len(tokens) - context_length + 3)
                tokens = tokens[start_index : start_index + context_length - 2]
            tokens = [sot_token] + tokens + [eot_token]
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result


class SyntaxMaskTokenizer(SimpleTokenizer):

    def __call__(self, texts: Union[str, List[str]], context_length: int = 77) -> torch.LongTensor:
        """
        Returns the tokenized representation of given input string(s).
        Apply syntax masking before tokenize.

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
        import nltk
        if not hasattr(self, '_nltk_init'):
            # run them for the first time
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
            self._nltk_init = True

        if isinstance(texts, str):
            texts = [texts]

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
            # sample the tokens and convert to tf.tensor
            sampled_tokens = np.take(np.array(list_tokens), sampled_ids, axis=0)

            new_text = ''
            for token in sampled_tokens:
                new_text = new_text + str(token) + ' '
            new_text = new_text.strip()
            new_texts.append(new_text)
        texts = new_texts

        sot_token = self.encoder["<start_of_text>"]
        eot_token = self.encoder["<end_of_text>"]
        all_tokens = [[sot_token] + self.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            # still need first truncate because some words produces two tokens
            if len(tokens) > context_length:
                tokens = tokens[:context_length]  # Truncate
                tokens[-1] = eot_token
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result


class SigLipTokenizer:
    """HuggingFace tokenizer wrapper for SigLIP T5 compatible sentencepiece vocabs
    """
    VOCAB_FILES = {
        # english, vocab_size=32_000
        "c4-en": "http://storage.googleapis.com/t5-data/vocabs/cc_en.32000/sentencepiece.model",
        # used in multilingual models (mT5, PaLI), vocab_size=250_000
        "mc4": "http://storage.googleapis.com/t5-data/vocabs/mc4.250000.100extra/sentencepiece.model",
    }

    def __init__(self, tokenizer_name: str):
        from transformers import T5TokenizerFast

        if tokenizer_name in self.VOCAB_FILES:
            # FIXME temporary hack?
            import fsspec
            import tempfile
            vocab_file = self.VOCAB_FILES[tokenizer_name]
            with tempfile.NamedTemporaryFile('wb') as dst:
                with fsspec.open(vocab_file, 'rb') as src:
                    dst.write(src.read())
                self.tokenizer = T5TokenizerFast(dst.name, legacy=False)
        else:
            self.tokenizer = T5TokenizerFast(tokenizer_name, legacy=False)

        self.tokenizer.pad_token_id = 1
        self.tokenizer.eos_token_id = 1

    def save_pretrained(self, dest):
        self.tokenizer.save_pretrained(dest)

    def __call__(self, texts: Union[str, List[str]], context_length: int = 64) -> torch.Tensor:
        # same cleaning as for default tokenizer, except lowercasing
        # adding lower (for case-sensitive tokenizers) will make it more robust but less sensitive to nuance
        if isinstance(texts, str):
            texts = [texts]
        texts = [canonicalize_text(basic_clean(text)) for text in texts]
        output = self.tokenizer(
            texts,
            return_tensors='pt',
            max_length=context_length,
            padding='max_length',
            truncation=True,
        )
        return output.input_ids
