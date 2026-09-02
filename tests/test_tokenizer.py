import inspect

import pytest
import torch

from open_clip.tokenizer import HFTokenizer, SigLipTokenizer, SimpleTokenizer, TikTokenTokenizer, Tokenizer


@pytest.mark.parametrize(
    "tokenizer_cls",
    [Tokenizer, SimpleTokenizer, HFTokenizer, SigLipTokenizer, TikTokenTokenizer],
)
def test_tokenizer_special_token_defaults(tokenizer_cls):
    encode_params = inspect.signature(tokenizer_cls.encode).parameters
    call_params = inspect.signature(tokenizer_cls.__call__).parameters
    decode_params = inspect.signature(tokenizer_cls.decode).parameters
    batch_decode_params = inspect.signature(tokenizer_cls.batch_decode).parameters

    assert encode_params["add_special_tokens"].default is False
    assert call_params["add_special_tokens"].default is True
    for params in (decode_params, batch_decode_params):
        assert params["skip_special_tokens"].default is False
        assert params["stop_at_eos"].default is True


def test_simple_tokenizer_special_token_controls():
    tokenizer = SimpleTokenizer(context_length=8)
    body = tokenizer.encode("hello")
    wrapped = tokenizer.encode("hello", add_special_tokens=True)

    assert wrapped == [tokenizer.sot_token_id, *body, tokenizer.eot_token_id]
    model_tokens = tokenizer("hello")[0]
    assert model_tokens[:len(wrapped)].tolist() == wrapped
    assert tokenizer.decode(model_tokens) == "<start_of_text>hello <end_of_text>"
    assert tokenizer.decode(model_tokens, skip_special_tokens=True) == "hello "
    assert tokenizer.decode(model_tokens, stop_at_eos=False).endswith("!!!!!")
    assert tokenizer.batch_decode(model_tokens.unsqueeze(0)) == ["<start_of_text>hello <end_of_text>"]

    body_tokens, body_valid = tokenizer(
        "hello",
        add_special_tokens=False,
        output_mask=True,
    )
    assert body_tokens[0, :len(body)].tolist() == body
    assert body_valid[0].sum().item() == len(body)


@pytest.mark.parametrize("reduction_mask", ["simple", "random", "shuffle"])
def test_simple_tokenizer_body_only_reduction_mask(reduction_mask):
    tokenizer = SimpleTokenizer(context_length=4, reduction_mask=reduction_mask)
    tokens, valid = tokenizer(
        "one two three four five",
        add_special_tokens=False,
        output_mask=True,
    )

    assert tokens.shape == valid.shape == (1, 4)
    assert valid.all()
    assert tokenizer.sot_token_id not in tokens
    assert tokenizer.eot_token_id not in tokens


def test_tiktoken_special_token_controls():
    pytest.importorskip("tiktoken")
    tokenizer = TikTokenTokenizer(context_length=8)
    body = tokenizer.encode("hello")
    wrapped = tokenizer.encode("hello", add_special_tokens=True)

    assert wrapped == [tokenizer.bos_token_id, *body, tokenizer.eot_token_id]
    model_tokens = tokenizer("hello")[0]
    assert tokenizer.decode(model_tokens) == "<|bos|>hello<|eos|>"
    assert tokenizer.decode(model_tokens, skip_special_tokens=True) == "hello"
    assert "<|pad|>" in tokenizer.decode(model_tokens, stop_at_eos=False)
    assert tokenizer.batch_decode(model_tokens.unsqueeze(0)) == ["<|bos|>hello<|eos|>"]
    unknown_id = tokenizer.vocab_size + 10
    assert tokenizer.decode([tokenizer.bos_token_id, *body, unknown_id, tokenizer.eot_token_id]) == (
        "<|bos|>hello<|eos|>"
    )


def test_tiktoken_native_specials_and_gap_ids():
    pytest.importorskip("tiktoken")
    # r50k_base: dense vocab with one native special (<|endoftext|>) registered *below* n_vocab.
    tokenizer = TikTokenTokenizer(encoding_name="r50k_base", context_length=32)
    native_eot = tokenizer.enc.eot_token
    assert native_eot < tokenizer.enc.n_vocab
    assert native_eot in tokenizer.all_special_ids
    assert native_eot not in tokenizer.encode("<|endoftext|>")  # encode_ordinary treats markup as plain text
    body = tokenizer.encode("hello")
    seq = [tokenizer.bos_token_id, *body, native_eot, *body, tokenizer.eot_token_id, tokenizer.pad_token_id]
    assert tokenizer.decode(seq) == "<|bos|>hello<|endoftext|>hello<|eos|>"
    assert tokenizer.decode(seq, skip_special_tokens=True) == "hellohello"

    # cl100k_base: gapped vocab. An unused id below n_vocab must be dropped by decode, not raise.
    tokenizer = TikTokenTokenizer(encoding_name="cl100k_base", context_length=32)
    ranks = set(tokenizer.enc._mergeable_ranks.values())
    # (not enc.is_special_token: broken in tiktoken 0.12, references a missing attribute)
    gap = next(i for i in range(tokenizer.enc.n_vocab) if i not in ranks and i not in tokenizer._native_special_ids)
    with pytest.raises(KeyError):
        tokenizer.enc.decode([gap])
    seq = [tokenizer.bos_token_id, *tokenizer.encode("hello world"), gap, *tokenizer.encode("!"), tokenizer.eot_token_id]
    assert tokenizer.decode(seq) == "<|bos|>hello world!<|eos|>"
    assert tokenizer.decode(seq, skip_special_tokens=True) == "hello world!"


class _FakeHFBackend:
    eos_token_id = 2

    def __init__(self):
        self.encode_add_special_tokens = None

    def encode(self, text, add_special_tokens=True):
        self.encode_add_special_tokens = add_special_tokens
        return [10, self.eos_token_id] if add_special_tokens else [10]

    def decode(self, tokens, skip_special_tokens=False):
        return f"{list(tokens)}:{skip_special_tokens}"

    def batch_decode(self, batch_tokens, skip_special_tokens=False):
        return [f"{list(tokens)}:{skip_special_tokens}" for tokens in batch_tokens]


def test_hf_tokenizer_encode_decode_controls_delegate_cleanly():
    tokenizer = HFTokenizer.__new__(HFTokenizer)
    tokenizer.tokenizer = _FakeHFBackend()
    tokenizer.tokenizer_mode = ""
    tokenizer.clean_fn = lambda text: text
    tokenizer.strip_sep_token = False
    tokenizer.eot_token_id = tokenizer.tokenizer.eos_token_id

    assert tokenizer.encode("hello") == [10]
    assert tokenizer.tokenizer.encode_add_special_tokens is False
    assert tokenizer.encode("hello", add_special_tokens=True) == [10, 2]
    assert tokenizer.decode(torch.tensor([10, 2, 99])) == "[10, 2]:False"
    assert tokenizer.decode([10, 2, 99], skip_special_tokens=True) == "[10, 2]:True"
    assert tokenizer.batch_decode([[10, 2, 99], [11]]) == ["[10, 2]:False", "[11]:False"]
    batch_tokens = torch.tensor([[10, 2, 99], [11, 2, 98]])
    assert tokenizer.batch_decode(batch_tokens) == ["[10, 2]:False", "[11, 2]:False"]
