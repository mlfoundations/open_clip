import pytest
import torch
import open_clip.tokenizer as tokenizer
from open_clip.tokenizer import SimpleTokenizer, TikTokenTokenizer, HFTokenizer

def test_clean_up_tokenization():
    spaced_text = "hello , world ! how are you ?"
    cleaned = tokenizer.clean_up_tokenization(spaced_text)
    assert cleaned == "hello, world! how are you?"
    
    spaced_text2 = "I 'm sorry , I ca n't do that ."
    cleaned2 = tokenizer.clean_up_tokenization(spaced_text2)
    assert cleaned2 == "I'm sorry, I can't do that."

def test_simple_tokenizer_decode():
    tok = SimpleTokenizer()
    text = "hello, world! how are you?"
    
    encoded = tok(text)[0] # padded tensor
    
    # Test default decode
    decoded_default = tok.decode(encoded)
    assert "<start_of_text>" in decoded_default
    assert "<end_of_text>" in decoded_default
    
    # Test skip_special_tokens
    decoded_skip = tok.decode(encoded, skip_special_tokens=True)
    assert "<start_of_text>" not in decoded_skip
    assert "<end_of_text>" not in decoded_skip
    assert "hello , world ! how are you ?" in decoded_skip
    
    # Test with cleanup
    decoded_clean = tok.decode(encoded, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    assert "hello, world! how are you?" in decoded_clean

def test_simple_tokenizer_batch_decode():
    tok = SimpleTokenizer()
    texts = ["hello world", "this is a test"]
    encoded = tok(texts)
    
    # default
    decoded = tok.batch_decode(encoded)
    assert len(decoded) == 2
    assert "<start_of_text>" in decoded[0]
    
    # skip special
    decoded_skip = tok.batch_decode(encoded, skip_special_tokens=True)
    assert decoded_skip[0].strip() == "hello world"
    assert decoded_skip[1].strip() == "this is a test"

def test_global_decode():
    texts = ["a short test"]
    encoded = tokenizer.tokenize(texts)
    
    # decode tensor
    decoded = tokenizer.decode(encoded[0], skip_special_tokens=True)
    assert decoded.strip() == "a short test"
    
    # batch_decode tensor
    decoded_batch = tokenizer.batch_decode(encoded, skip_special_tokens=True)
    assert len(decoded_batch) == 1
    assert decoded_batch[0].strip() == "a short test"

def test_tiktoken_tokenizer_decode():
    pytest.importorskip("tiktoken")
    tok = TikTokenTokenizer()
    
    text = "Hello, world! How are you?"
    encoded = tok(text)[0]
    
    # skip_special_tokens=True
    decoded_skip = tok.decode(encoded.tolist(), skip_special_tokens=True)
    assert decoded_skip == "Hello, world! How are you?"
    
    # skip_special_tokens=False
    decoded_with_special = tok.decode(encoded.tolist(), skip_special_tokens=False)
    assert "<|startoftext|>" in decoded_with_special
    assert "<|endoftext|>" in decoded_with_special
    
    # batch_decode
    batch_encoded = tok(["hello", "world"])
    batch_decoded = tok.batch_decode(batch_encoded, skip_special_tokens=True)
    assert batch_decoded == ["hello", "world"]

def test_hf_tokenizer_decode():
    pytest.importorskip("transformers")
    tok = HFTokenizer("hf-internal-testing/tiny-random-bert")
    
    texts = ["hello world", "test"]
    encoded = tok(texts)
    
    decoded_skip = tok.batch_decode(encoded, skip_special_tokens=True)
    assert "hello world" in decoded_skip[0]
    
    decoded_single = tok.decode(encoded[0], skip_special_tokens=True)
    assert "hello world" in decoded_single
