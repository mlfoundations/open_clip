import json
import importlib
from types import SimpleNamespace

import pytest

import open_clip
from open_clip.push_to_hf_hub import save_config_for_hf, save_for_hf


def test_save_config_for_hf_writes_image_preprocess_cfg(tmp_path):
    model = SimpleNamespace(
        visual=SimpleNamespace(
            image_mean=(0.1, 0.2, 0.3),
            image_std=(0.4, 0.5, 0.6),
            preprocess_cfg={
                "interpolation": "bicubic",
                "resize_mode": "shortest",
                "size": 224,
            },
        )
    )
    config_path = tmp_path / "open_clip_config.json"

    save_config_for_hf(model, config_path, model_config={"embed_dim": 32})

    saved = json.loads(config_path.read_text())
    assert saved["model_cfg"] == {"embed_dim": 32}
    assert saved["preprocess_cfg"] == {
        "mean": [0.1, 0.2, 0.3],
        "std": [0.4, 0.5, 0.6],
        "interpolation": "bicubic",
        "resize_mode": "shortest",
    }


def test_save_for_hf_ignores_non_dict_tokenizer_save_result(tmp_path):
    class TupleReturningTokenizer:
        def save_pretrained(self, dest):
            return ("tokenizer_config.json",)

    model = SimpleNamespace(
        visual=SimpleNamespace(
            image_mean=(0.1, 0.2, 0.3),
            image_std=(0.4, 0.5, 0.6),
            preprocess_cfg={},
        )
    )
    model_config = {
        "embed_dim": 32,
        "vision_cfg": {},
        "text_cfg": {"context_length": 77},
    }

    save_for_hf(model, TupleReturningTokenizer(), model_config, tmp_path, skip_weights=True)

    saved = json.loads((tmp_path / "open_clip_config.json").read_text())
    assert saved["model_cfg"] == model_config


def test_save_config_for_hf_allows_audio_model_without_visual(tmp_path):
    model = SimpleNamespace(audio=SimpleNamespace(cfg=SimpleNamespace(model_type="naflexvit")))
    model_config = {
        "embed_dim": 32,
        "audio_cfg": {"model_type": "naflexvit"},
        "text_cfg": {"context_length": 77},
    }
    config_path = tmp_path / "open_clip_config.json"

    save_config_for_hf(model, config_path, model_config=model_config)

    saved = json.loads(config_path.read_text())
    assert saved == {"model_cfg": model_config}


def test_save_for_hf_tiktoken_does_not_write_hf_clip_tokenizer_files(tmp_path):
    pytest.importorskip("tiktoken")
    from open_clip.tokenizer import TikTokenTokenizer

    model = SimpleNamespace(audio=SimpleNamespace(cfg=SimpleNamespace(model_type="naflexvit")))
    model_config = {
        "embed_dim": 32,
        "audio_cfg": {"model_type": "naflexvit"},
        "text_cfg": {
            "context_length": 16,
            "tokenizer_type": "tiktoken",
            "tiktoken_name": "r50k_base",
        },
    }
    tokenizer = TikTokenTokenizer(encoding_name="r50k_base", context_length=16)

    save_for_hf(model, tokenizer, model_config, tmp_path, skip_weights=True)

    saved = json.loads((tmp_path / "open_clip_config.json").read_text())
    saved_text_cfg = saved["model_cfg"]["text_cfg"]
    assert saved_text_cfg["tokenizer_type"] == "tiktoken"
    assert saved_text_cfg["tiktoken_name"] == "r50k_base"
    assert saved_text_cfg["tiktoken_bpe_path"] == "open_clip_tiktoken/r50k_base.tiktoken"
    assert saved_text_cfg["tiktoken_config_path"] == "open_clip_tiktoken/r50k_base.json"
    assert (tmp_path / saved_text_cfg["tiktoken_bpe_path"]).is_file()
    assert (tmp_path / saved_text_cfg["tiktoken_config_path"]).is_file()
    assert not (tmp_path / "tokenizer_config.json").exists()
    assert not (tmp_path / "vocab.json").exists()
    assert not (tmp_path / "merges.txt").exists()


def test_tiktoken_gapped_vocab_export_round_trips(tmp_path):
    pytest.importorskip("tiktoken")
    from open_clip.tokenizer import TikTokenTokenizer

    tokenizer = TikTokenTokenizer(encoding_name="cl100k_base", context_length=16)
    assets = tokenizer.save_pretrained(tmp_path)
    bpe_path = tmp_path / assets["tiktoken_bpe_path"]
    config_path = tmp_path / assets["tiktoken_config_path"]
    config = json.loads(config_path.read_text())

    assert "explicit_n_vocab" not in config

    # Old exports wrote enc.n_vocab here, which is invalid for gapped tiktoken vocabs; the mismatched
    # value must be ignored with a warning rather than tripping tiktoken's count assertion.
    config["explicit_n_vocab"] = tokenizer.enc.n_vocab
    config_path.write_text(json.dumps(config))
    with pytest.warns(UserWarning, match="explicit_n_vocab"):
        reloaded = TikTokenTokenizer(
            encoding_name="cl100k_base",
            context_length=16,
            bpe_path=bpe_path,
            encoding_config_path=config_path,
        )

    assert reloaded.enc.n_vocab == tokenizer.enc.n_vocab
    assert reloaded.eot_token_id == tokenizer.eot_token_id
    assert reloaded("hello", context_length=8).shape == (1, 8)


def test_tiktoken_synthetic_gapped_vocab_round_trips_offline(tmp_path, monkeypatch):
    pytest.importorskip("tiktoken")
    import tiktoken
    from open_clip.tokenizer import TikTokenTokenizer

    # Tiny hand-built encoding with a gap between the top merge rank and the special token id (ids 3-9
    # unused), mirroring cl100k_base/o200k_base structure without downloading a real vocab.
    gapped_enc = tiktoken.Encoding(
        "tiny_gapped",
        pat_str=r"\S+|\s+",
        mergeable_ranks={b"a": 0, b"b": 1, b"ab": 2},
        special_tokens={"<|endoftext|>": 10},
    )
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: gapped_enc)
    tokenizer = TikTokenTokenizer(encoding_name="tiny_gapped", context_length=16)
    assert tokenizer.enc.n_vocab == 11
    assert tokenizer.eot_token_id == 11

    assets = tokenizer.save_pretrained(tmp_path)
    config = json.loads((tmp_path / assets["tiktoken_config_path"]).read_text())
    assert "explicit_n_vocab" not in config

    monkeypatch.setattr(tiktoken, "get_encoding", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError))
    reloaded = TikTokenTokenizer(
        encoding_name="tiny_gapped",
        context_length=16,
        bpe_path=tmp_path / assets["tiktoken_bpe_path"],
        encoding_config_path=tmp_path / assets["tiktoken_config_path"],
    )

    assert reloaded.enc.n_vocab == tokenizer.enc.n_vocab
    assert reloaded.eot_token_id == tokenizer.eot_token_id
    assert reloaded.pad_token_id == tokenizer.pad_token_id
    assert reloaded.encode("abab") == tokenizer.encode("abab")
    assert reloaded("abab", context_length=8).shape == (1, 8)


def test_local_dir_tiktoken_uses_exported_asset(tmp_path, monkeypatch):
    pytest.importorskip("tiktoken")
    from open_clip.tokenizer import TikTokenTokenizer

    model = SimpleNamespace(audio=SimpleNamespace(cfg=SimpleNamespace(model_type="naflexvit")))
    model_config = {
        "embed_dim": 32,
        "audio_cfg": {"model_type": "naflexvit"},
        "text_cfg": {
            "context_length": 16,
            "tokenizer_type": "tiktoken",
            "tiktoken_name": "r50k_base",
        },
    }
    save_for_hf(
        model,
        TikTokenTokenizer(encoding_name="r50k_base", context_length=16),
        model_config,
        tmp_path,
        skip_weights=True,
    )

    import tiktoken

    monkeypatch.setattr(tiktoken, "get_encoding", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError))
    tokenizer = open_clip.get_tokenizer(f"local-dir:{tmp_path}")

    assert type(tokenizer).__name__ == "TikTokenTokenizer"
    assert tokenizer("hello", context_length=8).shape == (1, 8)


def test_local_dir_tiktoken_asset_paths_are_valid_model_cfg_fields(tmp_path, monkeypatch):
    pytest.importorskip("tiktoken")
    from open_clip.tokenizer import TikTokenTokenizer

    model = SimpleNamespace(audio=SimpleNamespace(cfg=SimpleNamespace(model_type="naflexvit")))
    model_config = {
        "embed_dim": 32,
        "audio_cfg": {"model_type": "naflexvit"},
        "text_cfg": {
            "context_length": 16,
            "tokenizer_type": "tiktoken",
            "tiktoken_name": "r50k_base",
        },
    }
    save_for_hf(
        model,
        TikTokenTokenizer(encoding_name="r50k_base", context_length=16),
        model_config,
        tmp_path,
        skip_weights=True,
    )
    resolved_model_cfg = open_clip.get_model_config(f"local-dir:{tmp_path}")
    assert resolved_model_cfg["text_cfg"]["tiktoken_bpe_path"] == "open_clip_tiktoken/r50k_base.tiktoken"
    assert resolved_model_cfg["text_cfg"]["tiktoken_config_path"] == "open_clip_tiktoken/r50k_base.json"
    open_clip.create_model(f"local-dir:{tmp_path}", load_weights=False)

    import tiktoken

    monkeypatch.setattr(tiktoken, "get_encoding", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError))
    tokenizer = open_clip.get_tokenizer(f"local-dir:{tmp_path}")

    assert type(tokenizer).__name__ == "TikTokenTokenizer"
    assert tokenizer("hello", context_length=8).shape == (1, 8)


def test_hf_hub_tiktoken_uses_exported_asset(tmp_path, monkeypatch):
    pytest.importorskip("tiktoken")
    from open_clip.tokenizer import TikTokenTokenizer
    import open_clip.factory as factory

    model = SimpleNamespace(audio=SimpleNamespace(cfg=SimpleNamespace(model_type="naflexvit")))
    model_config = {
        "embed_dim": 32,
        "audio_cfg": {"model_type": "naflexvit"},
        "text_cfg": {
            "context_length": 16,
            "tokenizer_type": "tiktoken",
            "tiktoken_name": "r50k_base",
        },
    }
    save_for_hf(
        model,
        TikTokenTokenizer(encoding_name="r50k_base", context_length=16),
        model_config,
        tmp_path,
        skip_weights=True,
    )
    hf_config = json.loads((tmp_path / "open_clip_config.json").read_text())

    monkeypatch.setattr(factory, "_get_hf_config", lambda model_id, cache_dir=None: hf_config)
    monkeypatch.setattr(
        factory,
        "download_pretrained_from_hf",
        lambda model_id, filename=None, cache_dir=None: str(tmp_path / filename),
    )

    import tiktoken

    monkeypatch.setattr(tiktoken, "get_encoding", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError))
    tokenizer = open_clip.get_tokenizer("hf-hub:rwightman/foo")

    assert type(tokenizer).__name__ == "TikTokenTokenizer"
    assert tokenizer("hello", context_length=8).shape == (1, 8)


def test_push_to_hf_hub_tiktoken_deletes_stale_hf_tokenizer_files(monkeypatch):
    pytest.importorskip("tiktoken")
    hf_push = importlib.import_module("open_clip.push_to_hf_hub")
    from open_clip.tokenizer import TikTokenTokenizer

    class FailHFTokenizer:
        def __init__(self, *args, **kwargs):
            raise AssertionError("tiktoken push must not substitute an HF CLIP tokenizer")

    captured = {}

    def fake_save_for_hf(model, tokenizer, model_config, save_directory, safe_serialization):
        captured["tokenizer"] = tokenizer
        captured["model_config"] = model_config

    def fake_upload_folder(**kwargs):
        captured["upload_kwargs"] = kwargs
        return kwargs

    monkeypatch.setattr(hf_push, "HFTokenizer", FailHFTokenizer)
    monkeypatch.setattr(hf_push, "create_repo", lambda *args, **kwargs: "https://huggingface.co/rwightman/foo")
    monkeypatch.setattr(hf_push, "repo_type_and_id_from_hf_id", lambda repo_url: (None, "rwightman", "foo"))
    monkeypatch.setattr(hf_push, "list_repo_files", lambda repo_id: [])
    monkeypatch.setattr(hf_push, "get_hf_file_metadata", lambda *args, **kwargs: object())
    monkeypatch.setattr(hf_push, "save_for_hf", fake_save_for_hf)
    monkeypatch.setattr(hf_push, "upload_folder", fake_upload_folder)

    model = SimpleNamespace(audio=SimpleNamespace(cfg=SimpleNamespace(model_type="naflexvit")))
    tokenizer = TikTokenTokenizer(encoding_name="r50k_base", context_length=16)
    model_config = {
        "embed_dim": 32,
        "audio_cfg": {"model_type": "naflexvit"},
        "text_cfg": {
            "context_length": 16,
            "tokenizer_type": "tiktoken",
            "tiktoken_name": "r50k_base",
        },
    }

    hf_push.push_to_hf_hub(model, tokenizer, model_config, repo_id="rwightman/foo")

    assert captured["tokenizer"] is tokenizer
    assert captured["model_config"] is model_config
    delete_patterns = captured["upload_kwargs"]["delete_patterns"]
    assert "tokenizer_config.json" in delete_patterns
    assert "vocab.json" in delete_patterns
    assert "merges.txt" in delete_patterns
