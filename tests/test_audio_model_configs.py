import pytest

import open_clip
from open_clip.clap_model import CLAP


pytest.importorskip("torchlibrosa", reason="HTSAT config instantiation tests require torchlibrosa")


def test_clap_htsat_tiny_config_instantiates_with_audio_deps():
    model = open_clip.create_model("CLAP-HTSAT-tiny", pretrained=None, device="cpu")

    assert isinstance(model, CLAP)
    assert model.audio.cfg.model_type == "HTSAT"
    assert model.audio.cfg.model_name == "tiny"
