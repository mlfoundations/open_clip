import torch
import pytest


def test_whisper_encoder_uses_gradient_checkpointing(monkeypatch):
    from open_clip.audio import whisper
    from open_clip.audio.whisper import WhisperEncoder

    def fake_log_mel_spectrogram(audio, n_mels=80, padding=0, device=None):
        return torch.randn(audio.shape[0], n_mels, 6, device=device or audio.device)

    checkpoint_calls = []
    real_checkpoint = whisper.checkpoint.checkpoint

    def wrapped_checkpoint(function, *args, **kwargs):
        checkpoint_calls.append(function)
        return real_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(whisper, "log_mel_spectrogram", fake_log_mel_spectrogram)
    monkeypatch.setattr(whisper.checkpoint, "checkpoint", wrapped_checkpoint)

    model = WhisperEncoder(
        n_mels=80,
        n_ctx=3,
        n_state=8,
        n_head=2,
        n_layer=1,
        output_dim=4,
        avg_pool=False,
        add_audio_bos_eos_token=False,
    )
    model.train()
    model.set_grad_checkpointing(True)

    out = model({"waveform": torch.randn(2, 32)})["embedding"]
    out.sum().backward()

    assert out.shape == (2, 3, 4)
    assert len(checkpoint_calls) == 1


def test_htsat_basic_layer_checkpointing_handles_tuple_output():
    torchlibrosa = pytest.importorskip("torchlibrosa", reason="HTSAT checkpointing test requires torchlibrosa")
    assert torchlibrosa is not None
    from open_clip.audio.htsat import BasicLayer

    layer = BasicLayer(
        dim=4,
        input_resolution=(2, 2),
        depth=2,
        num_heads=1,
        window_size=2,
        use_checkpoint=True,
    )
    layer.train()

    x = torch.randn(2, 4, 4, requires_grad=True)
    out, attn = layer(x)
    out.sum().backward()

    assert out.shape == (2, 4, 4)
    assert attn is not None
    assert x.grad is not None
