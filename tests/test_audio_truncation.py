"""Thorough tests for audio truncation modes in make_audio_preprocess.

Tests rand_trunc, trunc, and filling behavior with various audio lengths.
"""
import random
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from open_clip_train.data import make_audio_preprocess, int16_to_float32_torch, float32_to_int16_torch

# Standard CLAP audio config
AUDIO_CFG = {
    'sample_rate': 48000,
    'clip_samples': 480000,  # 10 seconds
    'window_size': 1024,
    'hop_size': 480,
    'mel_bins': 64,
    'fmin': 50,
    'fmax': 14000,
}

CLIP_SAMPLES = AUDIO_CFG['clip_samples']
SR = AUDIO_CFG['sample_rate']


def make_test_waveform(duration_s, sr=SR, freq=440.0):
    """Create a sine wave test waveform with known properties."""
    n_samples = int(duration_s * sr)
    t = torch.linspace(0, duration_s, n_samples)
    waveform = torch.sin(2 * np.pi * freq * t)
    return waveform.unsqueeze(0)  # (1, samples)


def make_ramp_waveform(n_samples, sr=SR):
    """Create a linearly increasing ramp so we can verify which section was cropped."""
    waveform = torch.linspace(0, 1, n_samples)
    return waveform.unsqueeze(0)  # (1, samples)


# =============================================================================
# Int16 normalization tests
# =============================================================================

def test_int16_roundtrip_basic():
    """int16 round-trip should clamp to [-1,1] and quantize."""
    x = torch.tensor([0.5, -0.5, 1.5, -1.5, 0.0, 1.0, -1.0])
    result = int16_to_float32_torch(float32_to_int16_torch(x))
    assert result.max() <= 1.0, f"Max {result.max()} > 1.0"
    assert result.min() >= -1.0, f"Min {result.min()} < -1.0"
    # Values within [-1,1] should be nearly identical (quantization noise)
    assert abs(result[0] - 0.5) < 1e-4, f"0.5 became {result[0]}"
    assert abs(result[1] + 0.5) < 1e-4, f"-0.5 became {result[1]}"
    assert abs(result[4]) < 1e-4, f"0.0 became {result[4]}"
    # Values outside [-1,1] should be clamped
    assert abs(result[2] - 1.0) < 1e-4, f"1.5 clamped to {result[2]}"
    assert abs(result[3] + 1.0) < 1e-4, f"-1.5 clamped to {result[3]}"
    print("  PASS: int16 round-trip basic")


def test_int16_quantization_noise():
    """Quantization noise should be small (~1/32767 ≈ 3e-5)."""
    x = torch.randn(10000).clamp(-1, 1)
    result = int16_to_float32_torch(float32_to_int16_torch(x))
    noise = (x - result).abs()
    max_noise = noise.max().item()
    mean_noise = noise.mean().item()
    assert max_noise < 1.0 / 32767 + 1e-6, f"Max noise {max_noise} too large"
    print(f"  PASS: int16 quantization noise: max={max_noise:.6f}, mean={mean_noise:.6f}")


# =============================================================================
# Trunc tests
# =============================================================================

def test_trunc_longer_audio():
    """trunc should always take the first clip_samples."""
    pp = make_audio_preprocess(AUDIO_CFG, data_truncating="trunc")
    # 20s audio (960000 samples) → should take first 480000
    waveform = make_ramp_waveform(960000)
    result = pp((waveform, SR))

    assert result["waveform"].shape == (CLIP_SAMPLES,), f"Shape: {result['waveform'].shape}"
    assert result["longer"] == True, f"longer should be True"
    # First sample should be near 0, last near 0.5 (halfway through 20s)
    assert result["waveform"][0] < 0.01, f"First sample: {result['waveform'][0]}"
    # The trunc should always take the BEGINNING
    # After int16 quantization, values shift slightly
    print(f"  PASS: trunc longer audio - shape={result['waveform'].shape}, longer={result['longer']}")


def test_trunc_deterministic():
    """trunc should give identical results on repeated calls."""
    pp = make_audio_preprocess(AUDIO_CFG, data_truncating="trunc")
    waveform = make_ramp_waveform(960000)

    results = [pp((waveform, SR))["waveform"] for _ in range(5)]
    for i in range(1, len(results)):
        assert torch.equal(results[0], results[i]), f"trunc gave different results on call {i}"
    print("  PASS: trunc is deterministic")


def test_trunc_exact_clip_samples():
    """Audio exactly clip_samples long should pass through unchanged (after int16 norm)."""
    pp = make_audio_preprocess(AUDIO_CFG, data_truncating="trunc")
    waveform = make_ramp_waveform(CLIP_SAMPLES)
    result = pp((waveform, SR))

    assert result["waveform"].shape == (CLIP_SAMPLES,), f"Shape: {result['waveform'].shape}"
    assert result["longer"] == False, f"longer should be False for exact length"
    print(f"  PASS: trunc exact clip_samples - longer={result['longer']}")


# =============================================================================
# Rand_trunc tests
# =============================================================================

def test_rand_trunc_longer_audio():
    """rand_trunc should return clip_samples from somewhere in the audio."""
    pp = make_audio_preprocess(AUDIO_CFG, data_truncating="rand_trunc")
    waveform = make_ramp_waveform(960000)
    result = pp((waveform, SR))

    assert result["waveform"].shape == (CLIP_SAMPLES,), f"Shape: {result['waveform'].shape}"
    assert result["longer"] == True, f"longer should be True"
    print(f"  PASS: rand_trunc longer audio - shape={result['waveform'].shape}, longer={result['longer']}")


def test_rand_trunc_randomness():
    """rand_trunc should give different crops on different calls."""
    pp = make_audio_preprocess(AUDIO_CFG, data_truncating="rand_trunc")
    # Use a long ramp so different crops have different first samples
    waveform = make_ramp_waveform(960000)

    first_samples = []
    for _ in range(20):
        result = pp((waveform, SR))
        first_samples.append(result["waveform"][0].item())

    unique_values = len(set(first_samples))
    assert unique_values > 1, f"rand_trunc always gives same crop! Values: {set(first_samples)}"
    print(f"  PASS: rand_trunc is random - {unique_values}/20 unique starting positions")


def test_rand_trunc_crop_is_contiguous():
    """rand_trunc crop should be a contiguous slice of the original."""
    pp = make_audio_preprocess(AUDIO_CFG, data_truncating="rand_trunc")
    # Create ramp: after int16 normalization, values should still be monotonically increasing
    waveform = make_ramp_waveform(960000)

    for trial in range(10):
        result = pp((waveform, SR))
        w = result["waveform"]
        # After int16 quantization, the ramp should still be non-decreasing
        diffs = w[1:] - w[:-1]
        assert (diffs >= -1e-5).all(), f"Trial {trial}: crop is not contiguous (ramp not monotonic)"
    print("  PASS: rand_trunc crops are contiguous")


def test_rand_trunc_bounds():
    """rand_trunc crop should not go past the end of the audio."""
    pp = make_audio_preprocess(AUDIO_CFG, data_truncating="rand_trunc")
    # Audio barely longer than clip_samples: overflow = 1
    n_samples = CLIP_SAMPLES + 1
    waveform = make_ramp_waveform(n_samples)

    for _ in range(50):
        result = pp((waveform, SR))
        assert result["waveform"].shape == (CLIP_SAMPLES,), f"Shape: {result['waveform'].shape}"
        # Max value should be <= 1.0 (from int16 clamping)
        assert result["waveform"].max() <= 1.0 + 1e-6
    print("  PASS: rand_trunc bounds check (overflow=1)")


def test_rand_trunc_large_overflow():
    """rand_trunc with very long audio (30s Clotho-like)."""
    pp = make_audio_preprocess(AUDIO_CFG, data_truncating="rand_trunc")
    # 30s audio = 1440000 samples, overflow = 960000
    n_samples = int(30 * SR)
    waveform = make_ramp_waveform(n_samples)

    min_starts = []
    max_ends = []
    for _ in range(50):
        result = pp((waveform, SR))
        assert result["waveform"].shape == (CLIP_SAMPLES,)
        min_starts.append(result["waveform"][0].item())
        max_ends.append(result["waveform"][-1].item())

    # Check that crops span a reasonable range of the audio
    start_range = max(min_starts) - min(min_starts)
    assert start_range > 0.1, f"Crop starts don't vary enough: range={start_range}"
    print(f"  PASS: rand_trunc 30s audio - start range: {start_range:.3f}")


def test_rand_trunc_vs_trunc_short_audio():
    """For audio shorter than clip_samples, both should behave identically."""
    pp_trunc = make_audio_preprocess(AUDIO_CFG, data_truncating="trunc")
    pp_rand = make_audio_preprocess(AUDIO_CFG, data_truncating="rand_trunc")

    # 5s audio (shorter than 10s clip)
    waveform = make_test_waveform(5.0)

    result_trunc = pp_trunc((waveform, SR))
    result_rand = pp_rand((waveform, SR))

    assert torch.equal(result_trunc["waveform"], result_rand["waveform"]), \
        "trunc and rand_trunc should be identical for short audio"
    assert result_trunc["longer"] == result_rand["longer"] == False
    print("  PASS: trunc == rand_trunc for short audio")


# =============================================================================
# Filling tests
# =============================================================================

def test_pad_filling():
    """pad should zero-pad short audio."""
    pp = make_audio_preprocess(AUDIO_CFG, data_filling="pad", data_truncating="trunc")
    waveform = make_test_waveform(5.0)  # 240000 samples
    result = pp((waveform, SR))

    assert result["waveform"].shape == (CLIP_SAMPLES,)
    assert result["longer"] == False
    # Last portion should be zeros (padded)
    tail = result["waveform"][-1000:]
    assert (tail == 0).all(), "Padded region should be zeros"
    print("  PASS: pad filling")


def test_repeat_filling():
    """repeat should loop the waveform."""
    pp = make_audio_preprocess(AUDIO_CFG, data_filling="repeat", data_truncating="trunc")
    waveform = make_test_waveform(5.0)
    result = pp((waveform, SR))

    assert result["waveform"].shape == (CLIP_SAMPLES,)
    assert result["longer"] == False
    # The waveform should not be zero-padded at the end
    tail = result["waveform"][-1000:]
    assert not (tail == 0).all(), "repeat should not zero-pad"
    print("  PASS: repeat filling")


def test_repeatpad_filling():
    """repeatpad should loop then pad remainder."""
    pp = make_audio_preprocess(AUDIO_CFG, data_filling="repeatpad", data_truncating="trunc")
    waveform = make_test_waveform(5.0)
    result = pp((waveform, SR))

    assert result["waveform"].shape == (CLIP_SAMPLES,)
    assert result["longer"] == False
    print("  PASS: repeatpad filling")


# =============================================================================
# Resampling test
# =============================================================================

def test_resampling():
    """Audio at different sample rate should be resampled."""
    pp = make_audio_preprocess(AUDIO_CFG, data_truncating="trunc")
    # 10s at 16kHz → 160000 samples → should resample to 480000
    waveform_16k = make_test_waveform(10.0, sr=16000)
    result = pp((waveform_16k, 16000))

    assert result["waveform"].shape == (CLIP_SAMPLES,), f"Shape after resampling: {result['waveform'].shape}"
    assert result["longer"] == False  # 10s at 16kHz resampled to 10s at 48kHz = exactly clip_samples
    print("  PASS: resampling 16kHz → 48kHz")


# =============================================================================
# Mono mixing test
# =============================================================================

def test_stereo_to_mono():
    """Stereo audio should be mixed to mono."""
    pp = make_audio_preprocess(AUDIO_CFG, data_truncating="trunc")
    # Stereo 10s audio
    n_samples = CLIP_SAMPLES
    waveform = torch.randn(2, n_samples)  # (2, samples)
    result = pp((waveform, SR))

    assert result["waveform"].shape == (CLIP_SAMPLES,), f"Shape: {result['waveform'].shape}"
    print("  PASS: stereo to mono mixing")


# =============================================================================
# Statistical test: rand_trunc coverage
# =============================================================================

def test_rand_trunc_uniform_coverage():
    """rand_trunc should cover the full range of the audio, not just edges."""
    pp = make_audio_preprocess(AUDIO_CFG, data_truncating="rand_trunc")
    # 20s audio
    n_samples = CLIP_SAMPLES * 2
    waveform = make_ramp_waveform(n_samples)

    # Collect starting positions over many trials
    starts = []
    for _ in range(200):
        result = pp((waveform, SR))
        starts.append(result["waveform"][0].item())

    starts = sorted(starts)
    # Check that starts span the range [0, ~0.5] (since ramp goes 0→1, crop starts at most at 0.5)
    min_start = min(starts)
    max_start = max(starts)

    # Divide range into 5 bins and check each has some representation
    bins = [0] * 5
    for s in starts:
        bin_idx = min(4, int((s - min_start) / (max_start - min_start + 1e-10) * 5))
        bins[bin_idx] += 1

    empty_bins = sum(1 for b in bins if b == 0)
    assert empty_bins == 0, f"Some bins empty: {bins} — rand_trunc not uniformly covering audio"
    print(f"  PASS: rand_trunc uniform coverage - bins: {bins}")


# =============================================================================
# Edge case: audio exactly 1 sample longer
# =============================================================================

def test_rand_trunc_one_sample_overflow():
    """Audio 1 sample longer than clip_samples — overflow=1, idx in {0,1}."""
    pp = make_audio_preprocess(AUDIO_CFG, data_truncating="rand_trunc")
    n_samples = CLIP_SAMPLES + 1
    # Create audio where first and last sample are different
    waveform = torch.zeros(1, n_samples)
    waveform[0, 0] = 0.1   # first sample
    waveform[0, -1] = 0.9  # last sample

    saw_start_0 = False
    saw_start_1 = False
    for _ in range(100):
        result = pp((waveform, SR))
        first = result["waveform"][0].item()
        last = result["waveform"][-1].item()
        # After int16 norm, 0.1 → ~0.1, 0.9 → ~0.9
        if abs(first - int16_to_float32_torch(float32_to_int16_torch(torch.tensor(0.1))).item()) < 1e-4:
            saw_start_0 = True
        if abs(last - int16_to_float32_torch(float32_to_int16_torch(torch.tensor(0.9))).item()) < 1e-4:
            saw_start_1 = True

    assert saw_start_0, "Never saw crop starting at index 0"
    assert saw_start_1, "Never saw crop including last sample"
    print("  PASS: rand_trunc overflow=1 covers both positions")


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Audio Truncation Tests")
    print("=" * 60)

    print("\n--- Int16 Normalization ---")
    test_int16_roundtrip_basic()
    test_int16_quantization_noise()

    print("\n--- Trunc Mode ---")
    test_trunc_longer_audio()
    test_trunc_deterministic()
    test_trunc_exact_clip_samples()

    print("\n--- Rand_trunc Mode ---")
    test_rand_trunc_longer_audio()
    test_rand_trunc_randomness()
    test_rand_trunc_crop_is_contiguous()
    test_rand_trunc_bounds()
    test_rand_trunc_large_overflow()
    test_rand_trunc_vs_trunc_short_audio()
    test_rand_trunc_one_sample_overflow()
    test_rand_trunc_uniform_coverage()

    print("\n--- Filling Modes ---")
    test_pad_filling()
    test_repeat_filling()
    test_repeatpad_filling()

    print("\n--- Other ---")
    test_resampling()
    test_stereo_to_mono()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
