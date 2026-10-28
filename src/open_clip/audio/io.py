"""Audio file decoding: soundfile (libsndfile) first, TorchCodec (FFmpeg) as an optional fallback.

soundfile reads WAV / FLAC / OGG (Vorbis, Opus) / MP3 with no system dependencies; TorchCodec, if installed,
covers anything else FFmpeg can decode (AAC / M4A, ...).
"""
import io
import os
from typing import BinaryIO, Tuple, Union

import torch

AudioSource = Union[str, os.PathLike, bytes, bytearray, BinaryIO]


def _as_file(source: AudioSource):
    if isinstance(source, (bytes, bytearray)):
        return io.BytesIO(source)
    if hasattr(source, "seek"):
        source.seek(0)
    return source


def _torchcodec_decoder(source: AudioSource, sf_error: Exception):
    try:
        from torchcodec.decoders import AudioDecoder
    except (ImportError, RuntimeError) as e:  # RuntimeError: installed but FFmpeg / torch libs fail to load
        raise RuntimeError(
            f"soundfile could not decode this audio ({sf_error}) and the torchcodec fallback is unavailable "
            f"({str(e).splitlines()[0]}). Install torchcodec and FFmpeg for formats beyond WAV/FLAC/OGG/MP3."
        ) from sf_error
    if hasattr(source, "read"):
        source.seek(0)
        source = source.read()
    elif isinstance(source, os.PathLike):
        source = os.fspath(source)
    return AudioDecoder(bytes(source) if isinstance(source, bytearray) else source)


def load_audio(source: AudioSource) -> Tuple[torch.Tensor, int]:
    """Decode audio from a path, bytes or file-like object to ``(waveform, sample_rate)``.

    ``waveform`` is float32 (channels, samples) in [-1, 1], at the file's native rate and channel count.
    """
    import soundfile as sf

    try:
        data, sr = sf.read(_as_file(source), dtype="float32", always_2d=True)
    except sf.LibsndfileError as e:
        samples = _torchcodec_decoder(source, e).get_all_samples()
        return samples.data.float(), int(samples.sample_rate)
    return torch.from_numpy(data.T.copy()), int(sr)


def audio_info(source: AudioSource) -> Tuple[int, int]:
    """Header-only ``(num_frames, sample_rate)``; no sample decoding.

    The soundfile path is exact; the TorchCodec fallback derives frames from the header duration (an estimate
    for some compressed formats).
    """
    import soundfile as sf

    try:
        info = sf.info(_as_file(source))
    except sf.LibsndfileError as e:
        meta = _torchcodec_decoder(source, e).metadata
        duration = meta.duration_seconds_from_header or 0.0
        return int(round(duration * meta.sample_rate)), int(meta.sample_rate)
    return int(info.frames), int(info.samplerate)
