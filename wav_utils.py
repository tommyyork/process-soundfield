import sys
import wave
import array
from typing import List

import numpy as np


def validate_wav_channels(path: str, min_channels: int = 4) -> int:
    """Validate that the WAV file has at least min_channels. Returns channel count."""
    try:
        with wave.open(path, "rb") as wav:
            n_channels = wav.getnchannels()
            if n_channels < min_channels:
                print(
                    f"Error: WAV file has {n_channels} channel(s). "
                    f"At least {min_channels} channels are required."
                )
                sys.exit(1)
            return n_channels
    except wave.Error as e:
        print(f"Error opening WAV file: {e}")
        sys.exit(1)


def select_channels(
    frame_bytes: bytes,
    n_channels: int,
    sampwidth: int,
    channel_indices: List[int],
) -> bytes:
    """Extract selected channels from interleaved frame bytes. Returns new interleaved bytes."""
    n_frames = len(frame_bytes) // (n_channels * sampwidth)
    bytes_per_frame = n_channels * sampwidth
    out = []
    for f in range(n_frames):
        frame_start = f * bytes_per_frame
        for ch in channel_indices:
            sample_start = frame_start + ch * sampwidth
            out.append(frame_bytes[sample_start : sample_start + sampwidth])
    return b"".join(out)


def _unpack_24bit_pcm(frame_bytes: bytes, n_channels: int) -> np.ndarray:
    """Convert 24-bit LE interleaved bytes to int32 array (n_frames, n_channels)."""
    n_frames = len(frame_bytes) // (n_channels * 3)
    raw = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(n_frames, n_channels * 3)
    samples = (
        raw[:, 0::3].astype(np.int32)
        | (raw[:, 1::3].astype(np.int32) << 8)
        | (raw[:, 2::3].astype(np.int32) << 16)
    )
    samples = np.where(samples >= 0x800000, samples - 0x1000000, samples)
    return samples


def _pack_24bit_pcm(samples: np.ndarray) -> bytes:
    """Convert int32 array (n_frames, n_channels) to 24-bit LE interleaved bytes."""
    samples = np.clip(samples.astype(np.int32), -8388608, 8388607)
    samples = samples & 0xFFFFFF
    n_frames, n_channels = samples.shape
    out = np.empty((n_frames, n_channels * 3), dtype=np.uint8)
    out[:, 0::3] = samples & 0xFF
    out[:, 1::3] = (samples >> 8) & 0xFF
    out[:, 2::3] = (samples >> 16) & 0xFF
    return out.tobytes()


def frames_to_float_matrix(
    frame_bytes: bytes,
    n_channels: int,
    sampwidth: int,
) -> np.ndarray:
    """Convert interleaved WAV frame bytes to float matrix (n_frames, n_channels) in [-1, 1]."""
    if sampwidth == 2:
        dtype = np.int16
        full_scale = 32767.0
        n_frames = len(frame_bytes) // (n_channels * 2)
        samples = np.frombuffer(frame_bytes, dtype=dtype)
    elif sampwidth == 3:
        samples = _unpack_24bit_pcm(frame_bytes, n_channels)
        full_scale = 8388607.0
        n_frames = samples.shape[0]
        return samples.astype(np.float64) / full_scale
    elif sampwidth == 4:
        dtype = np.int32
        full_scale = 2147483647.0
        n_frames = len(frame_bytes) // (n_channels * 4)
        samples = np.frombuffer(frame_bytes, dtype=dtype)
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")
    data = samples.reshape(n_frames, n_channels)
    return data.astype(np.float64) / full_scale


def float_matrix_to_frames(
    data: np.ndarray,
    sampwidth: int,
) -> bytes:
    """Convert float matrix (n_frames, n_channels) in [-1, 1] to interleaved WAV frame bytes."""
    data = np.clip(data, -1.0, 1.0)
    if sampwidth == 2:
        full_scale = 32767.0
        out = (data * full_scale).astype(np.int16)
        return out.tobytes()
    elif sampwidth == 3:
        full_scale = 8388607.0
        samples = (data * full_scale).astype(np.int32)
        return _pack_24bit_pcm(samples)
    elif sampwidth == 4:
        full_scale = 2147483647.0
        out = (data * full_scale).astype(np.int32)
        return out.tobytes()
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")


def _frames_to_float_matrix(
    frame_bytes: bytes,
    n_channels: int,
    sampwidth: int,
) -> np.ndarray:
    """
    Alias used by tests; delegates to frames_to_float_matrix.
    """
    return frames_to_float_matrix(frame_bytes, n_channels, sampwidth)

