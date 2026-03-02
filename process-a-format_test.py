"""
Integration tests for `process-a-format.py`.

These tests are intended to exercise the script
`utils/process-soundfield/process-a-format.py`.
"""

from __future__ import annotations

import math
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest


def _unpack_24bit_pcm(frame_bytes: bytes, n_channels: int) -> np.ndarray:
    """Convert 24-bit little-endian interleaved bytes to int32 array (n_frames, n_channels)."""
    n_frames = len(frame_bytes) // (n_channels * 3)
    raw = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(n_frames, n_channels * 3)
    samples = (
        raw[:, 0::3].astype(np.int32)
        | (raw[:, 1::3].astype(np.int32) << 8)
        | (raw[:, 2::3].astype(np.int32) << 16)
    )
    samples = np.where(samples >= 0x800000, samples - 0x1000000, samples)
    return samples


def _frames_to_float_matrix(
    frame_bytes: bytes,
    n_channels: int,
    sampwidth: int,
) -> np.ndarray:
    """
    Convert interleaved WAV frame bytes to float matrix (n_frames, n_channels) in [-1, 1].

    This mirrors the conversion performed inside `process-a-format.py` and is only
    used here for test-time inspection of the output file.
    """
    if sampwidth == 2:
        dtype = np.int16
        full_scale = 32767.0
        n_frames = len(frame_bytes) // (n_channels * 2)
        samples = np.frombuffer(frame_bytes, dtype=dtype)
    elif sampwidth == 3:
        samples = _unpack_24bit_pcm(frame_bytes, n_channels)
        full_scale = 8388607.0
        n_frames = samples.shape[0]
        return samples.astype(np.float64).reshape(n_frames, n_channels) / full_scale
    elif sampwidth == 4:
        dtype = np.int32
        full_scale = 2147483647.0
        n_frames = len(frame_bytes) // (n_channels * 4)
        samples = np.frombuffer(frame_bytes, dtype=dtype)
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")
    data = samples.reshape(n_frames, n_channels)
    return data.astype(np.float64) / full_scale


def test_process_a_format_mixpre_integration() -> None:
    """
    Invoke `process-a-format.py` with the flags:

        MixPre-004A.WAV MixPre-004A-processed.wav
        -input 3,4,5,6 --nr 5 -ss 00:05:00 -to 00:10:00 -describe -loudnorm

    and check that:
    - the output has 4 channels,
    - the per-channel volumes are within 24 dB of each other,
    - the output duration is 5 minutes.
    """
    script_dir = Path(__file__).resolve().parent
    script_path = script_dir / "process-a-format.py"

    input_path = script_dir / "MixPre-004A.WAV"
    output_path = script_dir / "MixPre-004A-processed.wav"

    if not input_path.exists():
        pytest.skip(f"Test input file not found: {input_path}")

    if not script_path.exists():
        pytest.skip(f"Script under test not found: {script_path}")

    if output_path.exists():
        output_path.unlink()

    cmd = [
        sys.executable,
        str(script_path),
        str(input_path),
        str(output_path),
        "-input",
        "3,4,5,6",
        "-nr",
        "5",
        "-ss",
        "00:05:00",
        "-to",
        "00:10:00",
        "-describe",
        "-loudnorm",
    ]

    subprocess.run(cmd, check=True, cwd=str(script_dir))

    assert output_path.exists(), "Expected output WAV file was not created."

    with wave.open(str(output_path), "rb") as wav_out:
        n_channels = wav_out.getnchannels()
        sampwidth = wav_out.getsampwidth()
        framerate = wav_out.getframerate()
        n_frames = wav_out.getnframes()
        frames = wav_out.readframes(n_frames)

    assert n_channels == 4, f"Expected 4 channels, got {n_channels}"

    duration_sec = n_frames / float(framerate)
    expected_duration = 5 * 60.0  # 5 minutes
    # Allow small tolerance for rounding when converting times to frames.
    assert math.isclose(
        duration_sec, expected_duration, rel_tol=0.0, abs_tol=0.5
    ), f"Expected duration ~{expected_duration}s, got {duration_sec:.3f}s"

    data = _frames_to_float_matrix(frames, n_channels, sampwidth)
    rms_per_channel = np.sqrt(np.mean(np.square(data), axis=0))
    # Avoid log of zero by clamping to a tiny positive value.
    rms_per_channel = np.maximum(rms_per_channel, 1e-12)
    rms_db = 20.0 * np.log10(rms_per_channel)

    max_db = float(np.max(rms_db))
    min_db = float(np.min(rms_db))
    diff_db = max_db - min_db

    assert (
        diff_db <= 24.0
    ), f"Channel RMS levels differ by {diff_db:.2f} dB, which exceeds 24 dB"


def test_process_a_format_mixpre_integration_no_loudnorm() -> None:
    """
    Invoke `process-a-format.py` with the flags:

        MixPre-004A.WAV MixPre-004A-processed-noloudnorm.wav
        -input 3,4,5,6 --nr 5 -ss 00:05:00 -to 00:10:00 -describe

    (i.e. same as the previous test, but omitting `-loudnorm`),
    and check that:
    - the output has 4 channels,
    - the per-channel volumes are within 24 dB of each other,
    - the output duration is 5 minutes.
    """
    script_dir = Path(__file__).resolve().parent
    script_path = script_dir / "process-a-format.py"

    input_path = script_dir / "MixPre-004A.WAV"
    output_path = script_dir / "MixPre-004A-processed-noloudnorm.wav"

    if not input_path.exists():
        pytest.skip(f"Test input file not found: {input_path}")

    if not script_path.exists():
        pytest.skip(f"Script under test not found: {script_path}")

    if output_path.exists():
        output_path.unlink()

    cmd = [
        sys.executable,
        str(script_path),
        str(input_path),
        str(output_path),
        "-input",
        "3,4,5,6",
        "-nr",
        "5",
        "-ss",
        "00:05:00",
        "-to",
        "00:10:00",
        "-describe",
    ]

    subprocess.run(cmd, check=True, cwd=str(script_dir))

    assert output_path.exists(), "Expected output WAV file was not created."

    with wave.open(str(output_path), "rb") as wav_out:
        n_channels = wav_out.getnchannels()
        sampwidth = wav_out.getsampwidth()
        framerate = wav_out.getframerate()
        n_frames = wav_out.getnframes()
        frames = wav_out.readframes(n_frames)

    assert n_channels == 4, f"Expected 4 channels, got {n_channels}"

    duration_sec = n_frames / float(framerate)
    expected_duration = 5 * 60.0  # 5 minutes
    assert math.isclose(
        duration_sec, expected_duration, rel_tol=0.0, abs_tol=0.5
    ), f"Expected duration ~{expected_duration}s, got {duration_sec:.3f}s"

    data = _frames_to_float_matrix(frames, n_channels, sampwidth)
    rms_per_channel = np.sqrt(np.mean(np.square(data), axis=0))
    rms_per_channel = np.maximum(rms_per_channel, 1e-12)
    rms_db = 20.0 * np.log10(rms_per_channel)

    max_db = float(np.max(rms_db))
    min_db = float(np.min(rms_db))
    diff_db = max_db - min_db

    assert (
        diff_db <= 24.0
    ), f"Channel RMS levels differ by {diff_db:.2f} dB, which exceeds 24 dB"
