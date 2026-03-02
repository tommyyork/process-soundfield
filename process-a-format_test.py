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

from wav_utils import select_channels, frames_to_float_matrix, _frames_to_float_matrix
from ambisonics import (
    assert_gain_relationships_preserved,
    assert_phase_relationships_preserved,
)


def test_process_a_format_with_loudnorm() -> None:
    """
    Invoke `process-a-format.py` with the flags:

        test-ambisonic.wav test-ambisonic-processed.wav
        -input 3,4,5,6 -nr 0 -ss 00:00:02 -to 00:00:12 -loudnorm

    and check that:
    - the output has 4 channels,
    - the per-channel volumes are within 24 dB of each other,
    - the output duration is 10 seconds,
    - the relative gain and phase relationships between channels are preserved.
    """
    script_dir = Path(__file__).resolve().parent
    script_path = script_dir / "process-a-format.py"

    input_path = script_dir / "testdata" / "test-ambisonic.wav"
    output_path = script_dir / "testdata" / "test-ambisonic-processed.wav"

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
        "0",
        "-ss",
        "00:00:02",
        "-to",
        "00:00:12",
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
    expected_duration = 10.0  # 10 seconds
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

    # --- Additional check: gain and phase relationships between channels ---
    with wave.open(str(input_path), "rb") as wav_in:
        in_n_channels = wav_in.getnchannels()
        in_sampwidth = wav_in.getsampwidth()
        in_framerate = wav_in.getframerate()
        in_n_frames = wav_in.getnframes()
        in_frames = wav_in.readframes(in_n_frames)

    start_sec = 2.0
    end_sec = 12.0
    start_frame = int(start_sec * in_framerate)
    end_frame = int(end_sec * in_framerate)
    frame_size = in_n_channels * in_sampwidth
    in_frames_slice = in_frames[start_frame * frame_size : end_frame * frame_size]

    # Select channels 3,4,5,6 (1-based) -> indices [2,3,4,5]
    channel_indices = [2, 3, 4, 5]
    in_selected_bytes = select_channels(
        in_frames_slice, in_n_channels, in_sampwidth, channel_indices
    )
    in_data = frames_to_float_matrix(
        in_selected_bytes, len(channel_indices), in_sampwidth
    )

    assert_gain_relationships_preserved(in_data, data)
    assert_phase_relationships_preserved(in_data, data)


def test_process_a_format_default_loudness() -> None:
    """
    Invoke `process-a-format.py` with the flags:

        test-ambisonic.wav test-ambisonic-processed-default.wav
        -input 3,4,5,6 -nr 0 -ss 00:00:02 -to 00:00:12

    and check that:
    - the output has 4 channels,
    - the per-channel volumes are within 24 dB of each other,
    - the output duration is 10 seconds,
    - the relative gain and phase relationships between channels are preserved.
    """
    script_dir = Path(__file__).resolve().parent
    script_path = script_dir / "process-a-format.py"

    input_path = script_dir / "testdata" / "test-ambisonic.wav"
    output_path = script_dir / "testdata" / "test-ambisonic-processed-default.wav"

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
        "0",
        "-ss",
        "00:00:02",
        "-to",
        "00:00:12",
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
    expected_duration = 10.0  # 10 seconds
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

    # --- Additional check: gain and phase relationships between channels ---
    with wave.open(str(input_path), "rb") as wav_in:
        in_n_channels = wav_in.getnchannels()
        in_sampwidth = wav_in.getsampwidth()
        in_framerate = wav_in.getframerate()
        in_n_frames = wav_in.getnframes()
        in_frames = wav_in.readframes(in_n_frames)

    start_sec = 2.0
    end_sec = 12.0
    start_frame = int(start_sec * in_framerate)
    end_frame = int(end_sec * in_framerate)
    frame_size = in_n_channels * in_sampwidth
    in_frames_slice = in_frames[start_frame * frame_size : end_frame * frame_size]

    # Select channels 3,4,5,6 (1-based) -> indices [2,3,4,5]
    channel_indices = [2, 3, 4, 5]
    in_selected_bytes = select_channels(
        in_frames_slice, in_n_channels, in_sampwidth, channel_indices
    )
    in_data = frames_to_float_matrix(
        in_selected_bytes, len(channel_indices), in_sampwidth
    )

    assert_gain_relationships_preserved(in_data, data)
    assert_phase_relationships_preserved(in_data, data)


def test_process_a_format_noisereduce_nr5() -> None:
    """
    Invoke `process-a-format.py` with the flags:

        test-ambisonic.wav test-ambisonic-processed-pca-nr5.wav
        -input 3,4,5,6 -nr 5 -ss 00:00:02 -to 00:00:12

    and check that:
    - the output has 4 channels,
    - the per-channel volumes are within 24 dB of each other,
    - the output duration is 10 seconds,
    - the relative gain and phase relationships between channels are broadly preserved
      despite noisereduce noise reduction at intensity 5.
    """
    script_dir = Path(__file__).resolve().parent
    script_path = script_dir / "process-a-format.py"

    input_path = script_dir / "testdata" / "test-ambisonic.wav"
    output_path = script_dir / "testdata" / "test-ambisonic-processed-pca-nr5.wav"

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
        "00:00:02",
        "-to",
        "00:00:12",
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
    expected_duration = 10.0  # 10 seconds
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

    # --- Additional check: gain and phase relationships between channels ---
    with wave.open(str(input_path), "rb") as wav_in:
        in_n_channels = wav_in.getnchannels()
        in_sampwidth = wav_in.getsampwidth()
        in_framerate = wav_in.getframerate()
        in_n_frames = wav_in.getnframes()
        in_frames = wav_in.readframes(in_n_frames)

    start_sec = 2.0
    end_sec = 12.0
    start_frame = int(start_sec * in_framerate)
    end_frame = int(end_sec * in_framerate)
    frame_size = in_n_channels * in_sampwidth
    in_frames_slice = in_frames[start_frame * frame_size : end_frame * frame_size]

    # Select channels 3,4,5,6 (1-based) -> indices [2,3,4,5]
    channel_indices = [2, 3, 4, 5]
    in_selected_bytes = select_channels(
        in_frames_slice, in_n_channels, in_sampwidth, channel_indices
    )
    in_data = frames_to_float_matrix(
        in_selected_bytes, len(channel_indices), in_sampwidth
    )

    assert_gain_relationships_preserved(in_data, data)
    assert_phase_relationships_preserved(in_data, data)

