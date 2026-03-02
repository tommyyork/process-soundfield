# Usage: python process-a-format.py [-input CH1,CH2,...] [-output CH1,CH2,...] [-nr 0-10]
#                                   [-ss START] [-to END] [-describe] [-loudnorm]
#                                   [-noisereduce]
#                                   <input.wav> <output.wav>
#
# -input: 1-based comma-separated channel list (e.g. 2,3,4,5); order is preserved.
# -output: 1-based comma-separated order for output channels (permutation of -input); omit to keep order.
# --nr: noise reduction intensity 1-10 (1=minimal, 10=maximal).
# -ss: start time for time slice (seconds or HH:MM:SS; omit to start at beginning).
# -to: end time for time slice (seconds or HH:MM:SS; omit to keep from start to end of file).
# -describe: run YAMnet on the processed audio and print an event table.
# -loudnorm: apply EBU R128 loudness normalization (target -23 LUFS) instead of 0 dB peak normalization.
# -noisereduce: use Noisereduce spectral-gating noise reduction instead of PCA.

import sys
import wave
import os
import math

import numpy as np

from normalization import (
    TARGET_LUFS_EBU,
    normalize_channels_to_0db,
    normalize_channels_loudnorm_ebu128,
)
from noise_reduction import pca_noise_reduction, noisereduce_spectral_gate
from wav_utils import (
    validate_wav_channels,
    select_channels,
    frames_to_float_matrix,
    float_matrix_to_frames,
)
from yamnet import describe_audio_with_yamnet


args = sys.argv[1:]
input_channels_arg: str | None = None
output_channels_arg: str | None = None
nr_arg: str | None = None
ss_arg: str | None = None
to_arg: str | None = None
describe_flag = False
loudnorm_flag = False
use_pca = False

while "-input" in args:
    i = args.index("-input")
    if i + 1 >= len(args):
        print("Error: -input requires a value (e.g. 2,3,4,5)")
        sys.exit(1)
    input_channels_arg = args[i + 1]
    args = args[:i] + args[i + 2 :]
while "-output" in args:
    i = args.index("-output")
    if i + 1 >= len(args):
        print("Error: -output requires a value (e.g. 5,4,3,2)")
        sys.exit(1)
    output_channels_arg = args[i + 1]
    args = args[:i] + args[i + 2 :]
while "-nr" in args:
    i = args.index("-nr")
    if i + 1 >= len(args):
        print("Error: -nr requires a value (0-10)")
        sys.exit(1)
    nr_arg = args[i + 1]
    args = args[:i] + args[i + 2 :]
while "-ss" in args:
    i = args.index("-ss")
    if i + 1 >= len(args):
        print("Error: -ss requires a value (start time in seconds)")
        sys.exit(1)
    ss_arg = args[i + 1]
    args = args[:i] + args[i + 2 :]
while "-to" in args:
    i = args.index("-to")
    if i + 1 >= len(args):
        print("Error: -to requires a value (end time in seconds)")
        sys.exit(1)
    to_arg = args[i + 1]
    args = args[:i] + args[i + 2 :]
if "-describe" in args:
    describe_flag = True
    args = [a for a in args if a != "-describe"]
if "-loudnorm" in args:
    loudnorm_flag = True
    args = [a for a in args if a != "-loudnorm"]
if "-pca" in args:
    use_pca = True
    args = [a for a in args if a != "-pca"]
if len(args) != 2:
    print(
        "Usage: python process-a-format.py [-input CH1,CH2,...] [-output CH1,CH2,...] "
        "[-nr 0-10] [-ss START] [-to END] [-describe] [-loudnorm] [-pca] "
        "<input.wav> <output.wav>"
    )
    sys.exit(1)

input_path = args[0]
output_path = args[1]

# Validate input file extension
if not input_path.lower().endswith(".wav"):
    print("Error: Input file must be a .wav file.")
    sys.exit(1)

# Validate input file exists
if not os.path.isfile(input_path):
    print(f"Error: File '{input_path}' does not exist.")
    sys.exit(1)


def parse_time_to_seconds(s: str) -> float:
    """
    Parse a time value to seconds. Accepts:
    - Plain seconds: "65", "1.5"
    - Timestamp (ffmpeg-style): "HH:MM:SS" or "HH:MM:SS.xxx", e.g. "00:01:05" -> 65, "01:00:01" -> 3601
    """
    s = s.strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError(f"Timestamp must be HH:MM:SS or HH:MM:SS.xxx, got {s!r}")
        try:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
        except ValueError as e:
            raise ValueError(f"Invalid timestamp {s!r}: {e}") from e
        if hours < 0 or minutes < 0 or seconds < 0 or minutes >= 60:
            raise ValueError(f"Invalid timestamp {s!r}: hours/minutes/seconds out of range")
        return hours * 3600 + minutes * 60 + seconds
    return float(s)


# Validate WAV has 4+ channels before processing
n_channels = validate_wav_channels(input_path, min_channels=4)

# Optional channel selection: -input CH1,CH2,... (1-based, order preserved); -output CH1,CH2,... (reorder for output)
channel_indices: list[int] | None = None  # None = use all channels in file order
output_reorder: list[int] | None = None   # None = keep order; else list of indices into selected channels for output order
if input_channels_arg is not None:
    try:
        input_list = [int(x.strip()) for x in input_channels_arg.split(",") if x.strip()]
        if len(input_list) < 4:
            print(f"Error: -input must specify at least 4 channels (got {len(input_list)})")
            sys.exit(1)
        if any(c < 1 or c > n_channels for c in input_list):
            print(f"Error: -input channel numbers must be 1–{n_channels} for a {n_channels}-channel file")
            sys.exit(1)
        channel_indices = [c - 1 for c in input_list]  # 0-based, order preserved
        print(f"Using -input channels (1-based): {input_channels_arg} → {len(channel_indices)} channel(s).")

        if output_channels_arg is not None:
            output_list = [int(x.strip()) for x in output_channels_arg.split(",") if x.strip()]
            if set(output_list) != set(input_list):
                print("Error: -output must list the same channel numbers as -input (permutation only)")
                sys.exit(1)
            if len(output_list) != len(input_list):
                print("Error: -output must have the same number of channels as -input")
                sys.exit(1)
            # output_reorder[i] = index into channel_indices for the channel that goes to output position i
            output_reorder = [channel_indices.index(c - 1) for c in output_list]
            print(f"Using -output order (1-based): {output_channels_arg}.")
    except ValueError as e:
        print(f"Error: -input and -output require comma-separated integers: {e}")
        sys.exit(1)
if output_channels_arg is not None and input_channels_arg is None:
    print("Error: -output can only be used with -input")
    sys.exit(1)

_PROGRESS_WIDTH = 40


def _progress(step_name: str, fraction: float) -> None:
    """Print a single progress line (overwrites with \r). fraction in [0, 1]."""
    filled = min(_PROGRESS_WIDTH, int(_PROGRESS_WIDTH * fraction))
    bar = "=" * filled + (">" if filled < _PROGRESS_WIDTH else "") + " " * (_PROGRESS_WIDTH - filled - 1)
    pct = min(100, int(100 * fraction))
    print(f"\r  {step_name}: [{bar}] {pct}%", end="", flush=True)


def _progress_done(step_name: str) -> None:
    """Mark a progress step as 100% complete and newline."""
    print(f"\r  {step_name}: [{'=' * _PROGRESS_WIDTH}] 100%")


try:
    print("Loading input file...")
    print(f"  Input: {input_path}")
    with wave.open(input_path, "rb") as wav_in:
        n_channels = wav_in.getnchannels()
        sampwidth = wav_in.getsampwidth()
        framerate = wav_in.getframerate()
        n_frames = wav_in.getnframes()
        comptype = wav_in.getcomptype()

        if comptype != "NONE" and comptype != "none":
            print(f"Error: Compressed WAV (comptype={comptype}) is not supported.")
            sys.exit(1)

        frame_bytes = wav_in.readframes(n_frames)

    # Time slice: -ss start time, -to end time (seconds or HH:MM:SS). Applied before any other processing.
    if ss_arg is not None or to_arg is not None:
        try:
            start_sec = parse_time_to_seconds(ss_arg) if ss_arg is not None else 0.0
            end_sec = parse_time_to_seconds(to_arg) if to_arg is not None else None
            if start_sec < 0:
                print("Error: -ss must be non-negative")
                sys.exit(1)
            if end_sec is not None and end_sec <= start_sec:
                print("Error: -to must be greater than -ss")
                sys.exit(1)
            start_frame = int(start_sec * framerate)
            end_frame = int(end_sec * framerate) if end_sec is not None else n_frames
            start_frame = max(0, min(start_frame, n_frames))
            end_frame = max(start_frame, min(end_frame, n_frames))
            frame_size = n_channels * sampwidth
            frame_bytes = frame_bytes[start_frame * frame_size : end_frame * frame_size]
            n_frames = end_frame - start_frame
            if n_frames == 0:
                print("Error: Time slice results in zero frames (check -ss and -to vs file duration).")
                sys.exit(1)
            if ss_arg is not None and to_arg is not None:
                print(f"  Time slice: {start_sec:.2f}s – {end_sec:.2f}s ({n_frames} frames).")
            elif ss_arg is not None:
                print(f"  Time slice: from {start_sec:.2f}s to end ({n_frames} frames).")
            else:
                print(f"  Time slice: from start to {end_sec:.2f}s ({n_frames} frames).")
        except ValueError as e:
            print(f"Error: -ss and -to require seconds or HH:MM:SS timestamp: {e}")
            sys.exit(1)

    if channel_indices is not None:
        frame_bytes = select_channels(frame_bytes, n_channels, sampwidth, channel_indices)
        n_channels = len(channel_indices)

    print(f"  Loaded {n_frames} frames, {n_channels} channels, {sampwidth * 8}-bit PCM.\n")

    _progress("Normalizing channels", 0.0)

    def norm_progress(frac: float) -> None:
        _progress("Normalizing channels", frac)

    if loudnorm_flag:
        normalized, norm_gains_db = normalize_channels_loudnorm_ebu128(
            frame_bytes, n_channels, sampwidth, framerate, progress_callback=norm_progress
        )
    else:
        normalized, norm_gains_db = normalize_channels_to_0db(
            frame_bytes, n_channels, sampwidth, progress_callback=norm_progress
        )
    _progress_done("Normalizing channels")

    # --- Normalization summary ---
    if loudnorm_flag:
        print(f"--- Normalization (EBU R128 loudness, target {TARGET_LUFS_EBU:.1f} LUFS) ---")
    else:
        print("--- Normalization (per-channel peak → 0 dB FS) ---")
    for ch in range(n_channels):
        print(f"  Channel {ch}: gain {norm_gains_db[ch]:+.2f} dB")
    avg_gain_db = sum(norm_gains_db) / n_channels if n_channels else 0
    print(f"  Average gain across channels: {avg_gain_db:+.2f} dB")

    # Start with all channels; apply six-channel rule first, then drop by gain
    keep_mask = list(range(n_channels))

    # If exactly 6 channels after normalization, drop the first two (channels 0 and 1)
    if n_channels == 6:
        keep_mask = [ch for ch in keep_mask if ch >= 2]
        print("\n  Six channels after normalization: dropping first two channels (0 and 1).")
        print(f"  Output will have {len(keep_mask)} channel(s).\n")

    # Drop channels with gain > 96 dB (likely silence or noise)
    MAX_GAIN_DB = 96.0
    dropped = [ch for ch in keep_mask if norm_gains_db[ch] > MAX_GAIN_DB]
    keep_mask = [ch for ch in keep_mask if norm_gains_db[ch] <= MAX_GAIN_DB]
    if dropped:
        print(f"  Dropping {len(dropped)} channel(s) with gain > {MAX_GAIN_DB} dB: {dropped}")
        for ch in dropped:
            print(f"    Channel {ch}: gain {norm_gains_db[ch]:+.2f} dB (exceeds threshold)")
    n_channels_kept = len(keep_mask)
    if n_channels_kept == 0:
        print("Error: All channels were dropped (gain > 96 dB on every channel). Nothing to output.")
        sys.exit(1)
    if n_channels_kept != n_channels:
        print(f"  Output will have {n_channels_kept} channel(s).\n")

    data = frames_to_float_matrix(normalized, n_channels, sampwidth)
    data = data[:, keep_mask]

    # Noise reduction: choose PCA or Noisereduce, and derive intensity/components from -nr
    nr_intensity: int | None = None
    if nr_arg is not None:
        try:
            nr_val = int(nr_arg)
            if nr_val < 0 or nr_val > 10:
                print("Error: -nr must be between 0 and 10")
                sys.exit(1)
            nr_intensity = nr_val
        except ValueError:
            print("Error: -nr value must be an integer 0-10")
            sys.exit(1)

    skip_nr = nr_intensity == 0 if nr_intensity is not None else False

    if skip_nr:
        print(
            "Noise reduction: -nr 0 specified; noise reduction (Noisereduce or PCA) "
            "would normally run here but has been skipped."
        )
    elif not use_pca:
        print(
            f"Applying Noisereduce spectral-gating noise reduction"
            f"{f' (intensity {nr_intensity}/10)' if nr_intensity is not None else ''}..."
        )
        _progress("Noisereduce", 0.0)

        def nr_progress(frac: float) -> None:
            _progress("Noisereduce", frac)

        try:
            data, nr_stats = noisereduce_spectral_gate(
                data,
                sample_rate=framerate,
                intensity=nr_intensity,
                stationary=False,
                progress_callback=nr_progress,
            )
        except (RuntimeError, ValueError) as e:
            print(f"Error during Noisereduce processing: {e}")
            sys.exit(1)
        _progress_done("Noisereduce")

        # --- Noisereduce summary ---
        print("\n--- Noisereduce spectral-gating noise reduction ---")
        print(f"  Variant: {'stationary' if nr_stats['stationary'] else 'non-stationary'}")
        if nr_stats["intensity"] is not None:
            print(
                f"  Intensity: {nr_stats['intensity']}/10 "
                f"(prop_decrease={nr_stats['prop_decrease']:.3f})"
            )
        else:
            print(f"  Intensity: default (prop_decrease={nr_stats['prop_decrease']:.3f})")
        print("  RMS change per channel (dB) after Noisereduce:")
        for ch in range(n_channels_kept):
            print(
                f"    Channel {ch}: {nr_stats['rms_change_db_per_channel'][ch]:+.2f} dB"
            )
    else:
        # PCA noise reduction
        if nr_intensity is not None:
            # Map 1–10 to number of components kept (1=max reduction, n-1=minimal)
            # nr=1 -> keep n-1 (minimal); nr=10 -> keep 1 (max).
            n = n_channels_kept
            n_components_pca = max(
                1,
                min(
                    n,
                    int(round(1 + max(0, n - 2) * (10 - nr_intensity) / 9)),
                ),
            )
            print(f"Applying PCA noise reduction (intensity {nr_intensity}/10)...")
        else:
            n_components_pca = max(1, n_channels_kept - 1)
            print("Applying PCA noise reduction...")

        print(f"  Input: {data.shape[0]} frames, {n_channels_kept} channels.")
        _progress("PCA noise reduction", 0.0)

        def pca_progress(frac: float) -> None:
            _progress("PCA noise reduction", frac)

        data, pca_stats = pca_noise_reduction(
            data, n_components=n_components_pca, progress_callback=pca_progress
        )
        _progress_done("PCA noise reduction")

        # --- PCA noise reduction summary ---
        print("--- PCA noise reduction ---")
        print(
            f"  Components kept: {pca_stats['n_components_kept']} / "
            f"{pca_stats['n_components_kept'] + pca_stats['n_components_dropped']}"
        )
        print(f"  Variance retained (signal): {pca_stats['variance_retained'] * 100:.2f}%")
        print(f"  Variance dropped (noise):  {pca_stats['variance_dropped'] * 100:.2f}%")
        print("  Explained variance per component (all):")
        for i, r in enumerate(pca_stats["explained_variance_ratio"]):
            print(f"    PC{i + 1}: {r * 100:.2f}%")
        print("  RMS change per channel (dB) after PCA:")
        for ch in range(n_channels_kept):
            print(f"    Channel {ch}: {pca_stats['rms_change_db_per_channel'][ch]:+.2f} dB")

    if output_reorder is not None:
        data = data[:, output_reorder]

    denoised_bytes = float_matrix_to_frames(data, sampwidth)

    if describe_flag:
        # YAMnet analysis on processed (denoised) audio, just before output
        print("\n--- YAMnet audio description (-describe) ---")
        mono = data.mean(axis=1).astype(np.float64)
        duration_sec = mono.shape[0] / framerate
        if framerate != 16000:
            target_len = int(round(duration_sec * 16000))
            mono = np.interp(
                np.linspace(0, len(mono) - 1, target_len),
                np.arange(len(mono)),
                mono,
            ).astype(np.float32)
            duration_sec = target_len / 16000.0
        else:
            mono = mono.astype(np.float32)
        _progress("YAMnet analysis", 0.0)

        def yamnet_progress(frac: float) -> None:
            _progress("YAMnet analysis", frac)

        events = describe_audio_with_yamnet(
            mono, duration_sec, max_events_per_minute=5, progress_callback=yamnet_progress
        )
        _progress_done("YAMnet analysis")
        if not events:
            print("  No events (audio too short).")
        else:
            print(f"  {'Start (s)':<12} {'End (s)':<12} {'Label':<30} {'Confidence':<10}")
            print("  " + "-" * 66)
            for start_s, end_s, label, conf in events:
                print(f"  {start_s:<12.2f} {end_s:<12.2f} {label:<30} {conf:<10.4f}")
        print()

    print("\nWriting output file...")
    print(f"  Output: {output_path}")
    print(f"  Channels: {n_channels_kept}, Sample width: {sampwidth} bytes, Frame rate: {framerate} Hz")
    with wave.open(output_path, "wb") as wav_out:
        wav_out.setnchannels(n_channels_kept)
        wav_out.setsampwidth(sampwidth)
        wav_out.setframerate(framerate)
        wav_out.setcomptype(comptype, "N/A")
        wav_out.writeframes(denoised_bytes)

    print(f"\nWrote: {output_path}")

except wave.Error as e:
    print(f"Error opening WAV file: {e}")
    sys.exit(1)
except ValueError as e:
    print(f"Error: {e}")
    sys.exit(1)

    