import math
from typing import List, Tuple

import numpy as np

from wav_utils import frames_to_float_matrix, float_matrix_to_frames, _unpack_24bit_pcm, _pack_24bit_pcm


TARGET_LUFS_EBU: float = -23.0  # Target loudness for -loudnorm (EBU R128, -23 LUFS)


def normalize_channels_to_0db(
    frame_bytes: bytes,
    n_channels: int,
    sampwidth: int,
    progress_callback=None,
) -> Tuple[bytes, List[float]]:
    """
    Normalize all channels together by the same gain so the peak across any channel
    reaches 0 dB FS (full scale). Returns (new frame bytes, list of gain in dB per channel,
    same value for each). Supports 16-, 24-, and 32-bit PCM.
    """
    import array

    if sampwidth == 2:  # 16-bit
        full_scale = 32767
        n_frames = len(frame_bytes) // (n_channels * 2)
        samples = array.array("h")
        samples.frombytes(frame_bytes)
        use_24bit = False
    elif sampwidth == 3:  # 24-bit
        full_scale = 8388607  # 2^23 - 1
        data = _unpack_24bit_pcm(frame_bytes, n_channels)
        n_frames = data.shape[0]
        use_24bit = True
    elif sampwidth == 4:  # 32-bit int
        full_scale = 2147483647
        n_frames = len(frame_bytes) // (n_channels * 4)
        samples = array.array("i")
        samples.frombytes(frame_bytes)
        use_24bit = False
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} (use 16-, 24-, or 32-bit PCM)")

    if use_24bit:
        # Single gain from global peak across all channels
        peak = int(np.max(np.abs(data)))
        if progress_callback:
            progress_callback(0.5)
        if peak > 0:
            scale = full_scale / peak
            gain_db = 20.0 * math.log10(scale)
            data = np.clip(
                np.round(data * scale).astype(np.int32),
                -full_scale,
                full_scale,
            )
            gains_db = [gain_db] * n_channels
        else:
            gains_db = [0.0] * n_channels
        if progress_callback:
            progress_callback(1.0)
        return _pack_24bit_pcm(data), gains_db

    # 16- and 32-bit path
    # Deinterleave: list of n_channels arrays
    channels = [array.array(samples.typecode) for _ in range(n_channels)]
    for i, s in enumerate(samples):
        channels[i % n_channels].append(s)

    # Single gain from global peak across all channels
    peak = 0
    for ch in channels:
        if ch:
            peak = max(peak, max(abs(s) for s in ch))
    if progress_callback:
        progress_callback(0.5)
    if peak > 0:
        scale = full_scale / peak
        gain_db = 20.0 * math.log10(scale)
        gains_db = [gain_db] * n_channels
        for ch in channels:
            for i in range(len(ch)):
                ch[i] = int(round(ch[i] * scale))
                ch[i] = max(-full_scale, min(full_scale, ch[i]))
    else:
        gains_db = [0.0] * n_channels
    if progress_callback:
        progress_callback(1.0)

    # Interleave back
    out = array.array(samples.typecode)
    for i in range(n_frames):
        for ch in channels:
            out.append(ch[i])

    return out.tobytes(), gains_db


def normalize_channels_loudnorm_ebu128(
    frame_bytes: bytes,
    n_channels: int,
    sampwidth: int,
    framerate: int,
    progress_callback=None,
) -> Tuple[bytes, List[float]]:
    """
    Normalize all channels together to a target loudness (EBU R128, -23 LUFS by default).
    Returns (new frame bytes, list of gain in dB per channel, same value for each).
    """
    try:
        import pyloudnorm as pyln
    except ImportError as e:
        print("Error: -loudnorm requires pyloudnorm. Install with: pip install pyloudnorm")
        raise SystemExit(1) from e

    # Convert to float matrix [-1, 1]
    if progress_callback:
        progress_callback(0.1)
    data = frames_to_float_matrix(frame_bytes, n_channels, sampwidth)

    # Integrated loudness across all channels
    meter = pyln.Meter(framerate)  # EBU R128 meter
    if progress_callback:
        progress_callback(0.3)
    loudness = meter.integrated_loudness(data)

    target = TARGET_LUFS_EBU
    if not np.isfinite(loudness):
        # Very quiet material can fall below EBU R128 absolute threshold and yield -inf.
        # Fall back to an approximate loudness based on overall RMS across all channels.
        rms = float(np.sqrt(np.mean(data**2)))
        if rms > 0.0:
            loudness = 20.0 * math.log10(rms + 1e-12)
        else:
            # Truly silent – nothing to normalize; avoid exploding the noise floor.
            gain_db = 0.0
            if progress_callback:
                progress_callback(0.6)
            data_norm = np.clip(data, -1.0, 1.0)
            if progress_callback:
                progress_callback(0.8)
            out_bytes = float_matrix_to_frames(data_norm, sampwidth)
            if progress_callback:
                progress_callback(1.0)
            gains_db = [gain_db] * n_channels
            return out_bytes, gains_db

    gain_db = target - loudness
    # Optional safety clamp to avoid extreme boosts/cuts
    max_gain_db = 60.0
    if gain_db > max_gain_db:
        print(f"Warning: loudness gain {gain_db:.2f} dB exceeds +{max_gain_db} dB; clamping.")
        gain_db = max_gain_db
    if gain_db < -max_gain_db:
        print(f"Warning: loudness gain {gain_db:.2f} dB below -{max_gain_db} dB; clamping.")
        gain_db = -max_gain_db
    scale = 10.0 ** (gain_db / 20.0)

    if progress_callback:
        progress_callback(0.6)
    data_norm = np.clip(data * scale, -1.0, 1.0)

    if progress_callback:
        progress_callback(0.8)
    out_bytes = float_matrix_to_frames(data_norm, sampwidth)
    if progress_callback:
        progress_callback(1.0)

    gains_db = [gain_db] * n_channels
    return out_bytes, gains_db

