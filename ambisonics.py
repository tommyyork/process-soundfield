from __future__ import annotations

from typing import Tuple

import numpy as np


def _channel_rms_db(data: np.ndarray) -> np.ndarray:
    """
    Compute per-channel RMS in dBFS for a multichannel signal.

    data: (n_frames, n_channels) float array in [-1, 1].
    """
    if data.ndim != 2:
        raise ValueError(f"data must be 2D (n_frames, n_channels), got shape {data.shape!r}")
    # RMS per channel, clamp away from zero to avoid -inf
    rms = np.sqrt(np.mean(np.square(data), axis=0))
    rms = np.maximum(rms, 1e-12)
    return 20.0 * np.log10(rms)


def assert_gain_relationships_preserved(
    input_data: np.ndarray,
    output_data: np.ndarray,
    max_relative_diff_db: float = 6.0,
) -> None:
    """
    Assert that relative gain relationships between channels are broadly preserved.

    This is intentionally tolerant: it checks that per-channel RMS levels in dB,
    relative to the average across channels, have not changed by more than
    `max_relative_diff_db` between input and output.
    """
    if input_data.shape != output_data.shape:
        raise AssertionError(
            f"Shape mismatch between input and output data: "
            f"{input_data.shape!r} vs {output_data.shape!r}"
        )

    in_db = _channel_rms_db(input_data)
    out_db = _channel_rms_db(output_data)

    in_rel = in_db - float(np.mean(in_db))
    out_rel = out_db - float(np.mean(out_db))

    diff = np.abs(in_rel - out_rel)
    max_diff = float(np.max(diff))
    if max_diff > max_relative_diff_db:
        raise AssertionError(
            f"Relative channel gains changed by {max_diff:.2f} dB, "
            f"which exceeds allowed {max_relative_diff_db:.2f} dB.\n"
            f"Input rel dB: {in_rel}\nOutput rel dB: {out_rel}"
        )


def assert_phase_relationships_preserved(
    input_data: np.ndarray,
    output_data: np.ndarray,
    min_corr_magnitude: float = 0.1,
    max_corr_diff: float = 0.5,
) -> None:
    """
    Assert that inter-channel phase/shape relationships are broadly preserved.

    This compares the channel–channel correlation matrices of the input and
    output. For all channel pairs with at least `min_corr_magnitude` absolute
    correlation in either input or output, it checks that:

    - The sign of the correlation is unchanged (no global phase flips), and
    - The magnitude of the correlation does not change by more than `max_corr_diff`.
    """
    if input_data.shape != output_data.shape:
        raise AssertionError(
            f"Shape mismatch between input and output data: "
            f"{input_data.shape!r} vs {output_data.shape!r}"
        )

    if input_data.ndim != 2:
        raise ValueError(f"data must be 2D (n_frames, n_channels), got shape {input_data.shape!r}")

    n_channels = input_data.shape[1]
    if n_channels < 2:
        # Nothing to compare; trivially preserved.
        return

    # Correlation matrices between channels (n_channels x n_channels)
    corr_in = np.corrcoef(input_data, rowvar=False)
    corr_out = np.corrcoef(output_data, rowvar=False)

    # Examine upper triangle (i < j) only.
    idx_i, idx_j = np.triu_indices(n_channels, k=1)
    cin = corr_in[idx_i, idx_j]
    cout = corr_out[idx_i, idx_j]

    # Consider only pairs with sufficiently strong correlation in either matrix.
    mask = (np.abs(cin) >= min_corr_magnitude) | (np.abs(cout) >= min_corr_magnitude)
    if not np.any(mask):
        # All correlations are effectively negligible; nothing meaningful to assert.
        return

    cin_sel = cin[mask]
    cout_sel = cout[mask]

    # Check sign consistency (avoid phase flips).
    sign_mismatch = cin_sel * cout_sel < 0
    if np.any(sign_mismatch):
        raise AssertionError(
            "Sign of inter-channel correlations changed for some channel pairs, "
            "indicating potential phase inversion between channels.\n"
            f"Input corr (selected): {cin_sel}\nOutput corr (selected): {cout_sel}"
        )

    # Check that correlation magnitudes have not changed too much.
    corr_diff = np.abs(cin_sel - cout_sel)
    max_diff = float(np.max(corr_diff))
    if max_diff > max_corr_diff:
        raise AssertionError(
            f"Inter-channel correlation changed by {max_diff:.2f}, "
            f"which exceeds allowed {max_corr_diff:.2f}.\n"
            f"Input corr (selected): {cin_sel}\nOutput corr (selected): {cout_sel}"
        )

