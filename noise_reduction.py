from typing import Any, Dict, Tuple

import numpy as np
from sklearn.decomposition import PCA

try:
    import noisereduce as _nr  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    _nr = None


def pca_noise_reduction(
    data: np.ndarray,
    n_components: int | None = None,
    progress_callback=None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Reduce noise by reconstructing the multichannel signal from the top PCA components.

    Lower-variance components are treated as noise and dropped.

    Parameters
    ----------
    data:
        (n_frames, n_channels) float array in [-1, 1].
    n_components:
        Number of PCA components to keep (default: n_channels - 1, min 1, max n_channels).
    progress_callback:
        Optional callable taking a single float in [0, 1]; called with 0.5 after PCA fit
        and 1.0 after reconstruction.

    Returns
    -------
    reconstructed:
        Denoised signal with same shape as ``data``.
    stats:
        Dict with variance and per-channel level information.
    """
    n_frames, n_channels = data.shape
    if n_components is None:
        n_components = max(1, n_channels - 1)
    n_components = min(n_components, n_channels)

    # Fit full PCA to get variance ratios for all components
    n_full = min(n_frames, n_channels)
    pca_full = PCA(n_components=n_full)
    pca_full.fit(data)
    explained_ratio = pca_full.explained_variance_ratio_
    if progress_callback:
        progress_callback(0.5)

    # Fit with kept components and reconstruct
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data)
    reconstructed = pca.inverse_transform(transformed)
    out = np.clip(reconstructed, -1.0, 1.0).astype(np.float64)
    if progress_callback:
        progress_callback(1.0)

    variance_retained = float(np.sum(explained_ratio[:n_components]))
    variance_dropped = (
        float(np.sum(explained_ratio[n_components:])) if n_components < n_full else 0.0
    )

    # RMS before/after per channel (in linear)
    rms_before = np.sqrt(np.mean(data**2, axis=0))
    rms_after = np.sqrt(np.mean(out**2, axis=0))

    # Preserve per-channel gain relationships by re-equalizing the RMS of each
    # output channel to match its input RMS (where meaningful). This keeps the
    # relative channel gains intact while still benefiting from the PCA-based
    # noise suppression in the time/frequency content.
    safe = (rms_before > 1e-10) & (rms_after > 1e-12)
    scales = np.ones_like(rms_before)
    scales[safe] = rms_before[safe] / rms_after[safe]
    out *= scales[np.newaxis, :]
    out = np.clip(out, -1.0, 1.0)

    # Recompute RMS after gain compensation for reporting.
    rms_after = np.sqrt(np.mean(out**2, axis=0))
    rms_change_db = 20.0 * np.log10(
        np.where(rms_before > 1e-10, rms_after / rms_before, 1.0)
    )

    stats: Dict[str, Any] = {
        "algorithm": "pca",
        "explained_variance_ratio": explained_ratio,
        "variance_retained": variance_retained,
        "variance_dropped": variance_dropped,
        "n_components_kept": n_components,
        "n_components_dropped": n_full - n_components,
        "rms_change_db_per_channel": rms_change_db,
    }
    return out, stats


def noisereduce_spectral_gate(
    data: np.ndarray,
    sample_rate: int,
    intensity: int | None = None,
    stationary: bool = False,
    progress_callback=None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Reduce noise using the Noisereduce spectral gating algorithm.

    This wraps ``noisereduce.reduce_noise`` as described by Sainburg & Zorea
    in their domain-general noise reduction paper
    (https://www.nature.com/articles/s41598-025-13108-x)
    and in the reference implementation at `timsainb/noisereduce`.

    Parameters
    ----------
    data:
        (n_frames, n_channels) float array in [-1, 1].
    sample_rate:
        Sampling rate in Hz.
    intensity:
        Optional integer 1–10 controlling the strength of the reduction.
        1 ≈ gentle (prop_decrease ≈ 0.3), 10 ≈ strong (prop_decrease = 1.0).
        If None, uses the library default (1.0).
    stationary:
        If True, use the stationary variant; otherwise use the non-stationary
        sliding-window variant recommended for varying background noise.
    progress_callback:
        Optional callable taking a single float in [0, 1]. Called with 0.0
        before processing and 1.0 after.

    Returns
    -------
    denoised:
        Denoised signal with same shape as ``data``.
    stats:
        Dict with configuration and per-channel RMS change information.
    """
    if _nr is None:
        raise RuntimeError(
            "The 'noisereduce' package is required for -noisereduce. "
            "Install it with 'pip install noisereduce'."
        )

    if data.ndim != 2:
        raise ValueError(
            f"'data' must be 2D (n_frames, n_channels); got shape {data.shape!r}"
        )

    n_frames, n_channels = data.shape

    if intensity is not None:
        try:
            intensity_int = int(intensity)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"intensity must be an integer between 1 and 10, got {intensity!r}"
            ) from exc
        if intensity_int < 1 or intensity_int > 10:
            raise ValueError(
                f"intensity must be between 1 and 10 (got {intensity_int})"
            )
        # Map 1–10 → prop_decrease ∈ [0.3, 1.0]
        prop_decrease = 0.3 + (intensity_int - 1) * (0.7 / 9.0)
    else:
        intensity_int = None
        prop_decrease = 1.0

    if progress_callback:
        progress_callback(0.0)

    # noisereduce expects (n_channels, n_frames) or (n_frames,)
    if n_channels == 1:
        y = data[:, 0]
    else:
        y = data.T  # (n_channels, n_frames)

    reduced = _nr.reduce_noise(
        y=y,
        sr=sample_rate,
        stationary=stationary,
        prop_decrease=prop_decrease,
    )

    if progress_callback:
        progress_callback(1.0)

    if n_channels == 1:
        out = np.asarray(reduced, dtype=np.float64)[:, np.newaxis]
    else:
        out = np.asarray(reduced, dtype=np.float64).T

    out = np.clip(out, -1.0, 1.0)

    # Simple stats: per-channel RMS change
    rms_before = np.sqrt(np.mean(data**2, axis=0))
    rms_after = np.sqrt(np.mean(out**2, axis=0))
    rms_change_db = 20.0 * np.log10(
        np.where(rms_before > 1e-10, rms_after / rms_before, 1.0)
    )

    stats: Dict[str, Any] = {
        "algorithm": "noisereduce",
        "stationary": stationary,
        "intensity": intensity_int,
        "prop_decrease": prop_decrease,
        "rms_change_db_per_channel": rms_change_db,
        "n_frames": n_frames,
        "n_channels": n_channels,
    }
    return out, stats

