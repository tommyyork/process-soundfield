from __future__ import annotations

import sys
from typing import Callable, List, Tuple

import numpy as np


def describe_audio_with_yamnet(
    waveform_mono_16k: np.ndarray,
    duration_sec: float,
    max_events_per_minute: int = 5,
    progress_callback: Callable[[float], None] | None = None,
) -> List[Tuple[float, float, str, float]]:
    """
    Run YAMnet on mono 16 kHz float waveform. Returns at most (duration_sec/60)*max_events_per_minute
    events, each (start_sec, end_sec, label, confidence). Uses 12-second buckets (5 per minute).
    progress_callback(fraction) is called with 0.0..1.0 during loading and inference.
    """
    try:
        import csv
        import types

        # Ensure pkg_resources exists (tensorflow_hub and friends sometimes require it)
        if "pkg_resources" not in sys.modules:
            try:
                import pkg_resources  # type: ignore[unused-import]
            except ImportError:
                stub = types.ModuleType("pkg_resources")
                try:
                    from packaging.version import parse as _parse_version  # type: ignore[unused-import]

                    def _pv(v: str) -> object:
                        return _parse_version(v)

                    stub.parse_version = _pv  # type: ignore[attr-defined]
                except Exception:

                    def _pv(v: str) -> str:
                        return v

                    stub.parse_version = _pv  # type: ignore[attr-defined]
                sys.modules["pkg_resources"] = stub
        import tensorflow as tf
        import tensorflow_hub as hub
    except ImportError as e:
        print(f"Error: {e}")
        raise SystemExit(1) from e

    # Load YAMnet (expects 16 kHz mono float32)
    if progress_callback:
        progress_callback(0.05)
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    if progress_callback:
        progress_callback(0.25)
    # Class names from model's CSV (path is inside hub cache)
    class_map_path = model.class_map_path().numpy()
    if isinstance(class_map_path, bytes):
        class_map_path = class_map_path.decode("utf-8")
    class_names: List[str] = []
    with tf.io.gfile.GFile(class_map_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names.append(row["display_name"])
    if progress_callback:
        progress_callback(0.35)
    # Ensure float32 and 1-D
    wav = np.array(waveform_mono_16k, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    # Run inference
    if progress_callback:
        progress_callback(0.4)
    scores, embeddings, spectrogram = model(wav)
    if progress_callback:
        progress_callback(0.9)
    scores = scores.numpy()  # (num_patches, 521)
    # YAMnet: ~0.48 s hop, so patch i covers ~[i*0.48, i*0.48+0.96]
    hop_sec = 0.48
    bucket_sec = 60.0 / max_events_per_minute  # 12 s for 5 per minute
    events: List[Tuple[float, float, str, float]] = []
    t = 0.0
    while t < duration_sec:
        end_bucket = min(t + bucket_sec, duration_sec)
        start_patch = int(t / hop_sec)
        end_patch = int(end_bucket / hop_sec)
        if start_patch >= scores.shape[0]:
            break
        end_patch = min(end_patch, scores.shape[0])
        bucket_scores = scores[start_patch:end_patch]  # (n, 521)
        max_per_class = np.max(bucket_scores, axis=0)
        top_idx = int(np.argmax(max_per_class))
        confidence = float(np.max(max_per_class))
        label = class_names[top_idx] if top_idx < len(class_names) else f"class_{top_idx}"
        events.append((t, end_bucket, label, confidence))
        t = end_bucket
    if progress_callback:
        progress_callback(1.0)
    return events

