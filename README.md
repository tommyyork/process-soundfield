# process-a-format
This script and its supporting modules written primarily using **Cursor** IDE’s AI coding assistant (GPT‑5.1-based), as a test of its performance as of March, 2026. It took a couple hours, and pretty specific back and forth about what features to include and how to implement specific noise reduction or normalization algirithms in the context of an ambisonic recording. It was written to process recordings I took in Joshua Tree, California, using a Rode NT-SF1 microphone and a MixPre-10 II with the ambisonics plugin, which recorded both stereo channels and A format ambisonic channels to wav files. YAMNet (available via the `-describe` flag) did a surprisingly good job of identifying animal noises, owls, and so on!

A command-line tool that processes **A-format ambisonic** WAV files by normalizing levels and applying configurable noise reduction. It reads a multichannel WAV (4+ channels), optionally selects and reorders channels via `-input` and `-output`, normalizes channels (either to 0 dB FS peak or to a target EBU R128 loudness with `-loudnorm`), drops silent or near-silent channels, then reduces noise via either PCA or the **Noisereduce** spectral-gating algorithm and writes a multichannel WAV suitable. Phase and gain relationships should be preserved using the default noise reduction algorithm (noisereduce) and either volume normalization option. However, PCA-based noise reduction will not.

## What it does

1. **Load** — Reads a WAV file (16-, 24-, or 32-bit PCM, 4+ channels).
2. **Time slice** — If `-ss` and/or `-to` are given, trims the loaded audio to the specified time range (in seconds or ffmpeg-style `HH:MM:SS` timestamps) before any other processing. Omit `-ss` to start at the beginning; omit `-to` to keep from the start time to the end of the file.
3. **Channel selection** — If `-input` is given, keeps only the listed 1-based channels in that order; otherwise uses all channels. Optionally `-output` specifies the order of those channels in the written file (a permutation of the `-input` list).
4. **Normalize** — By default, peak-normalizes all channels **together** with a single gain so that the peak across any channel reaches 0 dB FS (full scale). With `-loudnorm`, uses EBU R128 loudness normalization (target -23 LUFS) instead. In both cases, the same gain is applied to every channel and per-channel gain is still reported for each channel.
5. **Six-channel rule** — If exactly 6 channels remain after normalization, the first two channels (indices 0 and 1) are dropped so the output has 4 channels.
6. **Drop silent channels** — Any channel that required more than 96 dB gain (effectively silence or noise) is dropped and not written.
7. **Noise reduction** — Reduces noise in the multichannel signal using either:
   - **Noisereduce spectral-gating** (default): uses the spectral-gating algorithm from the domain-general noise reduction method of Sainburg & Zorea (*Sci Rep* 15, 30905, 2025; [`https://www.nature.com/articles/s41598-025-13108-x`](https://www.nature.com/articles/s41598-025-13108-x)). This mode is designed to better preserve ambisonic gain and phase relationships between channels.
   - **PCA-based noise reduction** (with `-pca`): runs PCA on the multichannel signal and reconstructs from the top *k* components; lower-variance components are treated as noise. Even with additional safeguards, PCA can still alter per-channel gain and inter-channel phase relationships, potentially damaging ambisonic information, so this mode should be used with caution.

   The `-nr` value controls the reduction intensity for **both** PCA and Noisereduce modes, and can also disable noise reduction entirely (see below).
8. **Describe (optional)** — If `-describe` is given, the tool runs **YAMnet** (a pre-trained audio event classifier from TensorFlow Hub) on the processed audio (after PCA noise reduction, just before writing). The audio is mixed to mono, resampled to 16 kHz, and analyzed in fixed-llength segments. The tool prints a table to the terminal with at most **5 events per minute** of audio. Each row has: start time (s), end time (s), YAMnet’s predicted label for that segment, and confidence (0–1). No other processing is skipped; the table is informational only.
9. **Write** — Writes the result to the output WAV with the same sample width and frame rate.

The tool prints summary statistics for normalization (per-channel gain in dB) and for PCA (variance retained/dropped, explained variance per component, RMS change per channel).

## Usage

```text
python process-a-format.py [-input CH1,CH2,...] [-output CH1,CH2,...] [-nr 0-10] [-ss START] [-to END] [-describe] [-loudnorm] [-pca] <input.wav> <output.wav>
```

- **<input.wav>** — Input WAV path (must have 4+ channels, uncompressed PCM).
- **<output.wav>** — Output WAV path.

## Options summary

Short reference in the style of a `--help` summary (one option and description per line):

```text
  -input CH1,CH2,...     Select which channels to use and their order (1-based, comma-separated; at least 4).
  -output CH1,CH2,...   Order of those channels in the output file (permutation of -input).
  -ss START             Start time for time slice (seconds or HH:MM:SS); omit to start at beginning.
  -to END               End time for time slice (seconds or HH:MM:SS); omit to keep to end of file.
  -nr 0-10              Noise reduction intensity (1=minimal, 10-maximal); 0 disables noise reduction.
  -describe             Run YAMnet and print event table (start, end, label, confidence; max 5 events/min).
  -loudnorm             Apply EBU R128 loudness normalization (target -23 LUFS) instead of 0 dB peak normalization.
  -pca                  Use PCA-based noise reduction instead of the default Noisereduce spectral-gating algorithm.
```

## Command-line options

### `-input CH1,CH2,...`

Select which channels of the file to use and their **order**, as a **1-based comma-separated** list. You can pick any subset of channels (at least 4) and list them in the order you want for processing.

- **Implementation:** After validating that the file has at least 4 channels, the script parses the list (e.g. `2,3,4,5`). Each value must be between 1 and the file’s channel count. The list is converted to 0-based indices in the same order (e.g. `2,3,4,5` → indices 1, 2, 3, 4). The input file is read in full; then a helper extracts only these channels from the interleaved frame buffer **in the order given**. All later steps (normalization, drop rules, PCA) run on this ordered subset.
- **Example:** On a 10-channel file, `-input 2,3,4,5` uses the 2nd, 3rd, 4th, and 5th channels in that order. You can also use `-input 5,4,3,2` to use the same four channels in reverse order.

### `-output CH1,CH2,...`

Set the **order of channels in the output file**. Must be a permutation of the channel numbers given to `-input` (same set, any order). If omitted, the output channel order is unchanged from the processing order (i.e. the order defined by `-input` or the file).

- **Implementation:** The script parses the list and checks that it contains exactly the same 1-based channel numbers as `-input`, with no duplicates or omissions. It then builds a reorder index so that output column *i* is the selected channel that corresponds to the *i*-th value in `-output`. This reorder is applied to the final data matrix immediately before encoding to bytes and writing the WAV. All processing (normalization, PCA, etc.) uses the `-input` order; only the written file reflects `-output`.
- **Example:** With `-input 2,3,4,5`, adding `-output 5,4,3,2` writes the file with channels reversed: file channel 5 first, then 4, then 3, then 2.

### `-describe`

Run **YAMnet** (TensorFlow Hub’s pre-trained audio event classifier) on the loaded audio and print an **event table** to the terminal. The table has four columns: **Start (s)**, **End (s)**, **Label** (YAMnet’s guess for that segment), and **Confidence** (0–1). There are at most **5 events per minute** of audio (segments are fixed-length buckets; the top-scoring class in each bucket is reported).

- **Implementation:** When `-describe` is present, after the WAV is loaded (and after any time slice and channel selection), the script mixes all channels to mono and resamples to 16 kHz if needed. It loads the YAMnet model from TensorFlow Hub and runs it on the waveform. YAMnet outputs 521 class scores per short window (~0.48 s hop). The audio is divided into consecutive segments of 12 seconds (5 per minute). For each segment, the script takes the top-scoring class across its windows and records the segment’s start time, end time, that label, and the confidence (max score). The table is printed to the terminal; processing then continues with normalization, PCA, and writing as usual. Requires **tensorflow** and **tensorflow_hub** (`pip install tensorflow tensorflow_hub`).

### `-ss START` and `-to END`

Trim the audio to a **time range** before any other processing. Times can be given as **seconds** (decimal allowed) or as **timestamp strings** in ffmpeg-style `HH:MM:SS` (and optionally `HH:MM:SS.xxx` for fractional seconds).

- **Seconds:** `10`, `65`, `1.5` — interpreted as time in seconds.
- **Timestamp:** `00:01:05` = 0 hours, 1 minute, 5 seconds = 65 seconds; `01:00:01` = 1 hour and 1 second = 3601 seconds. Use `HH:MM:SS` or `HH:MM:SS.xxx` (e.g. `00:01:05.500` for 65.5 seconds).
- `**-ss START`** — Start time of the slice. Only frames at or after this time are kept. If omitted, the slice starts at the beginning of the file (0 seconds).
- `**-to END**` — End time of the slice. Only frames before this time are kept. If omitted, nothing is trimmed from the end: the slice runs from the start time (or 0) to the end of the file.
- **Implementation:** After the WAV is loaded (and the frame buffer and sample rate are known), the script parses each value with a helper: if the string contains `:`, it is parsed as `HH:MM:SS` or `HH:MM:SS.xxx` (hours × 3600 + minutes × 60 + seconds); otherwise it is parsed as a float (seconds). The resulting start and end times in seconds are converted to frame indices using the file’s frame rate, clamped to the valid range, and the interleaved frame buffer is sliced. Channel selection, normalization, and all later steps run on this trimmed buffer.
- **Examples:**
  - `-ss 10` or `-ss 00:00:10` — Drop the first 10 seconds; keep from 10 s to the end.
  - `-to 60` or `-to 00:01:00` — Keep only the first 60 seconds.
  - `-ss 00:01:05 -to 00:05:30` — Keep only the segment from 1 min 5 s to 5 min 30 s (65 s to 330 s).
  - `-ss 01:00:01` — Keep from 1 hour 1 second to the end.

### `-nr 0-10`

Control the **intensity of noise reduction** (PCA or Noisereduce) from 1 (minimal) to 10 (maximal). With `-nr 0`, the noise reduction step is explicitly skipped.

- **Implementation (Noisereduce mode, default):** When `-pca` is **not** supplied, noise reduction uses the Noisereduce spectral-gating algorithm. The `-nr` value 1–10 is mapped to the underlying `prop_decrease` parameter to control spectral-gating strength (higher values apply stronger reduction); `-nr 0` still disables the noise reduction step entirely. This mode is generally preferred for preserving the ambisonic scene because it is designed to better maintain inter-channel gain and phase relationships.

- **Implementation (PCA mode, enabled with `-pca`):** PCA is run on the multichannel float matrix (after normalization and channel dropping). The number of components kept, *k*, is derived from `-nr` and the current channel count *n*:
  - `-nr` omitted: *k* = max(1, *n* − 1). One component is dropped (original default behavior).
  - `-nr 0`: Noise reduction is skipped entirely; the script prints that the PCA/Noisereduce step would normally run but has been skipped.
  - `-nr 1`: Minimal reduction — *k* = *n* − 1 (keep all but one component).
  - `-nr 10`: Maximal reduction — *k* = 1 (keep only the first principal component).
  - `-nr 2` … `-nr 9`: *k* is interpolated linearly between the above:  
    *k* = round(1 + max(0, *n* − 2) × (10 − *nr*) / 9), then clamped to [1, *n*].

  So for **4 channels**, for example:

  | `-nr` | Components kept |
  |-------|-----------------|
  | 1     | 3               |
  | 2–3   | 3               |
  | 4–7   | 2               |
  | 8–10  | 1               |

  The script fits PCA, keeps the top *k* components, reconstructs the signal from them, and writes the result. Even with additional safeguards (such as per-channel RMS re-equalization), PCA can still modify inter-channel phase and gain relationships, so when working with ambisonic material you should only use `-pca` if you are comfortable with this trade-off.

### `-loudnorm`

Apply **EBU R128 loudness normalization** instead of the default 0 dB FS peak normalization.

- **Implementation:** When `-loudnorm` is present, the script uses `pyloudnorm`’s EBU R128 `Meter` to measure integrated loudness (in LUFS) over all channels and computes a single gain so that the overall loudness reaches **-23 LUFS** (typical EBU R128 target). The same gain (in dB) is applied to every channel by scaling the float waveform before converting back to PCM. The per-channel gain values printed in the normalization summary are all this loudness-based gain. The six-channel rule and silent-channel drop (based on gain > 96 dB) still apply after loudness normalization.

## Requirements

- Python 3 with standard library (`wave`, `array`, `math`, `sys`, `os`, `csv`)
- **numpy**
- **scikit-learn** (for `sklearn.decomposition.PCA`)
- **tensorflow** and **tensorflow_hub** (only for the optional `-describe` flag)
- **pyloudnorm** (only for the optional `-loudnorm` flag)

Install dependencies (e.g. from a `requirements.txt` in the same or parent directory) with:

```text
pip install numpy scikit-learn
```

For the optional **-describe** feature (YAMnet audio description):

```text
pip install tensorflow tensorflow_hub
```

For the optional **-loudnorm** feature (EBU R128 loudness normalization):

```text
pip install pyloudnorm
```

## Supported formats

- **PCM bit depths:** 16-, 24-, and 32-bit integer PCM (uncompressed WAV).
- **Channels:** At least 4 channels required; channel selection and dropping may reduce the count before noise reduction and output.

