"""Plot offline visualizations from saved EMMF test outputs.

Example:
    # First run EMMF only_test to generate the outputs pickle:
    # python main.py --config_file configs/infer_configs/4Equipleth_EMMF.yaml
    #
    # Then create video-level visualization figures:
    # python visualization/plot_emmf_outputs.py \
    #   runs/exp/Equipleth_EMMF_10s_09/test_outputs/Equipleth_Equipleth_Equipleth_EMMF_Epoch52_Equipleth_outputs.pickle \
    #   --level video \
    #   --max-samples 20
"""

import argparse
import pickle
import random
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evaluation.metrics import _reform_data_from_dict


PLOT_KEYS = ("label", "fused", "rgb", "rf")
UNCERTAINTY_KEYS = ("gamma_rgb", "gamma_rf", "v_rgb", "v_rf")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create EMMF waveform, spectrum, modality, and uncertainty plots from an outputs pickle."
    )
    parser.add_argument("pickle_path", type=Path, help="Path to the *_outputs.pickle file saved during test.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated figures. Defaults to the pickle file parent directory.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help=(
            "Maximum number of samples to plot. In video level this means trials/subjects; "
            "in clip level this means clips. Use 0 or a negative value to plot all samples."
        ),
    )
    parser.add_argument(
        "--level",
        choices=("video", "clip"),
        default="video",
        help="Plot complete reconstructed trials/videos or individual clips.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=("sequential", "random"),
        default="sequential",
        help="Select samples in sorted order or randomly.",
    )
    parser.add_argument("--seed", type=int, default=100, help="Random seed used when --sample-mode random.")
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject IDs to plot. If omitted, samples are drawn from all subjects.",
    )
    parser.add_argument("--format", choices=("png", "pdf"), default="png", help="Figure file format.")
    parser.add_argument("--low-bpm", type=float, default=40.0, help="Lower frequency bound for spectrum plots.")
    parser.add_argument("--high-bpm", type=float, default=180.0, help="Upper frequency bound for spectrum plots.")
    return parser.parse_args()


def load_outputs(pickle_path):
    with pickle_path.open("rb") as handle:
        data = pickle.load(handle)

    visual_outputs = data.get("extra_outputs", {}).get("emmf_visual")
    if visual_outputs is None:
        raise ValueError(
            "The pickle does not contain extra_outputs['emmf_visual']. "
            "Run EMMF test after enabling visual output saving first."
        )
    return data, visual_outputs


def to_numpy(signal):
    if hasattr(signal, "detach"):
        signal = signal.detach().cpu().numpy()
    return np.asarray(signal, dtype=np.float64).reshape(-1)


def normalize_signal(signal):
    signal = to_numpy(signal)
    std = np.std(signal)
    if std < 1e-8:
        return signal - np.mean(signal)
    return (signal - np.mean(signal)) / std


def safe_name(value):
    if isinstance(value, tuple):
        value = "_".join(str(part) for part in value)
    value = str(value)
    value = re.sub(r"[^0-9A-Za-z_.-]+", "_", value)
    return value.strip("_") or "sample"


def sort_segment_key(key):
    if isinstance(key, tuple):
        return tuple(int(part) if str(part).isdigit() else str(part) for part in key)
    return (int(key),) if str(key).isdigit() else (str(key),)


def collect_clip_samples(visual_outputs, selected_subjects=None):
    selected_subjects = set(selected_subjects or [])
    samples = []
    for subject in sorted(visual_outputs.keys(), key=str):
        if selected_subjects and str(subject) not in selected_subjects:
            continue
        segments = visual_outputs[subject]
        for segment_key in sorted(segments.keys(), key=sort_segment_key):
            samples.append((subject, segment_key, segments[segment_key]))
    return samples


def merge_segments(segments):
    if not segments:
        return {}

    sample = {}
    for signal_key in PLOT_KEYS + UNCERTAINTY_KEYS:
        signal_segments = {
            segment_key: segment_sample[signal_key]
            for segment_key, segment_sample in segments.items()
        }
        sample[signal_key] = _reform_data_from_dict(signal_segments)
    return sample


def collect_video_samples(visual_outputs, selected_subjects=None):
    selected_subjects = set(selected_subjects or [])
    samples = []
    for subject in sorted(visual_outputs.keys(), key=str):
        if selected_subjects and str(subject) not in selected_subjects:
            continue
        samples.append((subject, "video", merge_segments(visual_outputs[subject])))
    return samples


def select_samples(samples, max_samples, sample_mode, seed):
    if sample_mode == "random":
        samples = list(samples)
        random.Random(seed).shuffle(samples)
    elif sample_mode != "sequential":
        raise ValueError("Unsupported sample mode: {}".format(sample_mode))

    if max_samples > 0:
        return samples[:max_samples]
    return samples


def get_samples(visual_outputs, level, selected_subjects=None, max_samples=20, sample_mode="sequential", seed=100):
    if level == "video":
        samples = collect_video_samples(visual_outputs, selected_subjects)
    elif level == "clip":
        samples = collect_clip_samples(visual_outputs, selected_subjects)
    else:
        raise ValueError("Unsupported visualization level: {}".format(level))
    return select_samples(samples, max_samples, sample_mode, seed)


def time_axis(sample, fs):
    length = len(to_numpy(sample["fused"]))
    return np.arange(length) / fs


def next_power_of_2(value):
    return 1 if value <= 1 else 2 ** (value - 1).bit_length()


def spectrum(signal, fs, low_bpm, high_bpm):
    signal = normalize_signal(signal)
    if signal.size < 2:
        return np.array([]), np.array([])
    signal = signal - np.mean(signal)
    nfft = next_power_of_2(signal.size)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    power = np.abs(np.fft.rfft(signal, n=nfft)) ** 2
    bpm = freqs * 60.0
    mask = (bpm >= low_bpm) & (bpm <= high_bpm)
    bpm = bpm[mask]
    power = power[mask]
    max_power = np.max(power) if power.size else 0.0
    if max_power > 0:
        power = power / max_power
    return bpm, power


def save_figure(fig, output_dir, stem, file_format):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "{}.{}".format(stem, file_format)
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return output_path


def plot_waveform(sample, fs, output_dir, stem, file_format):
    t = time_axis(sample, fs)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, normalize_signal(sample["label"]), label="GT PPG", linewidth=1.5)
    ax.plot(t, normalize_signal(sample["fused"]), label="Fused rPPG", linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title("Fused rPPG vs GT PPG")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    return save_figure(fig, output_dir / "waveforms", stem + "_waveform", file_format)


def plot_modal_compare(sample, fs, output_dir, stem, file_format):
    t = time_axis(sample, fs)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, normalize_signal(sample["rgb"]), label="RGB", linewidth=1.1)
    ax.plot(t, normalize_signal(sample["rf"]), label="RF", linewidth=1.1)
    ax.plot(t, normalize_signal(sample["fused"]), label="Fused", linewidth=1.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title("EMMF modality predictions")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    return save_figure(fig, output_dir / "modal_compare", stem + "_modal_compare", file_format)


def plot_spectrum(sample, fs, output_dir, stem, file_format, low_bpm, high_bpm):
    fig, ax = plt.subplots(figsize=(8, 4))
    for key, label in (("label", "GT PPG"), ("fused", "Fused"), ("rgb", "RGB"), ("rf", "RF")):
        bpm, power = spectrum(sample[key], fs, low_bpm, high_bpm)
        if bpm.size:
            ax.plot(bpm, power, label=label, linewidth=1.2)
    ax.set_xlabel("Heart rate (bpm)")
    ax.set_ylabel("Normalized power")
    ax.set_title("Power spectrum")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    return save_figure(fig, output_dir / "spectra", stem + "_spectrum", file_format)


def plot_uncertainty(sample, fs, output_dir, stem, file_format):
    t = time_axis(sample, fs)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, to_numpy(sample["gamma_rgb"]), label="gamma RGB", linewidth=1.2)
    axes[0].plot(t, to_numpy(sample["gamma_rf"]), label="gamma RF", linewidth=1.2)
    axes[0].set_ylabel("Gamma")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, to_numpy(sample["v_rgb"]), label="v RGB", linewidth=1.2)
    axes[1].plot(t, to_numpy(sample["v_rf"]), label="v RF", linewidth=1.2)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Evidence variance")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.suptitle("EMMF uncertainty parameters")
    return save_figure(fig, output_dir / "uncertainty", stem + "_uncertainty", file_format)


def ensure_required_keys(sample):
    missing = [key for key in PLOT_KEYS + UNCERTAINTY_KEYS if key not in sample]
    if missing:
        raise KeyError("Sample is missing required keys: {}".format(", ".join(missing)))


def main():
    args = parse_args()
    data, visual_outputs = load_outputs(args.pickle_path)
    output_dir = args.output_dir or args.pickle_path.parent
    fs = float(data.get("fs", 30))

    saved_paths = []
    samples = get_samples(
        visual_outputs,
        args.level,
        args.subjects,
        args.max_samples,
        args.sample_mode,
        args.seed,
    )
    for subject, segment_key, sample in samples:
        ensure_required_keys(sample)
        stem = "{}_{}".format(safe_name(subject), safe_name(segment_key))
        saved_paths.append(plot_waveform(sample, fs, output_dir, stem, args.format))
        saved_paths.append(plot_modal_compare(sample, fs, output_dir, stem, args.format))
        saved_paths.append(plot_spectrum(sample, fs, output_dir, stem, args.format, args.low_bpm, args.high_bpm))
        saved_paths.append(plot_uncertainty(sample, fs, output_dir, stem, args.format))

    print("Saved {} figures to {}".format(len(saved_paths), output_dir))


if __name__ == "__main__":
    main()
