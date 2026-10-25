#!/usr/bin/env python3
"""Compare Indic-Transcribe Core with Whisper on the same FLEURS examples.

Runs clean and deterministic additive-white-noise conditions. It loads the large
Indic checkpoint and Whisper sequentially to reduce peak GPU memory on Colab T4.
Model inference is not mocked: metrics are written only after real predictions.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Audio, load_dataset
from faster_whisper import WhisperModel
from huggingface_hub import HfApi, snapshot_download
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from speech_ai.audio import add_awgn  # noqa: E402
from speech_ai.metrics import corpus_error_rates, word_error_rate, character_error_rate  # noqa: E402

DEFAULT_MODEL_ID = "bodhan-ai/indic-transcribe-core"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--whisper-model-size", default="small")
    p.add_argument("--n-per-language", type=int, default=5)
    p.add_argument("--languages", nargs="+", default=["hi_in", "kn_in"])
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--noise-snr-db", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    p.add_argument("--compute-type", default="float16")
    p.add_argument("--output-dir", default=str(ROOT / "results" / "indic_comparison"))
    return p.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def release_gpu_memory(model_name: str) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Released {model_name} references and requested CUDA cache cleanup.")


def main() -> None:
    args = parse_args()
    if args.n_per_language < 1:
        raise ValueError("--n-per-language must be at least 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Select a Colab GPU runtime or use --device cpu.")

    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    lang_map = {"hi_in": "hi", "kn_in": "kn", "ta_in": "ta", "te_in": "te"}
    examples: list[dict] = []
    lang_to_examples: dict[str, list[dict]] = {}
    for config in args.languages:
        if config not in lang_map:
            raise ValueError(f"Unsupported FLEURS config {config!r}; add its ISO language mapping before evaluating it.")
        print(f"Loading FLEURS {config}/{args.split} …")
        dataset = load_dataset("google/fleurs", config, split=args.split)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
        n = min(args.n_per_language, len(dataset))
        if n == 0:
            raise RuntimeError(f"FLEURS config {config!r}, split {args.split!r} contains no examples")
        dataset = dataset.shuffle(seed=args.seed + sum(ord(c) for c in config)).select(range(n))
        lang_to_examples[config] = []
        for idx, item in enumerate(dataset):
            audio_item = item["audio"]
            if isinstance(audio_item, dict):
                sr = int(audio_item.get("sampling_rate", 16_000))
                audio_array = audio_item["array"]
            elif hasattr(audio_item, "get_all_samples"):
                decoded = audio_item.get_all_samples()
                sr = int(decoded.sample_rate)
                audio_array = decoded.data
                if hasattr(audio_array, "cpu"):
                    audio_array = audio_array.cpu().numpy()
            else:
                raise TypeError(f"Unsupported FLEURS audio object: {type(audio_item).__name__}")

            audio = np.asarray(audio_array, dtype=np.float32).squeeze()
            # AudioDecoder data is channel-first; downmix stereo if encountered.
            if audio.ndim == 2:
                audio = audio.mean(axis=0)
            audio = audio.reshape(-1)
            if sr != 16_000:
                raise ValueError(f"Expected 16 kHz after cast_column, got {sr} Hz")
            dataset_id = item.get("id")
            dataset_id_text = str(dataset_id) if dataset_id is not None and str(dataset_id).strip() else None
            # Selected-row index guarantees unique/safe file names even if a source ID is blank.
            sample_id = f"{config}-{idx:04d}"
            record = {
                "sample_id": sample_id,
                "dataset_id": dataset_id_text,
                "config": config,
                "language": lang_map[config],
                "split": args.split,
                "reference": str(item.get("transcription", "")),
                "audio": audio,
                "sample_rate": sr,
                "audio_duration_s": len(audio) / sr,
                "example_index": idx,
            }
            lang_to_examples[config].append(record)
            examples.append(record)
        print(f"  Selected {n} utterances")

    # Cache paths for the exact same samples and conditions across both models.
    conditions: dict[tuple[str, str], Path] = {}
    for ex in examples:
        clean_path = audio_dir / f"{ex['sample_id']}_clean.wav"
        sf.write(str(clean_path), ex["audio"], ex["sample_rate"], subtype="PCM_16")
        conditions[(ex["sample_id"], "clean")] = clean_path
        noisy = add_awgn(
            ex["audio"],
            snr_db=args.noise_snr_db,
            seed=args.seed + ex["example_index"] + sum(ord(c) for c in ex["config"]),
        )
        noisy_path = audio_dir / f"{ex['sample_id']}_noise.wav"
        sf.write(str(noisy_path), noisy, ex["sample_rate"], subtype="PCM_16")
        conditions[(ex["sample_id"], f"white_noise_{args.noise_snr_db:g}dB")] = noisy_path

    rows: list[dict] = []
    # Resolve and record the precise model commit used by this run.
    print(f"Resolving gated model {args.model_id!r}; access must have been approved first.")
    model_info = HfApi().model_info(args.model_id)
    model_revision = model_info.sha
    model_dir = snapshot_download(args.model_id, revision=model_revision)
    model_dir_str = str(model_dir)
    if model_dir_str not in sys.path:
        sys.path.insert(0, model_dir_str)

    print("Loading Indic-Transcribe Core …")
    from indic_transcribe import IndicTranscribe
    indic_model = IndicTranscribe.from_pretrained(model_dir_str)
    for ex in examples:
        for condition in ["clean", f"white_noise_{args.noise_snr_db:g}dB"]:
            wav_path = conditions[(ex["sample_id"], condition)]
            start = time.perf_counter()
            prediction = str(indic_model(str(wav_path), lang=ex["language"])).strip()
            latency = time.perf_counter() - start
            row = {k: ex[k] for k in ["sample_id", "dataset_id", "config", "language", "split", "reference", "audio_duration_s"]}
            row.update({
                "model": "indic-transcribe-core",
                "model_id": args.model_id,
                "model_revision": model_revision,
                "condition": condition,
                "prediction": prediction,
                "wer": word_error_rate(ex["reference"], prediction),
                "cer": character_error_rate(ex["reference"], prediction),
                "asr_latency_s": latency,
                "rtf": latency / ex["audio_duration_s"] if ex["audio_duration_s"] else float("nan"),
            })
            rows.append(row)
        if (ex["example_index"] + 1) % 10 == 0:
            print(f"Indic-Transcribe: {ex['config']} {ex['example_index'] + 1} examples")
    del indic_model
    release_gpu_memory("Indic-Transcribe Core")

    print(f"Loading Whisper {args.whisper_model_size!r} baseline …")
    whisper_model = WhisperModel(
        args.whisper_model_size,
        device=args.device,
        compute_type=args.compute_type,
    )
    for ex in examples:
        for condition in ["clean", f"white_noise_{args.noise_snr_db:g}dB"]:
            wav_path = conditions[(ex["sample_id"], condition)]
            start = time.perf_counter()
            segments, _info = whisper_model.transcribe(
                str(wav_path), language=ex["language"], task="transcribe",
                beam_size=1, temperature=0.0, vad_filter=True
            )
            segments = list(segments)  # faster-whisper transcribes lazily; timing must include iteration
            prediction = " ".join(seg.text.strip() for seg in segments).strip()
            latency = time.perf_counter() - start
            row = {k: ex[k] for k in ["sample_id", "dataset_id", "config", "language", "split", "reference", "audio_duration_s"]}
            row.update({
                "model": f"whisper-{args.whisper_model_size}-greedy",
                "model_id": f"Systran/faster-whisper-{args.whisper_model_size}",
                "model_revision": None,
                "condition": condition,
                "prediction": prediction,
                "wer": word_error_rate(ex["reference"], prediction),
                "cer": character_error_rate(ex["reference"], prediction),
                "asr_latency_s": latency,
                "rtf": latency / ex["audio_duration_s"] if ex["audio_duration_s"] else float("nan"),
            })
            rows.append(row)
        if (ex["example_index"] + 1) % 10 == 0:
            print(f"Whisper: {ex['config']} {ex['example_index'] + 1} examples")
    del whisper_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    predictions = pd.DataFrame(rows)
    pred_file = output_dir / f"indic_vs_whisper_{args.split}_predictions.csv"
    predictions.to_csv(pred_file, index=False, encoding="utf-8-sig")
    summary_rows: list[dict] = []
    for (config, language, condition, model_name), group in predictions.groupby(
        ["config", "language", "condition", "model"], dropna=False
    ):
        metric = corpus_error_rates(group["reference"].tolist(), group["prediction"].tolist())
        summary_rows.append({
            "config": config,
            "language": language,
            "condition": condition,
            "model": model_name,
            "n_samples": int(len(group)),
            "corpus_wer": metric["wer"],
            "corpus_cer": metric["cer"],
            "mean_latency_s": float(group["asr_latency_s"].mean()),
            "median_latency_s": float(group["asr_latency_s"].median()),
            "mean_rtf": float(group["rtf"].mean()),
            "median_rtf": float(group["rtf"].median()),
        })
    summary = pd.DataFrame(summary_rows).sort_values(["config", "condition", "model"])
    summary_file = output_dir / f"indic_vs_whisper_{args.split}_summary.csv"
    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")
    metadata = {
        "settings": vars(args),
        "indic_model_revision": model_revision,
        "indic_snapshot_path": model_dir_str,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "torch_version": torch.__version__,
        "package_versions": {
            name: package_version(name)
            for name in ["transformers", "datasets", "faster-whisper", "ctranslate2", "huggingface-hub"]
        },
        "n_unique_utterances": int(predictions["sample_id"].nunique()),
        "conditions": ["clean", f"white_noise_{args.noise_snr_db:g}dB"],
        "note": "Indic-Transcribe's reported checkpoint/commit and package versions are recorded. No model score is assumed; all metrics are computed from predictions in this run.",
    }
    metadata_file = output_dir / f"indic_vs_whisper_{args.split}_metadata.json"
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nPredictions: {pred_file}")
    print(f"Summary: {summary_file}")
    print(f"Metadata: {metadata_file}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
