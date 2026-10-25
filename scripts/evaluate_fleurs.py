#!/usr/bin/env python3
"""Evaluate Whisper ASR on human-read FLEURS audio under clean/noisy conditions.

Ablation: greedy decoding vs fixed beam search vs confidence-triggered adaptive
beam search. The known language is fixed across experiments to isolate decoding.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Audio, load_dataset
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from speech_ai.audio import add_awgn  # noqa: E402
from speech_ai.metrics import character_error_rate, corpus_error_rates, word_error_rate  # noqa: E402


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", default="small", help="Whisper size, e.g. base, small, medium")
    parser.add_argument("--n-per-language", type=int, default=50,
                        help="50 per language gives 100 unique human-read utterances by default")
    parser.add_argument("--languages", nargs="+", default=["hi_in", "kn_in"],
                        help="FLEURS configs; defaults to Hindi and Kannada")
    parser.add_argument("--noise-snr-db", type=float, default=10.0)
    parser.add_argument("--adaptive-threshold", type=float, default=-0.3,
                        help="Greedy avg segment log-probability below which beam search is triggered")
    parser.add_argument("--split", choices=["validation", "test"], default="test",
                        help="Use validation to select/freeze the threshold; report final results on test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Protect resume integrity: a partial checkpoint may only be resumed with
    # the exact same split/model/sample/seed/threshold/device settings.
    partial_path = output_dir / f"fleurs_{args.split}_predictions.partial.csv"
    final_path = output_dir / f"fleurs_{args.split}_predictions.csv"
    manifest_path = output_dir / f"fleurs_{args.split}_run_config.json"
    run_config = vars(args).copy()
    run_config["model_id"] = f"Systran/faster-whisper-{args.model_size}"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != run_config:
            raise RuntimeError(
                f"Run settings differ from existing manifest {manifest_path}. "
                "Use the same settings to resume, or choose a fresh --output-dir."
            )
    else:
        if partial_path.exists():
            raise RuntimeError(
                f"A checkpoint exists without a run manifest: {partial_path}. "
                "Refusing to risk mixing settings; inspect it or use a fresh --output-dir."
            )
        if final_path.exists():
            raise FileExistsError(
                f"Completed predictions already exist: {final_path}. "
                "Choose a fresh --output-dir to avoid overwriting results."
            )
        manifest_path.write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"Loading Whisper {args.model_size!r} (device={args.device}, compute_type={args.compute_type})")
    model = WhisperModel(args.model_size, device=args.device, compute_type=args.compute_type)

    # Per-utterance checkpointing: if Colab disconnects, completed utterances
    # can be resumed from this output directory instead of being recomputed.
    rows: list[dict] = []
    completed_ids: set[str] = set()
    if partial_path.exists():
        try:
            saved = pd.read_csv(partial_path)
            rows = saved.to_dict(orient="records")
            counts = saved.groupby("sample_id").size()
            completed_ids = set(counts[counts >= 6].index.astype(str))
            print(f"Resuming from checkpoint: {len(completed_ids)} utterances already complete")
        except Exception as exc:
            raise RuntimeError(f"Checkpoint exists but could not be read: {partial_path}: {exc}") from exc
    lang_map = {"hi_in": "hi", "kn_in": "kn", "ta_in": "ta", "te_in": "te"}
    for config in args.languages:
        language = lang_map.get(config, config.split("_")[0])
        print(f"\nLoading FLEURS config={config}; evaluating up to {args.n_per_language} utterances")
        dataset = load_dataset("google/fleurs", config, split=args.split)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
        n = min(args.n_per_language, len(dataset))
        if n < 1:
            raise RuntimeError(f"No examples found in FLEURS {config!r} {args.split} split")
        # Deterministic random sampling reduces dependence on dataset row ordering.
        dataset = dataset.shuffle(seed=args.seed + sum(ord(c) for c in config)).select(range(n))
        for idx in range(n):
            sample_id = f"{config}-{idx:04d}"
            if sample_id in completed_ids:
                if (idx + 1) % 10 == 0 or idx + 1 == n:
                    print(f"  {config}: {idx + 1}/{n} (checkpoint; skipped)")
                continue
            item = dataset[idx]
            audio_item = item["audio"]
            if isinstance(audio_item, dict):
                sample_rate = int(audio_item.get("sampling_rate", 16_000))
                audio_array = audio_item["array"]
            elif hasattr(audio_item, "get_all_samples"):
                decoded = audio_item.get_all_samples()
                sample_rate = int(decoded.sample_rate)
                audio_array = decoded.data
                if hasattr(audio_array, "cpu"):
                    audio_array = audio_array.cpu().numpy()
            else:
                raise TypeError(f"Unsupported FLEURS audio object: {type(audio_item).__name__}")
            audio = np.asarray(audio_array, dtype=np.float32).squeeze()
            if audio.ndim == 2:
                audio = audio.mean(axis=0)
            audio = audio.reshape(-1)
            # The FLEURS Audio column is cast to 16 kHz above. Fail loudly rather than
            # silently reporting RTF against a wrong sample rate.
            if sample_rate != 16_000:
                raise ValueError(f"Expected 16 kHz audio, got {sample_rate} Hz")
            reference = str(item.get("transcription", ""))
            audio_duration_s = len(audio) / sample_rate
            conditions = [("clean", audio)]
            conditions.append((f"white_noise_{args.noise_snr_db:g}dB", add_awgn(
                audio, snr_db=args.noise_snr_db, seed=args.seed + idx + sum(ord(c) for c in config)
            )))

            for condition, conditioned_audio in conditions:
                decoded = {}
                for beam_size, experiment in [(1, "baseline_greedy"), (5, "fixed_beam5")]:
                    start = time.perf_counter()
                    segments, info = model.transcribe(
                        conditioned_audio,
                        language=language,
                        task="transcribe",
                        beam_size=beam_size,
                        # faster-whisper defaults to a temperature schedule that can
                        # retry low-confidence/repetitive segments with non-zero-temperature
                        # sampling. Explicit 0.0 makes greedy/beam baselines reproducible.
                        temperature=0.0,
                        vad_filter=True,
                    )
                    segments = list(segments)
                    latency = time.perf_counter() - start
                    prediction = " ".join(segment.text.strip() for segment in segments).strip()
                    logprob = float(np.mean([segment.avg_logprob for segment in segments])) if segments else float("nan")
                    decoded[experiment] = {
                        "prediction": prediction,
                        "latency": latency,
                        "avg_logprob": logprob,
                        "info": info,
                    }
                    rows.append({
                        "sample_id": f"{config}-{idx:04d}",
                        "dataset": "google/fleurs",
                        "split": args.split,
                        "config": config,
                        "language": language,
                        "condition": condition,
                        "experiment": experiment,
                        "beam_size": beam_size,
                        "fallback_triggered": False,
                        "confidence_proxy": logprob if beam_size == 1 else np.nan,
                        "reference": reference,
                        "prediction": prediction,
                        "wer": word_error_rate(reference, prediction),
                        "cer": character_error_rate(reference, prediction),
                        "audio_duration_s": audio_duration_s,
                        "asr_latency_s": latency,
                        "rtf": latency / audio_duration_s if audio_duration_s else np.nan,
                        "detected_language": getattr(info, "language", None),
                        "language_probability": getattr(info, "language_probability", None),
                    })

                # Proposed improvement: low-confidence greedy outputs trigger a
                # second beam-5 pass. When triggered, latency includes both passes.
                greedy = decoded["baseline_greedy"]
                fallback = (not np.isfinite(greedy["avg_logprob"]) or
                            greedy["avg_logprob"] < args.adaptive_threshold)
                adaptive_prediction = decoded["fixed_beam5"]["prediction"] if fallback else greedy["prediction"]
                adaptive_latency = greedy["latency"] + decoded["fixed_beam5"]["latency"] if fallback else greedy["latency"]
                rows.append({
                    "sample_id": f"{config}-{idx:04d}",
                    "dataset": "google/fleurs",
                    "split": args.split,
                    "config": config,
                    "language": language,
                    "condition": condition,
                    "experiment": "adaptive_confidence_beam5",
                    "beam_size": 5,
                    "fallback_triggered": bool(fallback),
                    "confidence_proxy": greedy["avg_logprob"],
                    "reference": reference,
                    "prediction": adaptive_prediction,
                    "wer": word_error_rate(reference, adaptive_prediction),
                    "cer": character_error_rate(reference, adaptive_prediction),
                    "audio_duration_s": audio_duration_s,
                    "asr_latency_s": adaptive_latency,
                    "rtf": adaptive_latency / audio_duration_s if audio_duration_s else np.nan,
                    "detected_language": getattr(greedy["info"], "language", None),
                    "language_probability": getattr(greedy["info"], "language_probability", None),
                })
            # Each utterance yields six rows (3 methods x 2 conditions).
            # Write atomically so a runtime reset does not lose completed work.
            checkpoint_df = pd.DataFrame(rows)
            temp_path = partial_path.with_suffix(".tmp.csv")
            checkpoint_df.to_csv(temp_path, index=False, encoding="utf-8-sig")
            temp_path.replace(partial_path)
            completed_ids.add(sample_id)
            if (idx + 1) % 10 == 0 or idx + 1 == n:
                print(f"  {config}: {idx + 1}/{n} (checkpoint saved)")

    df = pd.DataFrame(rows)
    csv_path = output_dir / f"fleurs_{args.split}_predictions.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    if partial_path.exists():
        partial_path.unlink()

    summary: dict[str, dict] = {}
    grouped = df.groupby(["config", "language", "condition", "experiment"], dropna=False)
    summary_rows: list[dict] = []
    for (config, language, condition, experiment), group in grouped:
        aggregate = corpus_error_rates(group["reference"].tolist(), group["prediction"].tolist())
        out = {
            "config": config,
            "language": language,
            "condition": condition,
            "experiment": experiment,
            "n_samples": int(len(group)),
            "corpus_wer": aggregate["wer"],
            "corpus_cer": aggregate["cer"],
            "mean_latency_s": float(group["asr_latency_s"].mean()),
            "median_latency_s": float(group["asr_latency_s"].median()),
            "mean_rtf": float(group["rtf"].mean()),
            "median_rtf": float(group["rtf"].median()),
        }
        summary_rows.append(out)
        summary[f"{config}|{condition}|{experiment}"] = out
    if "fallback_triggered" in df.columns:
        adaptive = df[df["experiment"] == "adaptive_confidence_beam5"]
        trigger_rates = adaptive.groupby(["config", "language", "condition"])["fallback_triggered"].mean().to_dict()
        for record in summary_rows:
            if record["experiment"] == "adaptive_confidence_beam5":
                record["fallback_rate"] = float(trigger_rates.get((record["config"], record["language"], record["condition"]), 0.0))
    summary_df = pd.DataFrame(summary_rows).sort_values(["config", "condition", "experiment"])
    summary_df.to_csv(output_dir / f"fleurs_{args.split}_summary.csv", index=False, encoding="utf-8-sig")
    def package_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    metadata = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "settings": vars(args),
        "model_id": f"Systran/faster-whisper-{args.model_size}",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "torch_version": torch.__version__,
        "package_versions": {
            name: package_version(name)
            for name in ["transformers", "datasets", "faster-whisper", "ctranslate2", "huggingface-hub"]
        },
        "unique_utterances": int(df["sample_id"].nunique()),
        "conditions": ["clean", f"white_noise_{args.noise_snr_db:g}dB"],
        "groups": summary,
    }
    (output_dir / f"fleurs_{args.split}_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {csv_path}")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return df, summary


if __name__ == "__main__":
    evaluate(get_args())
