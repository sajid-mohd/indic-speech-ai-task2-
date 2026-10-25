#!/usr/bin/env python3
"""Compare adaptive decoding thresholds offline from one fixed validation run.

The evaluation CSV must contain deterministic baseline_greedy and fixed_beam5 rows
for the same (sample_id, condition) pairs. This script derives adaptive rows from
those two stored outputs, so every threshold is compared on exactly the same
transcripts and measured per-pass latency. No model inference is performed here.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from speech_ai.metrics import (  # noqa: E402
    character_error_rate,
    corpus_error_rates,
    word_error_rate,
)

KEYS = ["sample_id", "config", "language", "condition"]
REQUIRED = set(KEYS + [
    "experiment", "reference", "prediction", "confidence_proxy",
    "audio_duration_s", "asr_latency_s", "split",
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True,
                        help="Predictions CSV from one deterministic validation run")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for threshold-specific predictions and summaries")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[-0.4, -0.6, -0.8])
    return parser.parse_args()


def _check_unique(df: pd.DataFrame, label: str) -> None:
    if df.duplicated(KEYS).any():
        examples = df.loc[df.duplicated(KEYS, keep=False), KEYS].head(5)
        raise ValueError(f"Duplicate {label} rows by sample/condition. Examples:\n{examples}")


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict] = []
    groups = df.groupby(["config", "language", "condition", "experiment"], dropna=False)
    for (config, language, condition, experiment), group in groups:
        rates = corpus_error_rates(group["reference"].tolist(), group["prediction"].tolist())
        row = {
            "config": config,
            "language": language,
            "condition": condition,
            "experiment": experiment,
            "n_samples": int(group["sample_id"].nunique()),
            "corpus_wer": rates["wer"],
            "corpus_cer": rates["cer"],
            "mean_latency_s": float(group["asr_latency_s"].mean()),
            "median_latency_s": float(group["asr_latency_s"].median()),
            "mean_rtf": float(group["rtf"].mean()),
            "median_rtf": float(group["rtf"].median()),
        }
        if experiment.startswith("adaptive_threshold_"):
            row["fallback_rate"] = float(group["fallback_triggered"].mean())
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).sort_values(
        ["config", "condition", "experiment"]
    ).reset_index(drop=True)


def analyze(input_csv: Path, output_dir: Path, thresholds: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input predictions CSV does not exist: {input_csv}")
    if not thresholds:
        raise ValueError("Provide at least one threshold")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("Thresholds must be unique")

    source = pd.read_csv(input_csv)
    missing = REQUIRED - set(source.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")
    if set(source["split"].dropna().astype(str)) != {"validation"}:
        raise ValueError("Threshold selection must use validation data only")

    greedy = source[source["experiment"] == "baseline_greedy"].copy()
    beam = source[source["experiment"] == "fixed_beam5"].copy()
    if greedy.empty or beam.empty:
        raise ValueError("Input CSV must contain baseline_greedy and fixed_beam5 rows")
    _check_unique(greedy, "baseline_greedy")
    _check_unique(beam, "fixed_beam5")

    g = greedy[KEYS + ["split", "dataset", "reference", "prediction", "confidence_proxy",
                       "audio_duration_s", "asr_latency_s"]].rename(columns={
        "reference": "reference_greedy",
        "prediction": "prediction_greedy",
        "confidence_proxy": "confidence_proxy_greedy",
        "audio_duration_s": "audio_duration_s_greedy",
        "asr_latency_s": "asr_latency_s_greedy",
        "split": "split_greedy",
        "dataset": "dataset_greedy",
    })
    b = beam[KEYS + ["reference", "prediction", "asr_latency_s"]].rename(columns={
        "reference": "reference_beam",
        "prediction": "prediction_beam",
        "asr_latency_s": "asr_latency_s_beam",
    })
    paired = g.merge(b, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if not (paired["_merge"] == "both").all():
        counts = paired["_merge"].value_counts().to_dict()
        raise ValueError(f"Greedy and beam rows are not paired exactly: {counts}")
    if not (paired["reference_greedy"].fillna("").astype(str) == paired["reference_beam"].fillna("").astype(str)).all():
        raise ValueError("Reference transcripts differ between greedy and beam rows")
    if paired[["audio_duration_s_greedy", "asr_latency_s_greedy", "asr_latency_s_beam"]].isna().any().any():
        raise ValueError("Missing duration or latency values in baseline rows")

    # Preserve the deterministic greedy/beam rows exactly once in combined output.
    passthrough = source[source["experiment"].isin(["baseline_greedy", "fixed_beam5"])].copy()
    adaptive_frames: list[pd.DataFrame] = []
    threshold_summaries: list[pd.DataFrame] = []

    conf = pd.to_numeric(paired["confidence_proxy_greedy"], errors="coerce").to_numpy(dtype=float)
    greedy_latency = paired["asr_latency_s_greedy"].to_numpy(dtype=float)
    beam_latency = paired["asr_latency_s_beam"].to_numpy(dtype=float)
    duration = paired["audio_duration_s_greedy"].to_numpy(dtype=float)

    for threshold in thresholds:
        fallback = (~np.isfinite(conf)) | (conf < threshold)
        predictions = np.where(
            fallback,
            paired["prediction_beam"].fillna("").astype(str).to_numpy(),
            paired["prediction_greedy"].fillna("").astype(str).to_numpy(),
        )
        latency = np.where(fallback, greedy_latency + beam_latency, greedy_latency)
        rows: list[dict] = []
        for i, base_row in paired.reset_index(drop=True).iterrows():
            ref = str(base_row["reference_greedy"] or "")
            pred = str(predictions[i])
            dur = float(duration[i])
            rows.append({
                "sample_id": base_row["sample_id"],
                "dataset": base_row["dataset_greedy"],
                "split": "validation",
                "config": base_row["config"],
                "language": base_row["language"],
                "condition": base_row["condition"],
                "experiment": f"adaptive_threshold_{threshold:g}",
                "adaptive_threshold": float(threshold),
                "beam_size": 5,
                "fallback_triggered": bool(fallback[i]),
                "confidence_proxy": float(conf[i]) if np.isfinite(conf[i]) else np.nan,
                "reference": ref,
                "prediction": pred,
                "wer": word_error_rate(ref, pred),
                "cer": character_error_rate(ref, pred),
                "audio_duration_s": dur,
                "asr_latency_s": float(latency[i]),
                "rtf": float(latency[i] / dur) if dur > 0 else np.nan,
                "detected_language": None,
                "language_probability": np.nan,
            })
        adaptive_frames.append(pd.DataFrame(rows))

    combined = pd.concat([passthrough] + adaptive_frames, ignore_index=True, sort=False)
    summary = _summarize(combined)
    adaptive_summary = summary[summary["experiment"].str.startswith("adaptive_threshold_")].copy()
    # Equal macro weighting across language × acoustic-condition groups. We do not
    # choose on the held-out test. Latency/fallback are reported as secondary trade-offs.
    macro = adaptive_summary.groupby("experiment", as_index=False).agg(
        macro_wer=("corpus_wer", "mean"),
        macro_cer=("corpus_cer", "mean"),
        macro_mean_latency_s=("mean_latency_s", "mean"),
        macro_mean_rtf=("mean_rtf", "mean"),
        mean_fallback_rate=("fallback_rate", "mean"),
        strata_count=("condition", "count"),
    )
    macro["threshold"] = macro["experiment"].str.replace("adaptive_threshold_", "", regex=False).astype(float)
    macro = macro.sort_values(["macro_wer", "macro_mean_latency_s"]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "validation_threshold_predictions.csv"
    summary_path = output_dir / "validation_threshold_summary.csv"
    macro_path = output_dir / "validation_threshold_macro_summary.csv"
    combined.to_csv(pred_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    macro.to_csv(macro_path, index=False, encoding="utf-8-sig")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(input_csv),
        "thresholds": thresholds,
        "validation_only": True,
        "inference_performed": False,
        "comparison_method": "Offline adaptive selection from the same deterministic greedy and fixed beam-5 rows.",
        "selection_guidance": "Use macro WER across the four language-condition strata as the primary criterion; use CER, mean latency, RTF, and fallback rate to explain trade-offs. Freeze one threshold before evaluating test data.",
        "outputs": [str(pred_path), str(summary_path), str(macro_path)],
    }
    (output_dir / "validation_threshold_analysis.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary, macro


def main() -> None:
    args = parse_args()
    summary, macro = analyze(Path(args.input_csv), Path(args.output_dir), args.thresholds)
    print("\n=== PER-STRATUM SUMMARY ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== ADAPTIVE THRESHOLD MACRO SUMMARY (VALIDATION ONLY) ===")
    print(macro.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved analysis files to {args.output_dir}")
    print("No model inference was performed by this analysis step.")


if __name__ == "__main__":
    main()
