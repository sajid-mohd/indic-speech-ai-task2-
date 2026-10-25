#!/usr/bin/env python3
"""Aggregate human-reviewed response correctness and hallucination labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", nargs="?", default="results/qa_predictions_for_review.csv")
    parser.add_argument("--output-json", default="results/manual_response_metrics.json")
    args = parser.parse_args()
    df = pd.read_csv(args.input_csv, keep_default_na=False)
    required = {"human_correct_0_or_1", "human_hallucination_0_or_1"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing review columns: {sorted(missing)}")

    def binary_col(name: str) -> pd.Series:
        vals = df[name].astype(str).str.strip()
        invalid = vals[~vals.isin({"", "0", "1"})]
        if not invalid.empty:
            raise ValueError(f"{name} must contain only blank, 0, or 1; invalid examples: {invalid.head().tolist()}")
        return vals[vals.isin({"0", "1"})].astype(int)

    correct = binary_col("human_correct_0_or_1")
    hallucinated = binary_col("human_hallucination_0_or_1")
    correct_covered = int(correct.shape[0])
    hallucination_covered = int(hallucinated.shape[0])
    both_labeled = (
        df["human_correct_0_or_1"].astype(str).str.strip().isin({"0", "1"})
        & df["human_hallucination_0_or_1"].astype(str).str.strip().isin({"0", "1"})
    )
    paired_covered = int(both_labeled.sum())
    total = int(len(df))
    result = {
        "total_examples": int(len(df)),
        "correctness_labeled_n": correct_covered,
        "correctness_rate_on_labeled": float(correct.mean()) if correct_covered else None,
        "hallucination_labeled_n": hallucination_covered,
        "hallucination_rate_on_labeled": float(hallucinated.mean()) if hallucination_covered else None,
        "correctness_annotation_coverage": float(correct_covered / total) if total else 0.0,
        "hallucination_annotation_coverage": float(hallucination_covered / total) if total else 0.0,
        "paired_annotation_n": paired_covered,
        "annotation_coverage": float(paired_covered / total) if total else 0.0,
        "note": "Each rate uses its own explicitly labeled rows. Paired annotation coverage counts rows with both labels; blanks are not counted as zero.",
    }
    truncation_metrics = {}
    trunc_col = "response_truncated_0_or_1"
    if trunc_col in df.columns:
        truncated = binary_col(trunc_col)
        trunc_n = int(truncated.shape[0])
        truncation_metrics = {
            "truncation_labeled_n": trunc_n,
            "response_truncation_rate_on_labeled": float(truncated.mean()) if trunc_n else None,
            "truncation_annotation_coverage": float(trunc_n / total) if total else 0.0,
        }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.update(truncation_metrics)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
