from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_adaptive_thresholds.py"
SPEC = importlib.util.spec_from_file_location("analyze_adaptive_thresholds", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _source_row(sample_id, experiment, prediction, confidence, latency):
    return {
        "sample_id": sample_id,
        "dataset": "google/fleurs",
        "split": "validation",
        "config": "hi_in",
        "language": "hi",
        "condition": "clean",
        "experiment": experiment,
        "reference": "hello world",
        "prediction": prediction,
        "confidence_proxy": confidence,
        "audio_duration_s": 2.0,
        "asr_latency_s": latency,
        "rtf": latency / 2.0,
        "fallback_triggered": False,
    }


def test_threshold_analysis_uses_same_greedy_and_beam_rows(tmp_path):
    rows = [
        _source_row("hi_in-0001", "baseline_greedy", "hello word", -0.5, 0.5),
        _source_row("hi_in-0001", "fixed_beam5", "hello world", -0.5, 0.8),
        _source_row("hi_in-0001", "adaptive_confidence_beam5", "hello word", -0.5, 1.3),
        _source_row("hi_in-0002", "baseline_greedy", "hello", -0.9, 0.4),
        _source_row("hi_in-0002", "fixed_beam5", "hello world", -0.9, 0.7),
        _source_row("hi_in-0002", "adaptive_confidence_beam5", "hello world", -0.9, 1.1),
    ]
    input_csv = tmp_path / "predictions.csv"
    pd.DataFrame(rows).to_csv(input_csv, index=False)

    summary, macro = MODULE.analyze(input_csv, tmp_path / "out", [-0.4, -0.6, -0.8])

    adaptive = summary[summary["experiment"].str.startswith("adaptive_threshold_")]
    # -0.4 fallback for both items; -0.6/-0.8 fallback only for the lower-confidence item.
    rates = adaptive.groupby("experiment")["fallback_rate"].mean().to_dict()
    assert rates["adaptive_threshold_-0.4"] == 1.0
    assert rates["adaptive_threshold_-0.6"] == 0.5
    assert rates["adaptive_threshold_-0.8"] == 0.5
    assert set(macro["threshold"]) == {-0.4, -0.6, -0.8}
    assert (tmp_path / "out" / "validation_threshold_predictions.csv").exists()
    assert (tmp_path / "out" / "validation_threshold_summary.csv").exists()
    assert (tmp_path / "out" / "validation_threshold_macro_summary.csv").exists()


def test_threshold_analysis_rejects_mismatched_rows(tmp_path):
    rows = [
        _source_row("hi_in-0001", "baseline_greedy", "hello word", -0.5, 0.5),
        _source_row("hi_in-0002", "fixed_beam5", "hello world", -0.9, 0.7),
    ]
    input_csv = tmp_path / "bad_predictions.csv"
    pd.DataFrame(rows).to_csv(input_csv, index=False)
    try:
        MODULE.analyze(input_csv, tmp_path / "out", [-0.4])
    except ValueError as exc:
        assert "not paired exactly" in str(exc)
    else:
        raise AssertionError("Expected a paired-row validation error")
