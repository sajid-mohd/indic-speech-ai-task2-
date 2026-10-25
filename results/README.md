# Results included in this shareable package

This folder contains aggregate summary tables and JSON metadata only. The row-level QA prediction/review CSV and workbook are stored in private Drive because they contain prompt/reference/output text from a gated dataset. Do not publish row-level gated-dataset examples unless the dataset terms explicitly allow redistribution.

- `asr_validation_summary.csv`: the completed deterministic FLEURS validation run, per language/condition/method.
- `adaptive_threshold_sweep.csv`: offline threshold sweep on the same saved validation predictions.
- `asr_test_summary.csv`: held-out FLEURS test results at the frozen threshold (-0.3).
- `qa_run_summary.json`: 100-example Hindi QA inference metrics (CPU run).
- `qa_ai_assisted_review_summary.json`: concise rubric-based estimates; not independent human labels.
- `reproducibility_check.json`: 60/60 transcript agreement spot check.

All WER/CER values are fractions in CSV/JSON and are shown as percentages in the report.

Test-summary note: the bundled held-out test aggregate reports the latency fields captured in the final run summary. It does not include a median-latency field because that value was not retained in this release aggregate; a fresh evaluator run computes mean and median directly from per-example latencies.
