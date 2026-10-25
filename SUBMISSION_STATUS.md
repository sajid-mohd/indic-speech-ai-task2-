# Submission status

## Completed
- Speech → Indic ASR → Qwen → text-response inference pipeline implemented.
- FLEURS ASR benchmark: 50 Hindi + 50 Kannada clips; clean and 10 dB synthetic noise.
- Whisper greedy / fixed beam-5 / confidence-triggered adaptive ablation.
- Validation-only threshold sweep, threshold frozen at -0.3, held-out test evaluation.
- Reproducibility spot check: 60/60 paired output transcripts matched.
- 100-example Hindi synthetic-TTS end-to-end QA inference completed on CPU.
- Aggregate results, report, source code, tests and cleaned notebook are included.

## Transparent limitation
Response-quality counts are AI-assisted estimates (12 correct/adequate, 24 hallucination flags, 44 visible truncation flags). The official human review columns remain blank (0/100). Do not call these counts independently human-reviewed. If the assignment requires human-verified QA metrics, a human reviewer still must complete and score those labels.

## Public GitHub publication
The shareable package intentionally excludes row-level data from the gated QA dataset, raw prediction CSVs, local audio files, model caches and credentials. Aggregate result tables and review methodology are included.
