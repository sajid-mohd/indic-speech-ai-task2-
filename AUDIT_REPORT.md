# Code and environment audit

Audit date: 2026-10-23

## Checks completed locally

- Python syntax/bytecode compilation passed for all scripts, source modules, and tests.
- Unit tests passed: 10 tests cover Indic-script text normalization, WER/CER and corpus aggregation, deterministic noise, requested-SNR behavior, stereo downmix, invalid sample rate, empty audio, and resampling.
- Notebook JSON/format validates (nbformat 4.5), with all Python code cells passing AST syntax checks after excluding Jupyter shell/magic lines.
- Mocked orchestration tests passed for the Indic-vs-Whisper comparison script and the three-way Whisper decoding ablation. These tests validate data flow, output-file creation and summary schemas only; mock outputs are not model results and were removed after testing.
- Human-review scoring smoke test passed with partially labeled rows; separate and paired annotation coverage are reported correctly.
- The release archive is checked with `zipfile.testzip()` before distribution.

## Current documentation / access checks

- Indic-Transcribe Core's live Hugging Face model card documents the `snapshot_download` + `IndicTranscribe.from_pretrained` + `asr(path, lang="hi")` path, says its one-pass audio API refuses recordings over 45 seconds, and shows a gated-access requirement. The model card labels the license `other`; review and comply with the current model/base-model terms before using it.
- The FLEURS dataset card remains available and shows an ASR dataset with audio and transcription columns and CC-BY-4.0 license metadata.
- The Hindi speech instruction dataset remains available, is gated, and its current card describes paired Hindi speech questions and text responses; it is based on MMS Hindi TTS, so it is not natural-speaker evidence.
- Current faster-whisper documentation states GPU inference needs CUDA 12 cuBLAS and cuDNN 9. The notebook now checks whether CTranslate2 sees the GPU and fails early with guidance if it does not.

## Not verified in this environment

The audit container has CPU-only PyTorch and does not have `faster-whisper`, `datasets`, `transformers`, or the gated Indic-Transcribe weights installed. Therefore, **actual checkpoint download, CUDA model loading, and real inference have not been completed here**. Those require a live Colab T4 runtime and approved Hugging Face access for the gated model/data. The notebook starts with a 3-example-per-language validation smoke test so any access/runtime incompatibility is caught before the 100-example runs.

## Remaining runtime-dependent risks

- Hugging Face gated model/dataset approval must be granted to the logged-in account.
- CTranslate2 must see the Colab GPU and its compatible CUDA libraries.
- If Qwen plus Indic-Transcribe exceed available VRAM in the end-to-end QA run, run Whisper baseline separately or reduce response-model size; report any such change.
- Response correctness and hallucination results require actual generated responses and human labels; no values have been prefilled.


## Release v4.1 follow-up audit (2026-10-24)

- Fixed a dataset/split assumption in the end-to-end Q&A evaluation: the current Hugging Face card for `Pastaaaaa2003/Hindi-speech-instruct` lists train and validation splits, not a test split. The script now defaults to validation, checks available split names, pins the dataset revision for the run, and records that revision in the run manifest/summary.
- Added atomic per-example Q&A checkpoints, a configuration manifest to refuse incompatible resumes, a configurable `--max-new-tokens` setting, token-budget diagnostics, and a human-reviewed truncation label/rate.
- Notebook order now completes validation threshold selection before the locked FLEURS test evaluation. It adds a five-example Q&A pilot ahead of the 100-example run.
- Verified locally: ZIP integrity passes; notebook JSON validates; Python scripts/source/tests parse; `PYTHONPATH=src pytest -q` returns **10 passed**. The archive excludes bytecode and pytest cache artifacts.
- Still not verified: live Hugging Face gated access, actual Colab T4 inference, model-generated metrics, full 100-example Q&A run, and Drive resume after a real runtime termination. No actual model scores are included or implied by these static checks.

## Relevant current sources

- FLEURS dataset card and CC-BY-4.0 metadata: https://huggingface.co/datasets/google/fleurs
- Hindi speech-Q&A dataset card (synthetic MMS Hindi TTS audio; current split description): https://huggingface.co/datasets/Pastaaaaa2003/Hindi-speech-instruct
- `faster-whisper` CUDA requirement (CUDA 12 cuBLAS and cuDNN 9): https://github.com/SYSTRAN/faster-whisper
- Google Colab FAQ on runtime deletion and resource limits: https://research.google.com/colaboratory/faq.html
- Optional Vaani-aligned candidate SraVaani 1.0: https://huggingface.co/ARTPARK-IISc/SraVaani-1.0


## Release v5.0 follow-up audit (2026-10-24)

- User-run reproducibility audit of the earlier validation CSVs found 200 paired greedy rows in each comparison, zero reference mismatches, but 90/200 greedy prediction mismatches between m04 and m06 and 93/200 between m04 and m08. This showed that the previous per-threshold runs did not provide a controlled paired ablation.
- Inspected the evaluation calls: `faster-whisper` was invoked without an explicit `temperature`. Its documented default is a schedule containing non-zero values; low-log-probability/compression cases may retry with non-zero-temperature sampling.
- v5 explicitly passes `temperature=0.0` in `evaluate_fleurs.py`, `evaluate_indic_transcribe.py`, and the end-to-end Whisper pipeline.
- Added `scripts/analyze_adaptive_thresholds.py`: runs one fixed validation inference pass, then computes adaptive predictions for −0.4, −0.6, and −0.8 offline from the same greedy and beam-5 rows. This means thresholds are compared on the same samples, references, baseline transcripts, and fixed-beam transcripts. No model inference is performed by the analyzer.
- Notebook order updated so the deterministic validation run and offline analysis precede the manual `FINAL_THRESHOLD` gate. The held-out test remains separate and is not used to select the threshold.
- Previous v4 validation scores are diagnostic/history only and should not be used as the final ablation table because baseline decoding was not held constant across runs.
- Static testing for v5 is being performed separately. A live Colab GPU run remains required before claiming completed runtime results.

## Relevant decoding documentation

- `faster-whisper` transcribe API source documents the default temperature schedule and fallback behavior: https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py


## Submission build audit — 2026-10-25

### Runtime evidence supplied by the Colab run

- FLEURS deterministic validation and held-out test CSVs and summaries were generated in Colab. The Whisper calls used explicit `temperature=0.0`; an identical-settings spot-check yielded 60/60 matching paired transcripts on 10 unique clips.
- The validation threshold was selected on validation only and frozen at `-0.3`; the reported test table uses that frozen threshold.
- The end-to-end Hindi speech-QA run completed on 100 unique examples on CPU because GPU access was unavailable. It recorded transcript corpus WER 17.73%, CER 7.98%, mean pipeline latency 202.47 s and mean pipeline RTF 14.502.
- An AI-assisted rubric pass produced provisional counts (12 correct/adequate, 24 hallucination flags, 44 visibly incomplete). Official human-review fields remain blank. These values are not described as independent human annotations.
- Aggregate metrics only are bundled in this shareable release; the raw row-level QA CSV is kept private because it contains examples from a gated dataset.

### Code hardening in this release

- Added an FLEURS run-configuration manifest guard so checkpoint resumes refuse mismatched args and do not silently blend settings. The measured prediction files come from the earlier code revision with the same inference settings; this guard affects resume integrity, not decoding behavior.
- Cleaned the submission notebook: cleared outputs/execution counts, corrected the frozen threshold and threshold sweep, updated QA configuration/results paths, and separated AI-assisted estimates from official human labels.
- Updated README from a starter-scaffold disclaimer to a result-oriented report and added a GitHub publication guide. Row-level gated QA predictions, audio, model caches, and credentials are ignored for public GitHub publication.

### Static checks

- `python -m pytest -q`: 12 passed.
- Notebook JSON and all code cells were validated for a clean, unexecuted submission format.
- ZIP integrity checked before release.
