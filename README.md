# Indic Speech AI Research Assignment — Task 2

**Pipeline:** speech input → ASR → Qwen2.5-1.5B-Instruct → textual response

This repository contains the project code, a cleared Colab notebook, aggregate experiment results, and the final report for a reproducible Hindi/Kannada ASR benchmark plus a Hindi speech-question answering pipeline.

## What was evaluated

- **FLEURS ASR:** 50 Hindi and 50 Kannada read-speech utterances; clean audio and deterministic synthetic white noise at 10 dB SNR.
- **Decoder comparison:** Whisper small greedy (`beam_size=1`), fixed beam-5, and confidence-triggered adaptive beam-5.
- **Research extension:** one deterministic Whisper validation pass; an offline threshold sweep selects `-0.3` using validation macro WER subject to a ≤50% fallback-rate constraint; the threshold is frozen for the held-out test comparison.
- **End-to-end QA:** 100 Hindi synthetic-TTS questions; Indic-Transcribe Core ASR plus Qwen2.5-1.5B-Instruct; final run ran on CPU because GPU access was unavailable at that point.
- **Reproducibility spot check:** 60/60 paired transcripts were identical across two fresh runs of a 10-utterance subset.

## Results snapshot

### FLEURS held-out test (macro average of four groups: Hindi/Kannada × clean/noisy)

| Method | Macro WER ↓ | Macro CER ↓ | Mean latency |
|---|---:|---:|---:|
| Greedy | 86.76% | 73.64% | 0.673 s |
| Adaptive, threshold -0.3 | 84.83% | 72.31% | 1.012 s |
| Fixed beam-5 | **83.02%** | **70.62%** | 0.925 s |

Fixed beam-5 had the lowest macro WER and CER. Adaptive decoding improved over greedy but did not beat fixed beam-5 and was slower on these test measurements.

### End-to-end Hindi QA (100 examples, CPU run)

| Metric | Result |
|---|---:|
| Question-transcript corpus WER | 17.73% |
| Question-transcript corpus CER | 7.98% |
| Mean ASR latency | 90.07 s |
| Mean LLM latency | 112.40 s |
| Mean pipeline latency | 202.47 s |
| Median pipeline latency | 156.10 s |
| Mean pipeline RTF | 14.502 |
| Outputs reaching the 192-token generation limit | 44/100 (44%) |

The QA run used CPU and was slower than real time. Its paired question audio is MMS Hindi TTS, not natural-speaker speech.

A rubric-based review produced provisional estimates: 12/100 correct/adequate, 24/100 with a material hallucination flag, and 44/100 visibly incomplete. These assessments are not independent human annotations. The official `human_*` review fields are not populated and no human-verified rates are claimed. See `results/qa_ai_assisted_review_summary.json` and `FINAL_RESEARCH_REPORT.md`.

## Run locally / in Colab

Python 3.10–3.12 is recommended for a fresh environment; the recorded Colab runtime was Python 3.13. Use the package requirements and notebook for reproducing experiments. Some models and the QA dataset require Hugging Face authentication and approved gated access.

1. Open `Indic_Speech_AI_Research_Submission.ipynb` in Colab.
2. For the FLEURS ASR and Whisper ablation cells, select a GPU runtime (T4 or compatible) and run the notebook cells in order. Inference cells can take a long time; do not use **Run all** casually.
3. Authenticate with Hugging Face when prompted. Do not put tokens in notebook source, logs, or Git.
4. The 100-example QA run was measured with `device=cpu`, `compute_type=float32`; it took several hours. Re-run only if you need to reproduce it.
5. Human response-quality metrics require actual labels in `human_correct_0_or_1`, `human_hallucination_0_or_1`, and `response_truncated_0_or_1`, then `scripts/score_manual_review.py`.

The notebook has been cleared of execution outputs for a clean code review. The aggregate result tables are in `results/`. Raw per-example QA predictions (which include examples from a gated dataset) are intentionally retained in private Drive storage and are **not** included in this public-shareable package. Do not publish them unless the dataset's access terms permit redistribution.

## Layout

- `src/speech_ai/`: audio handling, metrics, and end-to-end pipeline.
- `scripts/`: model evaluation, threshold analysis, demo, and review scoring.
- `tests/`: unit and offline-analysis tests.
- `Indic_Speech_AI_Research_Submission.ipynb`: cleared, reproducible Colab runbook.
- `FINAL_RESEARCH_REPORT.md`: complete method, results, analysis, and limitations.
- `results/`: aggregate CSV/JSON tables only; raw row-level outputs are excluded.
- `GITHUB_SUBMISSION.md`: safe GitHub publication workflow from Colab.

## Important limitations

- FLEURS is read speech; noise is synthetic, not recorded real-world noise.
- The end-to-end QA dataset is Hindi-only and synthetic TTS audio.
- Kannada WER/CER in the Whisper ablation is near or above 100%; ASR quality is inadequate in these conditions.
- Adaptive confidence is a heuristic, not a calibrated confidence probability.
- The QA labels are AI-assisted, not independent human annotations. Do not call them human-reviewed.
- Check current model and dataset terms before redistributing weights, gated data, or example-level predictions.
