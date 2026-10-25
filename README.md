# Indic Speech AI Research Assignment — Task 2

**Pipeline:** Speech Input → ASR → Language Model → Text Response

This repository implements and evaluates an Indic speech-processing pipeline using pretrained models and custom evaluation infrastructure. It includes Hindi and Kannada ASR benchmarks, an end-to-end Hindi speech-question-answering system, reproducibility checks, and an adaptive-decoding research extension.

## 1. What was evaluated

- **FLEURS ASR:** 50 Hindi and 50 Kannada read-speech utterances, each evaluated under clean audio and deterministic synthetic white noise at 10 dB SNR.
- **Decoder comparison:** Whisper small with greedy decoding (`beam_size=1`), fixed beam-5, and confidence-triggered adaptive beam-5.
- **Research extension:** A validation-only threshold sweep selects the adaptive threshold of `-0.3` under a maximum 50% fallback-rate constraint. The selected threshold is frozen for the held-out test evaluation.
- **End-to-end speech QA:** 100 Hindi synthetic-TTS examples using Indic-Transcribe Core ASR and Qwen2.5-1.5B-Instruct.
- **Reproducibility check:** Two runs of a 10-utterance subset produced identical predictions for all 60 paired comparisons across two audio conditions and three decoding methods.

The FLEURS benchmark contains 100 unique source utterances. Clean and noisy versions are additional experimental conditions, not additional independent source utterances.

## 2. Held-out ASR results

The table reports macro averages across four groups: Hindi-clean, Hindi-noisy, Kannada-clean, and Kannada-noisy.

| Method | Macro WER ↓ | Macro CER ↓ | Mean latency |
|---|---:|---:|---:|
| Greedy | 86.76% | 73.64% | 0.673 s |
| Adaptive, threshold -0.3 | 84.83% | 72.31% | 1.012 s |
| Fixed beam-5 | **83.02%** | **70.62%** | 0.925 s |

Fixed beam-5 achieved the lowest macro WER and CER among the three evaluated methods. Adaptive decoding improved over greedy decoding but did not outperform fixed beam-5 and incurred greater latency than greedy decoding.

Kannada performance was poor, with WER/CER near or above 100% in several evaluated conditions. These results indicate that the tested Whisper configuration is not adequate for reliable Kannada transcription in this evaluation.

The noisy condition uses synthetic additive white Gaussian noise at 10 dB SNR. It does not represent the full range of real-world noise, reverberation, microphone variation, or spontaneous speech.

## 3. End-to-end Hindi speech QA results

The final 100-example QA run used CPU inference with float32 computation.

| Metric | Result |
|---|---:|
| Question-transcript corpus WER | 17.73% |
| Question-transcript corpus CER | 7.98% |
| Mean ASR latency | 90.07 s |
| Mean LLM latency | 112.40 s |
| Mean pipeline latency | 202.47 s |
| Median pipeline latency | 156.10 s |
| Mean ASR real-time factor | 4.321 |
| Mean pipeline real-time factor | 14.502 |
| Outputs reaching the 192-token generation limit | 44/100 (44%) |

The pipeline was slower than real time in this CPU run. These measurements should not be interpreted as GPU performance estimates.

The paired question audio was generated using MMS Hindi TTS. This is a controlled synthetic-speech evaluation, not an evaluation on spontaneous speech from human speakers. The end-to-end QA experiment was Hindi-only; no Kannada end-to-end response-quality result is claimed.

## 4. Response-quality evaluation and limitations

An AI-assisted rubric-based review produced the following provisional estimates:

| Measure | AI-assisted estimate |
|---|---:|
| Correct or adequate responses | 12/100 (12%) |
| Responses flagged for a material unsupported or contradictory claim | 24/100 (24%) |
| Visibly incomplete responses | 44/100 (44%) |

The review criteria were:

- **Correctness:** The response sufficiently and factually answers the question.
- **Hallucination:** The response contains a material unsupported or contradictory factual claim.
- **Visible truncation:** The response itself visibly ends incomplete. Reaching the generation token limit alone is not sufficient evidence of truncation.

**These are AI-assisted estimates, not independently human-verified metrics.** The official human-review fields remain unfilled, and the counts must not be described as human-verified results.

A spot-check of 20 examples identified at least one clear question/reference-answer mismatch (`sample-0088`) and multiple prompts containing more than one task, where the reference may address only one component. The spot-check did not establish a general CSV row-order shift, but it revealed content-quality issues that can affect correctness and hallucination estimates.

Consequently, the response-quality figures should be treated as exploratory rather than validated aggregate rates. Further review should identify ambiguous or mismatched examples using a documented rule and independently verify response labels before stronger conclusions are drawn.

The raw per-example QA predictions and row-level review files are intentionally excluded from this public repository because they contain examples from a gated dataset. They remain in private storage. Do not redistribute them unless the applicable dataset terms permit it.

See `results/qa_run_summary.json`, `results/qa_ai_assisted_review_summary.json`, and `FINAL_RESEARCH_REPORT.md` for the recorded metrics and limitations.

## 5. Research extension: adaptive decoding

The research extension investigates whether a confidence-triggered decoding strategy can improve transcription quality while avoiding a beam-search pass for every utterance.

Three methods were compared:

1. **Greedy baseline:** Beam size 1.
2. **Fixed beam-5:** Beam size 5 for every utterance.
3. **Adaptive beam-5:** Run greedy decoding first and trigger beam-5 when mean segment log-probability falls below a selected threshold.

Mean segment log-probability is an uncalibrated confidence heuristic, not a calibrated probability. The threshold was selected on validation data under a maximum 50% fallback-rate constraint and then frozen before held-out test evaluation.

The test results show that adaptive decoding improves macro WER over greedy decoding but does not beat fixed beam-5. Its greater latency also means the experiment does not demonstrate a complete accuracy-latency win. This is an important negative result and is retained as such.

## 6. Repository setup and reproducibility

Repository: [`sajid-mohd/indic-speech-ai-task2-`](https://github.com/sajid-mohd/indic-speech-ai-task2-)

The Colab notebook clones the repository directly; a separately distributed ZIP is not required for the normal setup.

### Requirements files

- `requirements.txt` describes a general local development environment.
- `requirements-colab.txt` describes the newer Google Colab environment and preserves Colab's preinstalled CUDA-enabled PyTorch setup.

These requirement ranges are not a complete lockfile for the historical experiment. In particular, the recorded QA run used `datasets 4.8.5` and `transformers 5.20.2`, whereas the Colab requirements target Datasets 5.1 or later. A fresh run using the Colab requirements may therefore use a different dependency environment from the recorded QA run.

The recorded package versions are documented in `results/qa_run_summary.json`. Do not claim exact environment reproduction from the requirements files alone.

### Run locally

Use a suitable Python environment and install the general development dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the offline tests from the repository root:

```bash
python -m pytest -q
```

If imports fail when running tests from the repository root, set the source directory on the Python path first. In PowerShell, use `$env:PYTHONPATH = "src"`; in macOS/Linux shells, use `export PYTHONPATH=src`.

### Run in Google Colab

1. Open `Indic_Speech_AI_Research_Submission.ipynb` in Colab.
2. Run the notebook's first setup cell. It clones `https://github.com/sajid-mohd/indic-speech-ai-task2-.git` into `/content/speech_ai_assignment`. If the project folder already exists, the cell reuses it when the expected files are present.
3. Select a compatible GPU runtime for the FLEURS and Whisper experiments when GPU inference is required.
4. Authenticate with Hugging Face when prompted and ensure that access to gated models or datasets has been approved.
5. Run the notebook cells in order. Long inference cells are deliberately not auto-executed; avoid using **Run all** without reviewing the runtime and experiment instructions.
6. The recorded 100-example end-to-end QA experiment used CPU and float32. Reproducing it can take several hours.

Never embed Hugging Face or GitHub tokens in notebook source, logs, or committed files.

### Model revisions and access

The recorded experiment pins the Indic-Transcribe Core revision and the Hindi speech-QA dataset revision. The exact Qwen model revision used in the original QA run was not recorded. It must not be retroactively described as pinned without evidence.

Check the current model cards, access conditions, and license terms before rerunning or redistributing any model or dataset material. Gated example-level data and model weights are not included in this repository.

## 7. Repository layout

- `src/speech_ai/`: audio handling, metrics, and end-to-end pipeline.
- `scripts/`: ASR evaluation, end-to-end QA evaluation, adaptive-threshold analysis, demo, and manual-review scoring.
- `tests/`: metric, audio-handling, and offline threshold-analysis tests.
- `Indic_Speech_AI_Research_Submission.ipynb`: Colab experiment runbook.
- `FINAL_RESEARCH_REPORT.md`: methods, results, analysis, and limitations.
- `EXPERIMENT_MANIFEST.json`: model, dataset, and research-extension metadata.
- `results/`: aggregate CSV/JSON results, not raw row-level gated QA examples.
- `MODEL_SELECTION.md`: model selection and access notes.
- `AUDIT_REPORT.md`: recorded code, test, and environment audit.

## 8. Important limitations

- FLEURS contains read speech; the added noise is synthetic.
- End-to-end QA is Hindi-only and uses synthetic MMS Hindi TTS audio.
- Kannada ASR performance was inadequate under the tested Whisper configuration.
- Adaptive confidence is a heuristic and is not calibrated.
- QA response-quality figures are AI-assisted estimates and have not been independently human-verified.
- Some QA question/reference pairs are ambiguous or mismatched and need further review.
- Dependency requirements are not an exact lockfile for the recorded runs.
- The original Qwen revision was not recorded.
- Model and dataset access and license terms must be checked before redistribution.

## 9. Conclusion

This submission provides a reproducible code structure for an Indic speech-to-text-response pipeline, evaluation infrastructure for ASR WER/CER and latency/RTF, clean/noisy and Hindi/Kannada comparisons, and a validation-selected adaptive-decoding ablation evaluated on held-out data.

Fixed beam-5 produced the best aggregate ASR results among the evaluated decoding methods. Adaptive decoding improved over greedy but did not outperform fixed beam-5 and incurred additional latency. The end-to-end QA experiment demonstrates the complete pipeline on 100 synthetic Hindi speech examples, while its high CPU latency, truncated responses, and provisional response-quality estimates highlight important limitations.

The next steps for stronger evidence are to validate QA examples and response labels independently, align the documented environment with a tested run, and record exact model revisions in future experiments.
