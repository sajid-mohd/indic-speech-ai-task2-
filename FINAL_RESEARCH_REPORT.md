# Speech AI Research Intern — Task 2: Implementation and Evaluation

**Status:** Inference completed; aggregate results documented. Response-quality estimates are AI-assisted and are not independent human annotation.

## 1. Objective and system

The project implements and evaluates a speech-to-text-response system:

**Speech input → Indic-Transcribe ASR → Qwen2.5-1.5B-Instruct → textual response**

The source package also includes Whisper small for ASR benchmarking and a confidence-triggered adaptive decoding research extension. The code implements dataset handling, audio normalization/noise generation, WER/CER calculations, per-example timing/RTF, checkpointing, experiment metadata, threshold analysis, and review scoring.

## 2. Experimental design

### 2.1 FLEURS ASR and decoding ablation

The benchmark used `google/fleurs`, 50 Hindi (`hi_in`) and 50 Kannada (`kn_in`) utterances, with each source recording evaluated in clean form and with deterministic additive white Gaussian noise at 10 dB SNR. The known language was passed to Whisper to isolate decoding behavior. WER/CER are corpus-level within each language-condition group.

Three Whisper methods were compared:

- **Greedy baseline:** `beam_size=1`.
- **Fixed beam-5:** `beam_size=5` for every example.
- **Adaptive confidence-triggered beam-5:** perform greedy decoding and select the fixed-beam result when mean segment log-probability is below a threshold. This log-probability is a heuristic confidence proxy, not a calibrated probability. Adaptive latency includes the greedy pass plus beam pass when fallback is used.

Whisper inference was run with explicit `temperature=0.0` for the final deterministic inference pass. A reproducibility spot check compared two independent runs on 10 unique utterances, two audio conditions, and three decoding methods: 60/60 paired transcripts matched.

### 2.2 Threshold selection

The threshold was selected on the validation split only. The decision rule was: among tested thresholds with fallback rate ≤50%, choose the lowest macro WER, where macro WER equally weights Hindi-clean, Hindi-noisy, Kannada-clean, and Kannada-noisy groups. The selected threshold was **-0.3**, with validation macro WER 83.72%, macro CER 70.75%, fallback rate 47.5%, and mean estimated latency 0.953 s. This threshold was frozen before the test run.

### 2.3 End-to-end speech QA

A separate evaluation processed 100 Hindi speech-QA examples from the validation split of `Pastaaaaa2003/Hindi-speech-instruct`, dataset revision `3dc2ceb4f564d25f673d446fe8533e1ed6472a50`, using Indic-Transcribe Core (revision `e083ab7dfeeddcb5c0b4a2d7df53f53e7b0f1537`) and `Qwen/Qwen2.5-1.5B-Instruct`. The audio was generated with MMS Hindi TTS; this is a controlled synthetic-speech benchmark, not evaluation on spontaneous human speech. GPU access was unavailable for this final 100-example run, so it was completed on CPU using float32.

## 3. FLEURS validation results

The table below preserves the direct validation-run adaptive output at the runner's threshold of `-0.6`; the threshold sweep in Section 4 recomputes adaptive decisions offline from the same greedy and beam-5 hypotheses and selects `-0.3`. The reported held-out test uses the selected `-0.3` threshold.

| Language | Condition | Method | N | Corpus WER | Corpus CER | Mean latency (s) | Mean RTF | Fallback rate |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Hindi | Clean | Adaptive | 50 | 59.29% | 35.89% | 0.550 | 0.0512 | 8% |
| Hindi | Clean | Greedy | 50 | 59.02% | 34.08% | 0.485 | 0.0451 | — |
| Hindi | Clean | Fixed beam-5 | 50 | 53.51% | 30.08% | 0.689 | 0.0633 | — |
| Hindi | Noise 10 dB | Adaptive | 50 | 79.20% | 50.66% | 0.546 | 0.0509 | 8% |
| Hindi | Noise 10 dB | Greedy | 50 | 79.38% | 50.95% | 0.484 | 0.0449 | — |
| Hindi | Noise 10 dB | Fixed beam-5 | 50 | 68.53% | 45.65% | 0.707 | 0.0656 | — |
| Kannada | Clean | Adaptive | 50 | 100.00% | 99.00% | 0.787 | 0.0647 | 2% |
| Kannada | Clean | Greedy | 50 | 100.00% | 99.00% | 0.787 | 0.0646 | — |
| Kannada | Clean | Fixed beam-5 | 50 | 106.02% | 97.71% | 1.089 | 0.0883 | — |
| Kannada | Noise 10 dB | Adaptive | 50 | 100.13% | 98.51% | 0.784 | 0.0643 | 4% |
| Kannada | Noise 10 dB | Greedy | 50 | 100.64% | 98.65% | 0.775 | 0.0629 | — |
| Kannada | Noise 10 dB | Fixed beam-5 | 50 | 104.10% | 99.93% | 1.112 | 0.0891 | — |

## 4. Adaptive-threshold validation sweep

| Threshold | Macro WER | Macro CER | Fallback rate | Mean latency (s) | Mean RTF |
|---:|---:|---:|---:|---:|---:|
| -1.0 | 84.63% | 70.63% | 1.5% | 0.635 | 0.0548 |
| -0.8 | 84.77% | 70.81% | 2.0% | 0.641 | 0.0552 |
| -0.7 | 84.65% | 70.60% | 3.0% | 0.646 | 0.0561 |
| -0.6 | 84.65% | 71.01% | 5.5% | 0.667 | 0.0578 |
| -0.5 | 84.90% | 71.20% | 12.5% | 0.711 | 0.0624 |
| -0.4 | 84.52% | 71.15% | 29.0% | 0.818 | 0.0723 |
| **-0.3 (selected)** | **83.72%** | 70.75% | 47.5% | 0.953 | 0.0842 |
| -0.2 | 84.22% | 69.96% | 96.5% | 1.491 | 0.1279 |
| -0.1 | 83.04% | 68.34% | 99.5% | 1.526 | 0.1304 |
| 0.0 | 83.04% | 68.34% | 100.0% | 1.532 | 0.1310 |

Thresholds -0.1 and 0.0 effectively select beam search nearly every time; their scores approach the fixed-beam behavior and do not provide a meaningful selective-compute advantage.

### Selected threshold (-0.3), recomputed by language and condition

These values are reconstructed offline from the same saved greedy and beam-5 predictions; no second ASR inference pass is needed for threshold selection.

| Language | Condition | Corpus WER | Corpus CER | Fallback rate | Mean estimated latency (s) |
|---|---|---:|---:|---:|---:|
| Hindi | Clean | 55.20% | 32.47% | 86% | 1.035 |
| Hindi | Noise 10 dB | 78.67% | 52.36% | 86% | 1.070 |
| Kannada | Clean | 100.00% | 99.65% | 10% | 0.880 |
| Kannada | Noise 10 dB | 101.02% | 98.51% | 8% | 0.829 |

## 5. Held-out FLEURS test results

The threshold was frozen at -0.3 before evaluation on the test split. Each row contains 50 source utterances per language-condition group.

| Language | Condition | Method | N | Corpus WER | Corpus CER | Mean latency (s) | Mean RTF | Fallback rate |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Hindi | Clean | Adaptive | 50 | 59.07% | 31.64% | 1.167 | 0.1083 | 90% |
| Hindi | Clean | Greedy | 50 | 62.88% | 35.79% | 0.532 | 0.0490 | — |
| Hindi | Clean | Fixed beam-5 | 50 | 58.47% | 30.99% | 0.722 | 0.0660 | — |
| Hindi | Noise 10 dB | Adaptive | 50 | 78.56% | 50.99% | 1.174 | 0.1101 | 88% |
| Hindi | Noise 10 dB | Greedy | 50 | 82.46% | 52.21% | 0.546 | 0.0514 | — |
| Hindi | Noise 10 dB | Fixed beam-5 | 50 | 71.27% | 46.27% | 0.742 | 0.0676 | — |
| Kannada | Clean | Adaptive | 50 | 101.69% | 103.33% | 0.849 | 0.0717 | 4% |
| Kannada | Clean | Greedy | 50 | 101.69% | 103.35% | 0.803 | 0.0684 | — |
| Kannada | Clean | Fixed beam-5 | 50 | 101.04% | 102.33% | 1.118 | 0.0956 | — |
| Kannada | Noise 10 dB | Adaptive | 50 | 100.00% | 103.29% | 0.857 | 0.0720 | 4% |
| Kannada | Noise 10 dB | Greedy | 50 | 100.00% | 103.19% | 0.811 | 0.0689 | — |
| Kannada | Noise 10 dB | Fixed beam-5 | 50 | 101.30% | 102.90% | 1.117 | 0.0946 | — |

### Macro-average test summary

| Method | Macro WER | Macro CER | Mean latency (s) |
|---|---:|---:|---:|
| Greedy baseline | 86.76% | 73.64% | 0.673 |
| Adaptive, threshold -0.3 | 84.83% | 72.31% | 1.012 |
| Fixed beam-5 | **83.02%** | **70.62%** | 0.925 |

Fixed beam-5 had the lowest macro WER and CER. Adaptive decoding improved macro WER over greedy by approximately 1.93 percentage points, but did not beat fixed beam-5 and incurred higher mean latency. The fallback rate was strongly language-dependent: 88–90% for Hindi and 4% for Kannada. This suggests the chosen log-probability threshold did not provide consistent selective computation across languages.

WER can exceed 100% because insertions may make edit operations exceed the number of reference words. Kannada performance remained very poor in this Whisper ablation; these values indicate a major weakness, not acceptable production quality.

## 6. End-to-end QA results (100 examples)

| Metric | Result |
|---|---:|
| Question transcript corpus WER | 17.73% |
| Question transcript corpus CER | 7.98% |
| Mean ASR latency | 90.07 s |
| Mean LLM latency | 112.40 s |
| Mean pipeline latency | 202.47 s |
| Median pipeline latency | 156.10 s |
| Mean ASR RTF | 4.321 |
| Mean pipeline RTF | 14.502 |
| Responses reaching the 192-token generation cap | 44/100 (44%) |

The QA pipeline was slower than real time in this CPU run. These CPU measurements should not be directly compared with the five-example T4 pilot because the hardware/runtime differed and the pilot sample was much smaller.

### Response-quality review

The 100 responses were assigned provisional rubric-based labels by an AI-assisted reviewer using these definitions:

- **Correctness:** The answer addresses the question accurately and sufficiently.
- **Hallucination:** A material unsupported or contradictory factual claim is present.
- **Visible truncation:** The response itself visibly ends incomplete. Reaching the token cap alone is not treated as proof of truncation.

| AI-assisted estimate | Count | Rate |
|---|---:|---:|
| Correct/adequate | 12/100 | 12% |
| Material hallucination flag | 24/100 | 24% |
| Visibly incomplete | 44/100 | 44% |

These figures are **AI-assisted estimates, not independently human-verified metrics**. The official `human_correct_0_or_1`, `human_hallucination_0_or_1`, and `response_truncated_0_or_1` columns remain blank in the original inference artifact.

A 20-example spot-check identified at least one clear content-level question/reference-answer mismatch (`sample-0088`) and several prompts containing multiple embedded tasks, where the reference may address only one component (including `sample-0010` and `sample-0020`). These issues may make some correctness and hallucination judgments ambiguous.

The spot-check did **not** establish a general CSV row-order shift. The identified concern is the content and structure of some question/reference pairs, not proof that rows are systematically misaligned. The full 100 examples have not been independently human-verified.

Consequently, the AI-assisted response-quality counts should be treated as **exploratory estimates, not validated aggregate rates**. Ambiguous or mismatched examples should be reviewed using a documented rule, and stronger response-quality claims should be made only after human reviewers have filled the official review fields. A reviewer can use the rubric in `README.md` and `scripts/score_manual_review.py` to record annotations and calculate metrics.

## 7. Reproducibility and artifacts

- Fresh-run spot check: **60/60 paired transcripts identical** for 10 utterances × two conditions × three methods.
- Unit/offline tests: **12 passed** in the recorded source-package audit environment. A fresh run of the current checkout is still recommended before submission.
- The complete raw QA prediction CSV and annotated row-level review files are retained in private Drive storage and are not included in this shareable package because they contain example-level content from a gated dataset. Aggregate QA metrics and the AI-assisted review summary are included here.
- The notebook clears all cell outputs, contains no embedded Hugging Face token, and documents model/dataset access requirements.

## 8. Limitations

1. FLEURS consists of read speech; added white noise is synthetic and does not cover reverberation, real-world noise, spontaneous speech, or diverse microphones.
2. The end-to-end response dataset is Hindi-only with MMS Hindi TTS audio, so no Kannada end-to-end answer-quality claim is made.
3. The end-to-end CPU run is slow; performance on T4 GPU was not measured for the full 100-example run.
4. Response correctness and hallucination labels are AI-assisted and have not been independently human-verified. A 20-example spot-check identified a clear question/reference mismatch and multiple multi-task prompts with potentially incomplete reference answers. It did not establish a general CSV row-order shift. These data-quality issues mean the response-quality figures remain exploratory.
5. The adaptive confidence proxy is uncalibrated and behaved differently across the two languages.
6. The exact Qwen revision used in the completed QA run was not recorded. Model and dataset access/license terms must be reviewed before redistribution. Do not publish gated data or model weights.

## 9. Conclusion

The project implements a speech-to-text-response pipeline, custom evaluation code, two-language clean/noisy ASR evaluation, a controlled decoder ablation, threshold selection on validation data, and held-out test evaluation. The adaptive method improved over greedy on macro test WER but was inferior to fixed beam-5 and more costly in latency; the original hypothesis was therefore not supported as a complete accuracy-latency win. Kannada recognition quality in the Whisper ablation was inadequate.

End-to-end inference completed on 100 Hindi synthetic-TTS questions, with measured ASR/pipeline latency and provisional response-quality estimates. The 20-example review found content-level question/reference issues but did not establish a general row-order shift. Independent human annotation and further evaluation on unambiguous QA examples remain necessary before stronger correctness and hallucination claims can be made.
