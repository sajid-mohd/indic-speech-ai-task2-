# Model Selection and Access Notes

## Models used in the completed experiments

- **End-to-end QA ASR:** `bodhan-ai/indic-transcribe-core`, recorded revision `e083ab7dfeeddcb5c0b4a2d7df53f53e7b0f1537`. Access requires accepting the model's access conditions. It is released under the Indic Open Model License v1.0 and is based on `nvidia/canary-1b-v2`, which has separate license terms. Review both before use or redistribution. See the [Indic-Transcribe Core model card](https://huggingface.co/bodhan-ai/indic-transcribe-core).

- **ASR decoding ablation:** `Systran/faster-whisper-small` through faster-whisper/CTranslate2. The experiments supplied the language explicitly and used `temperature=0.0` for deterministic decoding. Current GPU inference documentation specifies CUDA 12 cuBLAS and cuDNN 9. See the [faster-whisper documentation](https://github.com/SYSTRAN/faster-whisper).

- **Response model:** `Qwen/Qwen2.5-1.5B-Instruct`. The model card lists Apache-2.0. The exact revision used in the completed QA run was not recorded, so this release does not claim a pinned historical Qwen revision. See the [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).

- **FLEURS ASR benchmark:** `google/fleurs`, Hindi (`hi_in`) and Kannada (`kn_in`), 50 unique source utterances per language, evaluated under clean audio and synthetic white noise at 10 dB SNR. The dataset card lists CC BY 4.0. See the [FLEURS dataset card](https://huggingface.co/datasets/google/fleurs).

- **End-to-end speech QA dataset:** `Pastaaaaa2003/Hindi-speech-instruct`, recorded revision `3dc2ceb4f564d25f673d446fe8533e1ed6472a50`. The evaluated audio contains Hindi questions generated using MMS Hindi TTS; it is not natural-speaker evidence. The dataset card lists CC BY 4.0. Follow applicable access conditions and terms before redistributing examples. See the [Hindi speech-QA dataset card](https://huggingface.co/datasets/Pastaaaaa2003/Hindi-speech-instruct).

## Interpretation and access notes

- The completed end-to-end QA run used Indic-Transcribe Core plus Qwen2.5-1.5B-Instruct on CPU with float32 computation.
- The Whisper experiment is a separate ASR decoding ablation. Its results must not be substituted for the Indic-Transcribe QA results.
- Indic-Transcribe Core has a custom model license and separate base-model terms. Do not describe every component of this pipeline as having the same permissive license.
- The exact Qwen revision used in the completed run is unavailable in the saved run summary. Do not invent or retroactively claim a revision.
- No SraVaani, Qwen3.5, or other optional candidate is claimed to have been evaluated in this release.
- Do not publish gated example-level QA data, model weights, model caches, or authentication tokens. Check upstream model and dataset cards before redistribution because access terms and revisions can change.
