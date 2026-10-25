# Model selection and access notes

## Models used in the completed experiments

- **End-to-end QA ASR:** `bodhan-ai/indic-transcribe-core`, recorded model revision `e083ab7dfeeddcb5c0b4a2d7df53f53e7b0f1537`. This model is gated; use an approved Hugging Face account. Review the exact model and base-model terms before redistribution.
- **ASR decoding ablation:** `Systran/faster-whisper-small` via faster-whisper/CTranslate2, with a known language and explicit `temperature=0.0`.
- **Response model:** `Qwen/Qwen2.5-1.5B-Instruct`. Its exact revision was not recorded in the completed run, so do not claim a pinned revision.
- **FLEURS:** `google/fleurs`, Hindi (`hi_in`) and Kannada (`kn_in`), 50 utterances per language, clean and synthetic 10 dB white-noise conditions.
- **End-to-end Q&A:** `Pastaaaaa2003/Hindi-speech-instruct`, revision `3dc2ceb4f564d25f673d446fe8533e1ed6472a50`; question audio is MMS Hindi TTS and dataset access is gated.

No SraVaani, Qwen3.5, or other optional candidate is claimed to have been evaluated in this release. Check upstream model/dataset cards and access terms at submission time. Do not publish gated example-level data or model weights.
