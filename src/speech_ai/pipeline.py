"""Two-stage local pipeline: configurable Indic/Whisper ASR -> Qwen instruction LLM."""
from __future__ import annotations

import time
import tempfile
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .audio import prepare_audio

LANGUAGE_NAMES = {
    "hi": "Hindi",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "mr": "Marathi",
    "ml": "Malayalam",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "en": "English",
}

SYSTEM_PROMPTS = {
    "hi": (
        "आप एक सावधान हिंदी सहायक हैं। उपयोगकर्ता के प्रश्न का उत्तर हिंदी में दें। "
        "उत्तर संक्षिप्त और तथ्य-आधारित रखें। जानकारी निश्चित न हो तो साफ़ कहें कि आप निश्चित नहीं हैं; "
        "अनुमान लगाकर तथ्य न बनाएँ।"
    ),
    "kn": (
        "ನೀವು ಎಚ್ಚರಿಕೆಯಿಂದ ಉತ್ತರಿಸುವ ಕನ್ನಡ ಸಹಾಯಕರು. ಬಳಕೆದಾರರ ಪ್ರಶ್ನೆಗೆ ಕನ್ನಡದಲ್ಲೇ ಉತ್ತರಿಸಿ. "
        "ಉತ್ತರವನ್ನು ಸಂಕ್ಷಿಪ್ತವಾಗಿ ಮತ್ತು ವಾಸ್ತವಾಂಶ ಆಧಾರಿತವಾಗಿ ಇರಿಸಿ. ಖಚಿತ ಮಾಹಿತಿ ಇಲ್ಲದಿದ್ದರೆ ಅದನ್ನು ಸ್ಪಷ್ಟವಾಗಿ ಹೇಳಿ; "
        "ಊಹಿಸಿ ಮಾಹಿತಿಯನ್ನು ಸೃಷ್ಟಿಸಬೇಡಿ."
    ),
}


class SpeechAssistant:
    """Speech assistant with configurable ASR decoding for ablation studies."""

    def __init__(
        self,
        asr_model_size: str = "small",
        llm_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        load_llm: bool = True,
        device: str = "auto",
        compute_type: str = "auto",
        max_new_tokens: int = 160,
        asr_backend: str = "whisper",
        indic_model_id: str = "bodhan-ai/indic-transcribe-core",
    ) -> None:
        self.asr_backend = asr_backend
        self.indic_model_id = indic_model_id
        self.asr_model_revision = None
        self.llm_model_revision = None
        self.asr = None
        if asr_backend == "whisper":
            self.asr = WhisperModel(asr_model_size, device=device, compute_type=compute_type)
        elif asr_backend == "indic-transcribe":
            # Follow the current model card's supported wrapper API. Access to this
            # checkpoint is gated; the user must accept its terms and authenticate.
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(indic_model_id)
            model_dir_str = str(model_dir)
            self.asr_model_revision = Path(model_dir_str).name
            if model_dir_str not in sys.path:
                sys.path.insert(0, model_dir_str)
            from indic_transcribe import IndicTranscribe

            self.asr = IndicTranscribe.from_pretrained(model_dir_str)
        else:
            raise ValueError("asr_backend must be 'whisper' or 'indic-transcribe'")
        self.llm_model_id = llm_model_id
        self.max_new_tokens = max_new_tokens
        self.tokenizer = None
        self.llm = None
        if load_llm:
            self._load_llm()

    def _load_llm(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_id)
        if torch.cuda.is_available():
            dtype: Any = torch.float16
        else:
            dtype = torch.float32
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.llm_model_id,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.llm.eval()
        self.llm_model_revision = getattr(getattr(self.llm, "config", None), "_commit_hash", None)

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16_000,
        language: str = "hi",
        beam_size: int = 5,
    ) -> dict[str, Any]:
        x = prepare_audio(audio, sample_rate, 16_000)
        duration_s = len(x) / 16_000 if len(x) else 0.0
        start = time.perf_counter()
        if self.asr_backend == "indic-transcribe":
            # Indic-Transcribe's documented wrapper takes an audio file path.
            # Include the temporary WAV write in measured ASR time because the
            # end-to-end pipeline starts from an in-memory audio waveform here.
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_path = tmp.name
            try:
                start = time.perf_counter()
                sf.write(temp_path, x, 16_000, subtype="PCM_16")
                # The wrapper's one-pass API has a 45-second limit; use the
                # repository's chunked helper for longer user recordings.
                if duration_s > 45.0:
                    from long_form import transcribe_long
                    transcript = str(transcribe_long(self.asr, temp_path, lang=language)).strip()
                else:
                    transcript = str(self.asr(temp_path, lang=language)).strip()
                elapsed = time.perf_counter() - start
            finally:
                Path(temp_path).unlink(missing_ok=True)
            return {
                "transcript": transcript,
                "asr_latency_s": elapsed,
                "audio_duration_s": duration_s,
                "asr_rtf": elapsed / duration_s if duration_s > 0 else float("nan"),
                "detected_language": language,
                "language_probability": None,
                "avg_segment_logprob": float("nan"),
                "asr_backend": self.asr_backend,
            }

        segments, info = self.asr.transcribe(
            x,
            language=language,
            task="transcribe",
            beam_size=int(beam_size),
            temperature=0.0,  # deterministic decoding; see reproducibility audit
            vad_filter=True,
        )
        segment_rows = list(segments)  # force generator evaluation before timing ends
        elapsed = time.perf_counter() - start
        transcript = " ".join(seg.text.strip() for seg in segment_rows).strip()
        avg_logprob = (
            float(np.mean([seg.avg_logprob for seg in segment_rows]))
            if segment_rows
            else float("nan")
        )
        return {
            "transcript": transcript,
            "asr_latency_s": elapsed,
            "audio_duration_s": duration_s,
            "asr_rtf": elapsed / duration_s if duration_s > 0 else float("nan"),
            "detected_language": getattr(info, "language", language),
            "language_probability": getattr(info, "language_probability", None),
            "avg_segment_logprob": avg_logprob,
            "asr_backend": self.asr_backend,
        }

    def transcribe_adaptive(
        self,
        audio: np.ndarray,
        sample_rate: int = 16_000,
        language: str = "hi",
        confidence_threshold: float = -0.3,
        fallback_beam_size: int = 5,
    ) -> dict[str, Any]:
        """Run a greedy pass and only rerun with beam search when it looks uncertain.

        Whisper segment average log-probability is used as a simple uncertainty
        proxy. The threshold must be selected on a validation set and frozen
        before final test evaluation. This heuristic is not a calibrated probability.
        """
        if self.asr_backend != "whisper":
            raise ValueError("Adaptive confidence-triggered beam search is implemented for Whisper only")
        first_pass = self.transcribe(audio, sample_rate, language, beam_size=1)
        confidence_proxy = first_pass.get("avg_segment_logprob", float("nan"))
        use_fallback = (
            not np.isfinite(confidence_proxy) or confidence_proxy < confidence_threshold
        )
        if not use_fallback:
            return {
                **first_pass,
                "decoder": "adaptive_confidence_beam",
                "fallback_triggered": False,
                "confidence_threshold": confidence_threshold,
                "confidence_proxy": confidence_proxy,
            }
        second_pass = self.transcribe(
            audio, sample_rate, language, beam_size=fallback_beam_size
        )
        return {
            **second_pass,
            "asr_latency_s": first_pass["asr_latency_s"] + second_pass["asr_latency_s"],
            "asr_rtf": (
                first_pass["asr_latency_s"] + second_pass["asr_latency_s"]
            ) / second_pass["audio_duration_s"] if second_pass["audio_duration_s"] > 0 else float("nan"),
            "decoder": "adaptive_confidence_beam",
            "fallback_triggered": True,
            "confidence_threshold": confidence_threshold,
            "confidence_proxy": confidence_proxy,
        }

    def respond(self, transcript: str, language: str = "hi") -> dict[str, Any]:
        if self.llm is None or self.tokenizer is None:
            raise RuntimeError("LLM was not loaded. Initialize SpeechAssistant(load_llm=True).")
        if not transcript.strip():
            fallback = "कृपया फिर से बोलें; मैं प्रश्न स्पष्ट रूप से नहीं समझ पाया।" if language == "hi" else (
                "ದಯವಿಟ್ಟು ಮತ್ತೊಮ್ಮೆ ಹೇಳಿ; ಪ್ರಶ್ನೆ ಸ್ಪಷ್ಟವಾಗಿ ಅರ್ಥವಾಗಲಿಲ್ಲ." if language == "kn" else
                "Please repeat that; I could not understand the question clearly."
            )
            return {"response": fallback, "llm_latency_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0}

        language_name = LANGUAGE_NAMES.get(language, language)
        system_text = SYSTEM_PROMPTS.get(
            language,
            f"You are a careful assistant. Reply in {language_name}. Be concise and factual. If uncertain, say so rather than inventing facts.",
        )
        if language == "hi":
            user_text = (
                "नीचे उपयोगकर्ता की वाणी का ASR प्रतिलेख है। आशय समझकर हिंदी में उत्तर दें। "
                "यदि प्रतिलेख अस्पष्ट हो तो स्पष्टीकरण पूछें।\n\n"
                f"प्रतिलेख: {transcript}"
            )
        elif language == "kn":
            user_text = (
                "ಕೆಳಗೆ ಬಳಕೆದಾರರ ಮಾತಿನ ASR ಪಠ್ಯವಿದೆ. ಅದರ ಅರ್ಥವನ್ನು ಗ್ರಹಿಸಿ ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸಿ. "
                "ಪಠ್ಯ ಸ್ಪಷ್ಟವಾಗಿಲ್ಲದಿದ್ದರೆ ಸ್ಪಷ್ಟೀಕರಣ ಕೇಳಿ.\n\n"
                f"ಪ್ರತಿಲಿಪಿ: {transcript}"
            )
        else:
            user_text = (
                f"The following is the user's speech transcribed by ASR. Answer in {language_name}. "
                "If the transcript is ambiguous, ask a clarifying question.\n\n"
                f"Transcript: {transcript}"
            )
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        llm_start = time.perf_counter()
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer(prompt, return_tensors="pt")
        model_device = next(self.llm.parameters()).device
        model_inputs = {key: value.to(model_device) for key, value in model_inputs.items()}
        prompt_tokens = int(model_inputs["input_ids"].shape[-1])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        with torch.inference_mode():
            outputs = self.llm.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        generated = outputs[0][prompt_tokens:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        elapsed = time.perf_counter() - llm_start
        return {
            "response": response,
            "llm_latency_s": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": int(generated.shape[-1]),
        }

    def process_array(
        self,
        audio: np.ndarray,
        sample_rate: int = 16_000,
        language: str = "hi",
        beam_size: int = 5,
        decoder: str = "beam",
        confidence_threshold: float = -0.3,
    ) -> dict[str, Any]:
        total_start = time.perf_counter()
        if self.asr_backend == "indic-transcribe":
            if decoder == "adaptive":
                raise ValueError("Adaptive decoding is currently implemented only for Whisper")
            asr_result = self.transcribe(audio, sample_rate, language, beam_size=beam_size)
            asr_result["decoder"] = "indic_transcribe_default"
            asr_result["fallback_triggered"] = False
        elif decoder == "adaptive":
            asr_result = self.transcribe_adaptive(
                audio, sample_rate, language,
                confidence_threshold=confidence_threshold,
                fallback_beam_size=beam_size,
            )
        elif decoder == "greedy":
            asr_result = self.transcribe(audio, sample_rate, language, beam_size=1)
            asr_result["decoder"] = "greedy"
            asr_result["fallback_triggered"] = False
        elif decoder == "beam":
            asr_result = self.transcribe(audio, sample_rate, language, beam_size=beam_size)
            asr_result["decoder"] = f"beam_{beam_size}"
            asr_result["fallback_triggered"] = False
        else:
            raise ValueError("decoder must be one of: greedy, beam, adaptive")
        llm_result = self.respond(asr_result["transcript"], language)
        total_elapsed = time.perf_counter() - total_start
        return {
            **asr_result,
            **llm_result,
            "pipeline_latency_s": total_elapsed,
            "pipeline_rtf": total_elapsed / asr_result["audio_duration_s"]
            if asr_result["audio_duration_s"] > 0 else float("nan"),
            "language": language,
            # Indic-Transcribe uses its own model-default decoding; do not report a Whisper beam size for it.
            "beam_size": beam_size if self.asr_backend == "whisper" else None,
            "decoder": asr_result.get("decoder", decoder),
            "asr_backend": self.asr_backend,
            "fallback_triggered": asr_result.get("fallback_triggered", False),
        }

    def process_file(
        self, audio_path: str, language: str = "hi", beam_size: int = 5,
        decoder: str = "beam", confidence_threshold: float = -0.3,
    ) -> dict[str, Any]:
        import soundfile as sf
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        return self.process_array(
            audio, sample_rate, language, beam_size,
            decoder=decoder, confidence_threshold=confidence_threshold,
        )
