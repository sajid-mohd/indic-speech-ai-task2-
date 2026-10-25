#!/usr/bin/env python3
"""Run one local audio file through the end-to-end speech assistant."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from speech_ai.pipeline import SpeechAssistant  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path")
    parser.add_argument("--language", choices=["hi", "kn", "ta", "te", "bn", "mr", "ml", "gu", "pa", "en"], default="hi")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--decoder", choices=["greedy", "beam", "adaptive"], default="beam")
    parser.add_argument("--adaptive-threshold", type=float, default=-0.7)
    parser.add_argument("--asr-model-size", default="small")
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    assistant = SpeechAssistant(
        asr_model_size=args.asr_model_size,
        llm_model_id=args.llm_model,
        load_llm=True,
        device=args.device,
    )
    print(json.dumps(
        assistant.process_file(
            args.audio_path, args.language, args.beam_size,
            decoder=args.decoder, confidence_threshold=args.adaptive_threshold,
        ), ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
