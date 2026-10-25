"""Text normalization and reference-based ASR metrics for Indic scripts."""
from __future__ import annotations

import unicodedata
from typing import Iterable, Sequence


def normalize_text(text: str) -> str:
    """Normalize Unicode and punctuation without stripping Indic diacritics."""
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    chars: list[str] = []
    for ch in text:
        category = unicodedata.category(ch)
        # Treat punctuation and formatting/control characters as separators;
        # preserve letters, combining marks, and numerals from any script.
        if category.startswith("P"):
            chars.append(" ")
        elif category in {"Cf", "Cc", "Cs"}:
            # Drop invisible formatting/control marks such as zero-width joiners;
            # do not split the surrounding Indic grapheme cluster.
            continue
        else:
            chars.append(ch)
    return " ".join("".join(chars).split())


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Levenshtein distance using O(min(n,m)) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            substitution = previous[j - 1] + (ref_token != hyp_token)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Utterance WER. Returns 0.0 when both are empty, 1.0 for empty ref."""
    ref_tokens = normalize_text(reference).split()
    hyp_tokens = normalize_text(hypothesis).split()
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return _edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Utterance CER over Unicode characters, excluding whitespace."""
    ref_chars = list("".join(normalize_text(reference).split()))
    hyp_chars = list("".join(normalize_text(hypothesis).split()))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return _edit_distance(ref_chars, hyp_chars) / len(ref_chars)


def corpus_error_rates(
    references: Iterable[str], hypotheses: Iterable[str]
) -> dict[str, float]:
    """Corpus WER/CER, aggregating edit counts rather than averaging rows."""
    refs = list(references)
    hyps = list(hypotheses)
    if len(refs) != len(hyps):
        raise ValueError(f"Reference/prediction count mismatch: {len(refs)} != {len(hyps)}")

    word_errors = char_errors = total_words = total_chars = 0
    for ref, hyp in zip(refs, hyps):
        ref_words = normalize_text(ref).split()
        hyp_words = normalize_text(hyp).split()
        ref_chars = list("".join(ref_words))
        hyp_chars = list("".join(hyp_words))
        word_errors += _edit_distance(ref_words, hyp_words)
        char_errors += _edit_distance(ref_chars, hyp_chars)
        total_words += len(ref_words)
        total_chars += len(ref_chars)

    return {
        "wer": word_errors / total_words if total_words else 0.0,
        "cer": char_errors / total_chars if total_chars else 0.0,
        "word_edits": float(word_errors),
        "character_edits": float(char_errors),
        "reference_words": float(total_words),
        "reference_characters": float(total_chars),
    }
