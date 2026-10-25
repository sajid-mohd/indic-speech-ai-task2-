import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speech_ai.metrics import character_error_rate, corpus_error_rates, normalize_text, word_error_rate
from speech_ai.audio import add_awgn
import numpy as np


def test_normalization_preserves_indic_letters():
    assert normalize_text("नमस्ते, दुनिया!") == "नमस्ते दुनिया"


def test_exact_match_metrics_are_zero():
    assert word_error_rate("नमस्ते दुनिया", "नमस्ते दुनिया") == 0
    assert character_error_rate("नमस्ते दुनिया", "नमस्ते दुनिया") == 0


def test_substitution_is_penalized():
    assert word_error_rate("राम घर गया", "राम स्कूल गया") > 0
    assert character_error_rate("hello", "hallo") > 0


def test_corpus_metrics_aggregate():
    metrics = corpus_error_rates(["a b", "c"], ["a x", "c"])
    assert metrics["wer"] == 1 / 3
    assert metrics["word_edits"] == 1


def test_noise_is_deterministic_and_changes_signal():
    audio = np.sin(np.linspace(0, 10, 16000)).astype(np.float32) * 0.2
    a = add_awgn(audio, 10, seed=7)
    b = add_awgn(audio, 10, seed=7)
    assert np.allclose(a, b)
    assert not np.allclose(a, audio)


def test_awgn_hits_requested_snr_without_loudness_pre_scaling():
    audio = (0.2 * np.sin(np.linspace(0, 20, 16000))).astype(np.float32)
    noisy = add_awgn(audio, snr_db=10.0, seed=123)
    noise = noisy - audio
    signal_power = float(np.mean(audio.astype(np.float64) ** 2))
    noise_power = float(np.mean(noise.astype(np.float64) ** 2))
    measured_snr_db = 10 * np.log10(signal_power / noise_power)
    assert abs(measured_snr_db - 10.0) < 0.2


def test_prepare_audio_downmixes_samples_by_channels():
    from speech_ai.audio import prepare_audio
    stereo = np.column_stack([
        np.ones(1000, dtype=np.float32) * 0.2,
        np.ones(1000, dtype=np.float32) * 0.4,
    ])
    mono = prepare_audio(stereo, sample_rate=16_000)
    assert mono.shape == (1000,)
    assert np.allclose(mono, 0.3, atol=1e-6)


def test_prepare_audio_rejects_invalid_sample_rate():
    from speech_ai.audio import prepare_audio
    import pytest
    with pytest.raises(ValueError):
        prepare_audio(np.ones(10, dtype=np.float32), sample_rate=0)


def test_prepare_audio_handles_empty_audio():
    from speech_ai.audio import prepare_audio
    empty = prepare_audio(np.array([], dtype=np.float32), sample_rate=16_000)
    assert empty.shape == (0,)
    assert empty.dtype == np.float32


def test_prepare_audio_resamples_to_target_rate():
    from speech_ai.audio import prepare_audio
    wave = np.sin(np.linspace(0, 20, 4410)).astype(np.float32)
    resampled = prepare_audio(wave, sample_rate=44_100, target_rate=16_000)
    assert resampled.dtype == np.float32
    assert abs(len(resampled) - 1600) <= 1
