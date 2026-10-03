"""
chords.py — Chord recognition and annotation.

The pipeline mirrors a classic template-matching chord recogniser:

  1. Compute a 12-bin chromagram from the STFT (fold spectral energy into
     pitch classes).
  2. Compare each chroma frame against a bank of chord templates (major, minor,
     dim, aug, sus2, sus4, maj7, min7, dom7 — all 12 roots) by cosine
     similarity.
  3. Smooth the label sequence with a majority-vote median filter, then merge
     consecutive frames into labelled segments.

Annotations can then be stored back (the ``/api/analyze/<id>/chords/annotate``
endpoint) so a human can correct the automatic labels.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from . import analysis, dsp

_QUALITY_SUFFIX = {
    "maj": "",
    "min": "m",
    "dim": "dim",
    "aug": "aug",
    "sus2": "sus2",
    "sus4": "sus4",
    "maj7": "maj7",
    "min7": "m7",
    "dom7": "7",
}

_QUALITY_LONG = {
    "maj": "major",
    "min": "minor",
    "dim": "diminished",
    "aug": "augmented",
    "sus2": "suspended 2nd",
    "sus4": "suspended 4th",
    "maj7": "major 7th",
    "min7": "minor 7th",
    "dom7": "dominant 7th",
}

_TEMPLATES = {
    "maj": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    "min": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    "dim": [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    "aug": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    "sus2": [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "sus4": [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    "maj7": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    "min7": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    "dom7": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
}


def _rotate(vec: List[float], n: int) -> List[float]:
    return vec[n:] + vec[:n]


def _norm(vec: List[float]) -> List[float]:
    s = math.sqrt(sum(x * x for x in vec))
    return [x / s for x in vec] if s > 1e-12 else vec


def build_templates() -> List[Dict]:
    """Return the chord template bank (unit-normalised 12-D vectors)."""
    out = []
    for quality, vec in _TEMPLATES.items():
        for root in range(12):
            name = dsp.NOTE_NAMES[root]
            label = name + _QUALITY_SUFFIX[quality]
            out.append({
                "label": label,
                "root": name,
                "quality": quality,
                "quality_long": _QUALITY_LONG[quality],
                "vector": _norm(_rotate(vec, root)),
            })
    return out


_TEMPLATE_BANK = build_templates()


def _best_chord(chroma: List[float]) -> Tuple[str, float]:
    best_label = "N.C."
    best_sim = -1.0
    for t in _TEMPLATE_BANK:
        sim = sum(a * b for a, b in zip(chroma, t["vector"]))
        if sim > best_sim:
            best_sim = sim
            best_label = t["label"]
    return best_label, best_sim


def _mode_smooth(labels: List[str], width: int) -> List[str]:
    """Majority-vote smoothing of a categorical label sequence."""
    n = len(labels)
    if width < 1 or n == 0:
        return list(labels)
    half = width // 2
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        window = labels[lo:hi]
        out.append(max(set(window), key=window.count))
    return out


def recognize(path: str, nfft: int = 2048, hop: int = 512,
              smooth_width: int = 9) -> Dict:
    """Run the full chord-recognition pipeline on a file."""
    chroma: List[List[float]] = []
    sr = 44100
    for mag, sr in analysis.stream_stft(path, nfft, hop):
        chroma.append(_frame_chroma(mag, sr, nfft))

    if not chroma:
        return {"chroma": [], "times": [], "labels": [], "confidence": [], "segments": []}

    labels = [_best_chord(c)[0] for c in chroma]
    confidence = [_best_chord(c)[1] for c in chroma]
    labels = _mode_smooth(labels, smooth_width)

    times = [i * hop / sr for i in range(len(chroma))]
    segments = _merge_segments(times, labels, hop / sr)

    return {
        "sr": sr,
        "times": times,
        "chroma": chroma,
        "labels": labels,
        "confidence": confidence,
        "segments": segments,
    }


def _frame_chroma(mag: List[float], sr: float, nfft: int) -> List[float]:
    c = [0.0] * 12
    bins = len(mag)
    for k in range(1, bins):
        f = k * sr / nfft
        m = dsp.hz_to_midi(f)
        if m is None:
            continue
        c[int(round(m)) % 12] += mag[k]
    s = sum(c)
    return [v / s for v in c] if s > 1e-12 else c


def _merge_segments(times: List[float], labels: List[str], step: float) -> List[Dict]:
    """Merge consecutive equal labels into (start, end, label) segments."""
    segments = []
    if not labels:
        return segments
    start = times[0]
    prev = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != prev:
            segments.append({"start": round(start, 3), "end": round(times[i - 1] + step, 3),
                             "label": prev})
            start = times[i]
            prev = labels[i]
    segments.append({"start": round(start, 3), "end": round(times[-1] + step, 3),
                     "label": prev})
    return [s for s in segments if s["end"] - s["start"] >= step / 2]


CHORD_LABELS = [t["label"] for t in _TEMPLATE_BANK]
