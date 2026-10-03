"""
separation.py — Simulated source separation.

The task explicitly asks for *simulated* separation, so three complementary
techniques are provided, each self-implemented and streamable:

  1. **HPSS** (harmonic / percussive) — median-filter the magnitude
     spectrogram along the time axis (harmonic enhancement) and the frequency
     axis (percussive enhancement), derive soft masks, and re-synthesise two
     stems with the original phase via overlap-add ISTFT.
  2. **Vocal / accompaniment** (mid/side) — decompose a stereo mix into its
     centre (usually lead vocal) and side (usually accompaniment) components.
  3. **Bands** (bass / mid / treble) — split the spectrum with biquad filters.

Modes 2 and 3 stream in constant memory.  Mode 1 needs the spectrogram in
memory, so it processes a configurable leading window of the file (see
``max_seconds``) — a documented trade-off of the median-filter HPSS.
"""

from __future__ import annotations

from bisect import bisect_left, insort
from typing import Dict, List, Optional, Sequence, Tuple

from . import audio_io, dsp

SEPARATION_MODES = ["harmonic_percussive", "vocal_accompaniment", "bands"]


# --------------------------------------------------------------------------- #
# Sliding median filter
# --------------------------------------------------------------------------- #

def median_filter_1d(seq: Sequence[float], w: int) -> List[float]:
    """Sliding median of an odd window ``w`` (O(w) per step).

    The result has the same length as the input; the window edges are handled
    by replicating the first/last median values."""
    n = len(seq)
    if n == 0:
        return []
    if w >= n:
        w = n if n % 2 == 1 else n - 1
    if w < 1:
        return list(seq)
    half = w // 2
    win = sorted(seq[:w])
    out = [win[half]]
    for i in range(w, n):
        old = seq[i - w]
        idx = bisect_left(win, old)
        win.pop(idx)
        insort(win, seq[i])
        out.append(win[half])
    # Edge-pad to the original length (w is odd => 2*half == w-1).
    out = [out[0]] * half + out + [out[-1]] * half
    return out[:n]


def _median_time(mag: List[List[float]], w: int) -> List[List[float]]:
    """Median-filter each frequency bin across time (harmonic enhancement)."""
    T = len(mag)
    if T == 0:
        return []
    B = len(mag[0])
    # Transpose to per-bin sequences.
    cols = [[mag[t][k] for t in range(T)] for k in range(B)]
    filtered = [median_filter_1d(col, w) for col in cols]
    # Transpose back.
    return [[filtered[k][t] for k in range(B)] for t in range(T)]


def _median_freq(mag: List[List[float]], w: int) -> List[List[float]]:
    """Median-filter each frame across frequency (percussive enhancement)."""
    return [median_filter_1d(frame, w) for frame in mag]


# --------------------------------------------------------------------------- #
# Mode 1: harmonic / percussive (HPSS)
# --------------------------------------------------------------------------- #

def hpss_separate(path: str, out_harmonic: str, out_percussive: str,
                  max_seconds: float = 60.0, nfft: int = 2048, hop: int = 512,
                  win_harm: int = 17, win_perc: int = 17,
                  margin: float = 1.0) -> Dict:
    """Separate a file into harmonic and percussive stems."""
    with audio_io.WavReader(path) as r:
        sr = r.sr
        channels = r.channels
        total = r.nframes

    # Cap the processed length and downsample very high rates to bound work.
    cap_frames = int(max_seconds * sr)
    n = min(total, cap_frames)

    # Read the mono mix for the capped window.
    mono: List[float] = []
    with audio_io.WavReader(path) as r:
        remaining = n
        while remaining > 0:
            chunk = r.read_chunk(min(1 << 16, remaining))
            if chunk is None:
                break
            mono.extend(audio_io.to_mono(chunk))
            remaining -= len(chunk[0])

    frames = dsp.stft(mono, nfft, hop, "hann")
    if not frames:
        raise ValueError("signal too short for HPSS separation")
    bins = nfft // 2 + 1
    mag = [[abs(fr[k]) for k in range(bins)] for fr in frames]

    H = _median_time(mag, win_harm)
    P = _median_freq(mag, win_perc)

    # Soft masks.
    harm_frames: List[List[complex]] = []
    perc_frames: List[List[complex]] = []
    for t, fr in enumerate(frames):
        hmask = [0.0] * bins
        pmask = [0.0] * bins
        for k in range(bins):
            h = H[t][k] ** margin
            p = P[t][k] ** margin
            denom = h + p
            hmask[k] = h / denom if denom > 1e-9 else 0.0
            pmask[k] = p / denom if denom > 1e-9 else 0.0
        hfr = [0.0j] * nfft
        pfr = [0.0j] * nfft
        for k in range(bins):
            hfr[k] = fr[k] * hmask[k]
            pfr[k] = fr[k] * pmask[k]
            if 0 < k < nfft // 2:  # conjugate-symmetric negative bins
                hfr[nfft - k] = fr[nfft - k] * hmask[k]
                pfr[nfft - k] = fr[nfft - k] * pmask[k]
        harm_frames.append(hfr)
        perc_frames.append(pfr)

    harm_sig = dsp.istft(harm_frames, nfft, hop, "hann", length=len(mono) + nfft)
    perc_sig = dsp.istft(perc_frames, nfft, hop, "hann", length=len(mono) + nfft)
    start = nfft // 2
    harm_sig = harm_sig[start:start + len(mono)]
    perc_sig = perc_sig[start:start + len(mono)]

    # Normalise and de-click.
    harm_sig = dsp.fade_in_out(dsp.normalize(harm_sig), 0.01, sr)
    perc_sig = dsp.fade_in_out(dsp.normalize(perc_sig), 0.01, sr)

    audio_io.save(out_harmonic, audio_io.AudioData([harm_sig], sr))
    audio_io.save(out_percussive, audio_io.AudioData([perc_sig], sr))

    return {
        "mode": "harmonic_percussive",
        "sr": sr,
        "channels": channels,
        "processed_seconds": len(mono) / sr,
        "total_seconds": total / sr if sr else 0,
    }


# --------------------------------------------------------------------------- #
# Mode 2: vocal / accompaniment (mid/side)
# --------------------------------------------------------------------------- #

def mid_side_separate(path: str, out_vocal: str, out_accompaniment: str) -> Dict:
    """Split a stereo mix into centre (vocal) and sides (accompaniment)."""
    with audio_io.WavReader(path) as r:
        sr = r.sr
        channels = r.channels
        if channels < 2:
            raise ValueError("vocal/accompaniment separation requires a stereo file")

        with audio_io.WavWriter(out_vocal, sr, 2, 2) as wv, \
             audio_io.WavWriter(out_accompaniment, sr, 2, 2) as wa:
            for chunk in r.iter_chunks():
                L = chunk[0]
                R = chunk[1]
                n = len(L)
                mid = [(L[i] + R[i]) * 0.5 for i in range(n)]
                side = [(L[i] - R[i]) * 0.5 for i in range(n)]
                wv.write_chunk([mid, mid])          # vocal = centre, both channels
                wa.write_chunk([side, [-s for s in side]])  # accompaniment = sides
    return {"mode": "vocal_accompaniment", "sr": sr, "channels": channels}


# --------------------------------------------------------------------------- #
# Mode 3: bass / mid / treble bands
# --------------------------------------------------------------------------- #

def bands_separate(path: str, out_bass: str, out_mid: str, out_treble: str,
                   bass_hi: float = 200.0, mid_hi: float = 4000.0) -> Dict:
    """Split a file into bass / mid / treble frequency bands."""
    with audio_io.WavReader(path) as r:
        sr = r.sr
        channels = r.channels
        with audio_io.WavWriter(out_bass, sr, channels, 2) as wb, \
             audio_io.WavWriter(out_mid, sr, channels, 2) as wm, \
             audio_io.WavWriter(out_treble, sr, channels, 2) as wt:
            for chunk in r.iter_chunks():
                bass = [dsp.bandpass_split(c, sr, 20.0, bass_hi) for c in chunk]
                mid = [dsp.bandpass_split(c, sr, bass_hi, mid_hi) for c in chunk]
                treble = [dsp.bandpass_split(c, sr, mid_hi, min(sr / 2, 20000.0)) for c in chunk]
                wb.write_chunk(bass)
                wm.write_chunk(mid)
                wt.write_chunk(treble)
    return {"mode": "bands", "sr": sr, "channels": channels}


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #

def separate(path: str, mode: str, out_paths: List[str],
             max_seconds: float = 60.0) -> Dict:
    """Run a separation mode; ``out_paths`` holds the destination WAV paths."""
    if mode == "harmonic_percussive":
        return hpss_separate(path, out_paths[0], out_paths[1], max_seconds=max_seconds)
    if mode == "vocal_accompaniment":
        return mid_side_separate(path, out_paths[0], out_paths[1])
    if mode == "bands":
        return bands_separate(path, out_paths[0], out_paths[1], out_paths[2])
    raise ValueError(f"unknown separation mode {mode!r}")
