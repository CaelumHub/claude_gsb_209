"""
analysis.py — High-level music-information-retrieval analyses.

Every analyser streams the audio from disk (never holding the whole file in
memory) so that arbitrarily large files can be analysed with a bounded memory
footprint.  The STFT is computed with a sliding carry buffer so frames are
seamless across chunk boundaries.

Analysers
---------
  * spectrogram      — downsampled dB magnitude spectrogram (for rendering)
  * spectral         — spectral centroid / rolloff / flatness / flux + RMS / ZCR
  * pitch            — fundamental-frequency contour (FFT autocorrelation)
  * beats            — tempo (BPM) and beat positions via onset-strength flux
  * waveform         — min/max envelope for efficient waveform rendering
"""

from __future__ import annotations

import math
from typing import Dict, Generator, List, Optional, Tuple

from . import audio_io, dsp


# --------------------------------------------------------------------------- #
# Streaming generators
# --------------------------------------------------------------------------- #

def stream_stft(path: str, nfft: int = 2048, hop: int = 512,
                win: str = "hann") -> Generator[Tuple[List[float], float], None, None]:
    """Yield (magnitude-spectrum, sample-rate) frames, streamed from disk.

    Only ``nfft//2+1`` positive bins are returned.  A carry buffer makes the
    frames seamless across the fixed-size read chunks."""
    w = dsp.window(win, nfft)
    carry: List[float] = [0.0] * (nfft // 2)  # centre-pad once at the start
    with audio_io.WavReader(path) as r:
        sr = r.sr
        while True:
            chunk = r.read_chunk(1 << 16)
            if chunk is None:
                break
            carry.extend(audio_io.to_mono(chunk))
            while len(carry) >= nfft:
                seg = carry[:nfft]
                frame = dsp.fft([seg[k] * w[k] for k in range(nfft)])
                bins = nfft // 2 + 1
                mag = [abs(frame[k]) for k in range(bins)]
                yield mag, sr
                carry = carry[hop:]


def stream_windows(path: str, win_len: int = 2048,
                   hop: int = 512) -> Generator[Tuple[List[float], float], None, None]:
    """Yield (windowed time-domain frame, sample-rate) from disk."""
    carry: List[float] = []
    with audio_io.WavReader(path) as r:
        sr = r.sr
        while True:
            chunk = r.read_chunk(1 << 16)
            if chunk is None:
                break
            carry.extend(audio_io.to_mono(chunk))
            while len(carry) >= win_len:
                yield list(carry[:win_len]), sr
                carry = carry[hop:]


# --------------------------------------------------------------------------- #
# Spectrogram
# --------------------------------------------------------------------------- #

def analyze_spectrogram(path: str, nfft: int = 2048, hop: int = 512,
                        win: str = "hann", max_time: int = 512,
                        max_freq: int = 256) -> Dict:
    """Downsampled dB spectrogram suitable for canvas rendering and JSON transfer.

    The full-resolution STFT is averaged on the fly into a ``max_time x max_freq``
    grid so the result is compact and memory stays bounded regardless of file size.
    """
    with audio_io.WavReader(path) as r:
        total_frames = r.nframes
        sr = r.sr
    est_frames = max(1, total_frames // hop)
    bucket = max(1, est_frames // max_time)
    n_bins = nfft // 2 + 1

    rows: List[List[float]] = []
    acc = [0.0] * n_bins
    count = 0

    def _flush() -> None:
        nonlocal acc, count
        if count == 0:
            return
        # Frequency downsampling by averaging groups of bins.
        group = max(1, n_bins // max_freq)
        row = []
        for b in range(0, n_bins, group):
            seg = acc[b:b + group]
            if seg:
                row.append(sum(seg) / (len(seg) * count))
        rows.append(dsp.to_db([row], floor=-100.0)[0])
        acc = [0.0] * n_bins
        count = 0

    for mag, _ in stream_stft(path, nfft, hop, win):
        for k in range(n_bins):
            acc[k] += mag[k]
        count += 1
        if count >= bucket:
            _flush()
    _flush()

    times = [i * bucket * hop / sr for i in range(len(rows))]
    freqs = [k * sr / nfft for k in range(max_freq)]
    # freqs represent band centres; recompute as the centre of each group
    group = max(1, n_bins // max_freq)
    freqs = [((b + group // 2) * sr / nfft) for b in range(0, n_bins, group)][:max_freq]

    return {
        "times": times,
        "freqs": freqs,
        "data": rows,
        "sr": sr,
        "nfft": nfft,
        "hop": hop,
        "window": win,
    }


# --------------------------------------------------------------------------- #
# Spectral features
# --------------------------------------------------------------------------- #

def analyze_spectral(path: str, nfft: int = 2048, hop: int = 512,
                     rolloff_pct: float = 0.85) -> Dict:
    """Time series of spectral features (centroid, rolloff, flatness, flux, rms, zcr)."""
    bins = nfft // 2 + 1
    freqs = dsp.rfft_freqs(nfft, 44100)  # placeholder; recomputed per frame sr
    sr = 44100

    centroids: List[float] = []
    rolloffs: List[float] = []
    flatness: List[float] = []
    flux: List[float] = []
    prev = None
    frame_idx = 0
    hop_actual = hop

    for mag, sr in stream_stft(path, nfft, hop):
        freqs = dsp.rfft_freqs(nfft, sr)
        centroids.append(dsp.spectral_centroid(mag, freqs))
        rolloffs.append(dsp.spectral_rolloff(mag, freqs, rolloff_pct))
        flatness.append(dsp.spectral_flatness(mag))
        flux.append(dsp.spectral_flux(mag, prev))
        prev = mag
        frame_idx += 1

    # RMS + ZCR from the time-domain windows.
    rms_series: List[float] = []
    zcr_series: List[float] = []
    for frame, _sr in stream_windows(path, nfft, hop):
        sr = _sr
        rms_series.append(dsp.rms(frame))
        zcr_series.append(dsp.zero_crossing_rate(frame))

    times = [i * hop / sr for i in range(frame_idx)]
    return {
        "sr": sr,
        "times": times,
        "centroid": centroids,
        "rolloff": rolloffs,
        "flatness": flatness,
        "flux": flux,
        "rms": rms_series,
        "zcr": zcr_series,
    }


# --------------------------------------------------------------------------- #
# Pitch
# --------------------------------------------------------------------------- #

def analyze_pitch(path: str, win_len: int = 2048, hop: int = 512,
                  fmin: float = 50.0, fmax: float = 2000.0) -> Dict:
    """Fundamental-frequency contour with note names and aggregate stats."""
    w = dsp.hann(win_len)
    times: List[float] = []
    f0s: List[float] = []
    sr = 44100
    idx = 0
    for frame, sr in stream_windows(path, win_len, hop):
        windowed = [frame[k] * w[k] for k in range(win_len)]
        f0 = dsp.pitch_autocorr(windowed, sr, fmin, fmax)
        f0s.append(f0 if f0 is not None else 0.0)
        times.append(idx * hop / sr)
        idx += 1

    voiced = [f for f in f0s if f > 0]
    stats = {
        "voiced_ratio": len(voiced) / len(f0s) if f0s else 0.0,
        "mean_f0": sum(voiced) / len(voiced) if voiced else 0.0,
        "min_f0": min(voiced) if voiced else 0.0,
        "max_f0": max(voiced) if voiced else 0.0,
    }
    notes = [dsp.note_display(f) for f in f0s]

    # Median pitch as the most probable sung/played note.
    if voiced:
        import statistics
        stats["median_f0"] = statistics.median(voiced)
        stats["median_note"] = dsp.note_display(statistics.median(voiced))
    else:
        stats["median_f0"] = 0.0
        stats["median_note"] = "--"

    return {
        "sr": sr,
        "times": times,
        "f0": f0s,
        "notes": notes,
        "stats": stats,
    }


# --------------------------------------------------------------------------- #
# Beats & tempo
# --------------------------------------------------------------------------- #

def analyze_beats(path: str, nfft: int = 2048, hop: int = 512,
                  min_bpm: float = 40.0, max_bpm: float = 240.0) -> Dict:
    """Tempo (BPM) and beat times from a spectral-flux onset envelope."""
    onset: List[float] = []
    prev = None
    sr = 44100
    for mag, sr in stream_stft(path, nfft, hop):
        onset.append(dsp.spectral_flux(mag, prev))
        prev = mag

    frame_rate = sr / hop
    # Normalise the onset envelope.
    mx = max(onset) if onset else 1.0
    if mx > 1e-9:
        onset = [v / mx for v in onset]

    tempo = dsp.estimate_tempo(onset, frame_rate, min_bpm, max_bpm)
    beats = dsp.detect_beats(onset, frame_rate, tempo)
    times = [i / frame_rate for i in range(len(onset))]

    # Onset peaks (for visualisation).
    peaks = dsp.local_maxima(dsp.smooth(onset, 5))

    return {
        "sr": sr,
        "tempo": round(tempo, 2),
        "times": times,
        "onset": onset,
        "onset_peaks": [times[p] for p in peaks],
        "beats": beats,
    }


# --------------------------------------------------------------------------- #
# Waveform envelope
# --------------------------------------------------------------------------- #

def waveform_envelope(path: str, points: int = 2000,
                      channel: int = 0) -> Dict:
    """Min/max envelope per bucket for fast, accurate waveform rendering."""
    with audio_io.WavReader(path) as r:
        sr = r.sr
        channels = r.channels
        nframes = r.nframes
        ch = min(channel, channels - 1)
        bucket = max(1, nframes // points)

        mins: List[float] = []
        maxs: List[float] = []
        cur_min = 1.0
        cur_max = -1.0
        cnt = 0
        while True:
            chunk = r.read_chunk(1 << 18)
            if chunk is None:
                break
            data = chunk[ch]
            for v in data:
                if v < cur_min:
                    cur_min = v
                if v > cur_max:
                    cur_max = v
                cnt += 1
                if cnt >= bucket:
                    mins.append(cur_min)
                    maxs.append(cur_max)
                    cur_min = 1.0
                    cur_max = -1.0
                    cnt = 0
        if cnt > 0:
            mins.append(cur_min)
            maxs.append(cur_max)

    return {
        "sr": sr,
        "channels": channels,
        "frames": nframes,
        "duration": nframes / sr if sr else 0,
        "points": len(maxs),
        "mins": mins,
        "maxs": maxs,
    }


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #

ANALYSERS = {
    "spectrogram": analyze_spectrogram,
    "spectral": analyze_spectral,
    "pitch": analyze_pitch,
    "beats": analyze_beats,
    "waveform": waveform_envelope,
}


def run(kind: str, path: str, **kwargs) -> Dict:
    fn = ANALYSERS.get(kind)
    if fn is None:
        raise ValueError(f"unknown analysis kind {kind!r}")
    return fn(path, **kwargs)
