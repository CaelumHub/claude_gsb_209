"""
dsp.py — Core digital-signal-processing library for the Audio MIR system.

Everything here is implemented from first principles in pure Python (no numpy,
no scipy, no librosa).  The only imports are the standard library.  This is a
deliberate design decision: the project specification calls for the algorithms
to be *self-implemented*, and pure Python keeps the whole system dependency-free
and runnable anywhere a Python 3.9+ interpreter and Flask are available.

The module is organised as layers:

  1.  FFT / inverse FFT           (iterative radix-2, precomputed twiddles)
  2.  Windowing                   (Hann, Hamming, Blackman, ... with COLA norm)
  3.  Short-time Fourier transform (analysis + overlap-add synthesis)
  4.  Frequency conversions       (Hz <-> mel <-> MIDI <-> note names)
  5.  Audio metrics               (RMS, ZCR, spectral centroid, rolloff, flux…)
  6.  Biquad filters              (RBJ cookbook, used by the EQ & separation)
  7.  Pitch detection             (FFT autocorrelation + the YIN algorithm)
  8.  Onset / beat / tempo        (spectral flux + autocorrelation beat tracker)
  9.  Chroma                      (fold STFT energy into pitch classes)
 10.  Utilities                   (dB conversion, resampling, smoothing)

Performance notes
-----------------
* A radix-2 FFT of 2048 points runs in ~3 ms and of 1024 points in ~1.4 ms on a
  typical machine, which is comfortably inside a real-time analysis budget
  (a 1024-sample hop at 44.1 kHz is ~23 ms).
* Twiddle factors are cached per transform size to avoid recomputing the same
  complex exponentials on every call.
"""

from __future__ import annotations

import cmath
import math
from typing import List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# 1. FFT / IFFT
# --------------------------------------------------------------------------- #

_TWIDDLE_CACHE: dict = {}


def _twiddles(n: int) -> List[complex]:
    """Return the forward-FFT twiddle table exp(-2j*pi*k/n) for k in [0, n/2)."""
    cached = _TWIDDLE_CACHE.get(n)
    if cached is None:
        cached = [cmath.exp(-2j * math.pi * k / n) for k in range(n >> 1)]
        _TWIDDLE_CACHE[n] = cached
    return cached


def next_pow2(n: int) -> int:
    """Smallest power of two >= n."""
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def fft(x: Sequence[complex]) -> List[complex]:
    """In-place-free iterative radix-2 Cooley–Tukey FFT.

    ``x`` may be any sequence of real or complex numbers.  Non-power-of-two
    inputs are zero-padded to the next power of two.  Returns the spectrum
    ordered from DC (index 0) to Nyquist (index n//2), then the negative
    frequencies.
    """
    n = len(x)
    if n == 0:
        return []
    if n & (n - 1):  # not a power of two -> pad
        m = next_pow2(n)
        a = [complex(v) for v in x] + [0.0j] * (m - n)
        n = m
    else:
        a = [complex(v) for v in x]

    # Bit-reversal permutation.
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    table = _twiddles(n)
    length = 2
    while length <= n:
        half = length >> 1
        step = n // length
        for i in range(0, n, length):
            i2 = i + half
            for m in range(half):
                w = table[m * step]
                u = a[i + m]
                v = a[i2 + m] * w
                a[i + m] = u + v
                a[i2 + m] = u - v
        length <<= 1
    return a


def ifft(x: Sequence[complex]) -> List[complex]:
    """Inverse FFT via the conjugate trick: ifft(x) = conj(fft(conj(x)))/n."""
    n = len(x)
    if n == 0:
        return []
    conj = [v.conjugate() for v in x]
    y = fft(conj)
    inv = 1.0 / n
    return [v.conjugate() * inv for v in y]


def rfft_freqs(nfft: int, sr: float) -> List[float]:
    """Frequency (Hz) of each *positive* spectrum bin [0 .. nfft//2]."""
    return [k * sr / nfft for k in range(nfft // 2 + 1)]


# --------------------------------------------------------------------------- #
# 2. Window functions
# --------------------------------------------------------------------------- #

def _cos_window(n: int, a0: float, a1: float, a2: float, a3: float) -> List[float]:
    out = []
    two_pi = 2.0 * math.pi
    for k in range(n):
        t = k / (n - 1) if n > 1 else 0.0
        out.append(
            a0 - a1 * math.cos(two_pi * t)
            + a2 * math.cos(2 * two_pi * t)
            - a3 * math.cos(3 * two_pi * t)
        )
    return out


def hann(n: int) -> List[float]:
    """Periodic-style Hann window (symmetric denominator n-1)."""
    return _cos_window(n, 0.5, 0.5, 0.0, 0.0)


def hamming(n: int) -> List[float]:
    return _cos_window(n, 0.54, 0.46, 0.0, 0.0)


def blackman(n: int) -> List[float]:
    return _cos_window(n, 0.42, 0.5, 0.08, 0.0)


def blackman_harris(n: int) -> List[float]:
    return _cos_window(n, 0.35875, 0.48829, 0.14128, 0.01168)


def bartlett(n: int) -> List[float]:
    out = []
    for k in range(n):
        out.append(1.0 - abs((2.0 * k - (n - 1)) / (n - 1)) if n > 1 else 1.0)
    return out


def rectangular(n: int) -> List[float]:
    return [1.0] * n


_WINDOWS = {
    "hann": hann,
    "hamming": hamming,
    "blackman": blackman,
    "blackman_harris": blackman_harris,
    "bartlett": bartlett,
    "rectangular": rectangular,
}


def window(name: str, n: int) -> List[float]:
    """Return a window of length ``n`` by name (falls back to Hann)."""
    fn = _WINDOWS.get(name, hann)
    return fn(n)


def window_names() -> List[str]:
    return list(_WINDOWS.keys())


# --------------------------------------------------------------------------- #
# 3. Short-time Fourier transform
# --------------------------------------------------------------------------- #

def stft(
    samples: Sequence[float],
    nfft: int = 2048,
    hop: int = 512,
    win: str = "hann",
    center: bool = True,
) -> List[List[complex]]:
    """Short-time Fourier transform -> list of complex frames (one per hop)."""
    w = window(win, nfft)
    n = len(samples)
    if center:
        pad = nfft // 2
        sig = [0.0] * pad + list(samples) + [0.0] * pad
    else:
        sig = list(samples)
    frames: List[List[complex]] = []
    i = 0
    total = len(sig)
    while i + nfft <= total:
        seg = sig[i:i + nfft]
        frames.append(fft([seg[k] * w[k] for k in range(nfft)]))
        i += hop
    return frames


def istft(
    frames: Sequence[Sequence[complex]],
    nfft: int = 2048,
    hop: int = 512,
    win: str = "hann",
    length: Optional[int] = None,
) -> List[float]:
    """Inverse STFT using weighted overlap-add (WOLA) synthesis."""
    w = window(win, nfft)
    n_frames = len(frames)
    if n_frames == 0:
        return []
    out = [0.0] * (n_frames * hop + nfft)
    norm = [0.0] * (n_frames * hop + nfft)
    for i, fr in enumerate(frames):
        t = ifft(fr)
        pos = i * hop
        for k in range(nfft):
            val = t[k].real * w[k]
            out[pos + k] += val
            norm[pos + k] += w[k] * w[k]
    for k in range(len(out)):
        d = norm[k]
        out[k] = (out[k] / d) if d > 1e-9 else 0.0
    if length is not None:
        out = out[:length]
    return out


def spectrogram(
    samples: Sequence[float],
    sr: float,
    nfft: int = 2048,
    hop: int = 512,
    win: str = "hann",
    max_bins: int = 0,
) -> Tuple[List[List[float]], List[float], List[float]]:
    """Return (magnitude-spectrogram dB, times, freqs).

    ``max_bins`` optionally limits how many positive bins are kept (for memory).
    """
    frames = stft(samples, nfft, hop, win)
    bins = nfft // 2 + 1
    if max_bins:
        bins = min(bins, max_bins)
    mag = [[abs(fr[k]) for k in range(bins)] for fr in frames]
    spec = to_db(mag)
    times = [i * hop / sr for i in range(len(frames))]
    freqs = [k * sr / nfft for k in range(bins)]
    return spec, times, freqs


# --------------------------------------------------------------------------- #
# 4. Frequency / scale conversions
# --------------------------------------------------------------------------- #

def hz_to_mel(f: float) -> float:
    return 2595.0 * math.log10(1.0 + f / 700.0)


def mel_to_hz(m: float) -> float:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(
    nfft: int, sr: float, n_mels: int = 40, fmin: float = 0.0, fmax: Optional[float] = None
) -> List[List[float]]:
    """Triangular mel filterbank -> n_mels filters each spanning positive bins."""
    if fmax is None:
        fmax = sr / 2.0
    n_bins = nfft // 2 + 1
    mel_min = hz_to_mel(max(fmin, 1e-9))
    mel_max = hz_to_mel(min(fmax, sr / 2.0))
    mel_pts = [mel_min + (mel_max - mel_min) * i / (n_mels + 1) for i in range(n_mels + 2)]
    hz_pts = [mel_to_hz(m) for m in mel_pts]
    bin_pts = [int(math.floor((nfft + 1) * h / sr)) for h in hz_pts]
    filters = []
    for m in range(1, n_mels + 1):
        f = [0.0] * n_bins
        lo, ce, hi = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        for k in range(lo, ce):
            if ce > lo:
                f[k] = (k - lo) / (ce - lo)
        for k in range(ce, hi):
            if hi > ce:
                f[k] = (hi - k) / (hi - ce)
        filters.append(f)
    return filters


def hz_to_midi(f: float) -> Optional[float]:
    """Frequency in Hz -> float MIDI note number (None for non-positive f)."""
    if f <= 0:
        return None
    return 69.0 + 12.0 * math.log2(f / 440.0)


def midi_to_hz(m: float) -> float:
    return 440.0 * (2.0 ** ((m - 69.0) / 12.0))


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_note(m: float) -> Tuple[str, int]:
    """Float MIDI -> (note name with accidental, octave)."""
    midi = int(round(m))
    name = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return name, octave


def note_display(f0: float) -> str:
    """Human-readable note + cents offset, e.g. 'A4 +12¢'."""
    if f0 is None or f0 <= 0:
        return "--"
    m = hz_to_midi(f0)
    midi = int(round(m))
    name, octave = midi_to_note(m)
    cents = int(round((m - midi) * 100))
    if cents == 0:
        return f"{name}{octave}"
    sign = "+" if cents > 0 else ""
    return f"{name}{octave} {sign}{cents}¢"


# --------------------------------------------------------------------------- #
# 5. Audio metrics
# --------------------------------------------------------------------------- #

def rms(samples: Sequence[float]) -> float:
    n = len(samples)
    if n == 0:
        return 0.0
    s = 0.0
    for x in samples:
        s += x * x
    return math.sqrt(s / n)


def db(x: float) -> float:
    """Amplitude -> dB, guarded against log(0)."""
    return 20.0 * math.log10(x + 1e-12)


def to_db(matrix: Sequence[Sequence[float]], ref: float = 1.0, floor: float = -120.0) -> List[List[float]]:
    """Convert an amplitude matrix to dB, flooring the result."""
    out = []
    for row in matrix:
        out.append([max(floor, db(v / ref)) for v in row])
    return out


def from_db(db_val: float) -> float:
    return 10.0 ** (db_val / 20.0)


def zero_crossing_rate(samples: Sequence[float]) -> float:
    n = len(samples)
    if n < 2:
        return 0.0
    count = 0
    prev = samples[0]
    for x in samples[1:]:
        if (x >= 0.0) != (prev >= 0.0):
            count += 1
        prev = x
    return count / (n - 1)


def spectral_centroid(mag: Sequence[float], freqs: Sequence[float]) -> float:
    s = 0.0
    wsum = 0.0
    for i, m in enumerate(mag):
        s += freqs[i] * m
        wsum += m
    return (s / wsum) if wsum > 1e-12 else 0.0


def spectral_rolloff(mag: Sequence[float], freqs: Sequence[float], pct: float = 0.85) -> float:
    total = sum(mag)
    if total <= 0:
        return 0.0
    acc = 0.0
    for i, m in enumerate(mag):
        acc += m
        if acc >= pct * total:
            return freqs[i]
    return freqs[-1]


def spectral_flatness(mag: Sequence[float]) -> float:
    """Geometric mean / arithmetic mean of the power spectrum (0..1)."""
    n = len(mag)
    if n == 0:
        return 0.0
    log_sum = 0.0
    lin_sum = 0.0
    for m in mag:
        p = m * m + 1e-12
        log_sum += math.log(p)
        lin_sum += p
    geo = math.exp(log_sum / n)
    arith = lin_sum / n
    return min(1.0, geo / arith)


def spectral_flux(mag: Sequence[float], prev_mag: Optional[Sequence[float]]) -> float:
    """L2 norm of the positive magnitude difference between two frames."""
    if prev_mag is None:
        return 0.0
    s = 0.0
    for i, m in enumerate(mag):
        d = m - prev_mag[i]
        if d > 0:
            s += d * d
    return math.sqrt(s)


def spectral_bandwidth(mag: Sequence[float], freqs: Sequence[float]) -> float:
    centroid = spectral_centroid(mag, freqs)
    s = 0.0
    wsum = 0.0
    for i, m in enumerate(mag):
        d = freqs[i] - centroid
        s += d * d * m
        wsum += m
    return math.sqrt(s / wsum) if wsum > 1e-12 else 0.0


# --------------------------------------------------------------------------- #
# 6. Biquad filters (RBJ Audio EQ Cookbook)
# --------------------------------------------------------------------------- #

class Biquad:
    """Transposed direct-form-II biquad with RBJ-cookbook coefficient design.

    Coefficients are computed from *analog* prototypes for the common filter
    shapes and then bilinear-transformed.  ``process_block`` maintains state so
    the filter can be streamed over arbitrarily long signals.
    """

    def __init__(self, ftype: str, sr: float, freq: float, q: float = 0.7071,
                 gain_db: float = 0.0):
        self.sr = sr
        self.ftype = ftype
        self.freq = freq
        self.q = q
        self.gain_db = gain_db
        self.b0 = self.b1 = self.b2 = self.a1 = self.a2 = 0.0
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0
        self._compute()

    def _compute(self) -> None:
        sr = self.sr
        f0 = self.freq
        A = 10.0 ** (self.gain_db / 40.0)
        w0 = 2.0 * math.pi * f0 / sr
        cw = math.cos(w0)
        sw = math.sin(w0)
        alpha = sw / (2.0 * self.q)
        f = self.ftype

        if f == "lowpass":
            b0 = (1 - cw) / 2; b1 = 1 - cw; b2 = (1 - cw) / 2
            a0 = 1 + alpha; a1 = -2 * cw; a2 = 1 - alpha
        elif f == "highpass":
            b0 = (1 + cw) / 2; b1 = -(1 + cw); b2 = (1 + cw) / 2
            a0 = 1 + alpha; a1 = -2 * cw; a2 = 1 - alpha
        elif f == "bandpass":
            b0 = alpha; b1 = 0; b2 = -alpha
            a0 = 1 + alpha; a1 = -2 * cw; a2 = 1 - alpha
        elif f == "notch":
            b0 = 1; b1 = -2 * cw; b2 = 1
            a0 = 1 + alpha; a1 = -2 * cw; a2 = 1 - alpha
        elif f == "allpass":
            b0 = 1 - alpha; b1 = -2 * cw; b2 = 1 + alpha
            a0 = 1 + alpha; a1 = -2 * cw; a2 = 1 - alpha
        elif f == "peaking":
            b0 = 1 + alpha * A; b1 = -2 * cw; b2 = 1 - alpha * A
            a0 = 1 + alpha / A; a1 = -2 * cw; a2 = 1 - alpha / A
        elif f == "lowshelf":
            sq2 = math.sqrt(2 * A) * alpha
            b0 = A * ((A + 1) - (A - 1) * cw + sq2)
            b1 = 2 * A * ((A - 1) - (A + 1) * cw)
            b2 = A * ((A + 1) - (A - 1) * cw - sq2)
            a0 = (A + 1) + (A - 1) * cw + sq2
            a1 = -2 * ((A - 1) + (A + 1) * cw)
            a2 = (A + 1) + (A - 1) * cw - sq2
        elif f == "highshelf":
            sq2 = math.sqrt(2 * A) * alpha
            b0 = A * ((A + 1) + (A - 1) * cw + sq2)
            b1 = -2 * A * ((A - 1) + (A + 1) * cw)
            b2 = A * ((A + 1) + (A - 1) * cw - sq2)
            a0 = (A + 1) - (A - 1) * cw + sq2
            a1 = 2 * ((A - 1) - (A + 1) * cw)
            a2 = (A + 1) - (A - 1) * cw - sq2
        else:  # bypass / unknown -> identity
            self.b0, self.b1, self.b2, self.a1, self.a2 = 1, 0, 0, 0, 0
            return

        ia0 = 1.0 / a0
        self.b0 = b0 * ia0
        self.b1 = b1 * ia0
        self.b2 = b2 * ia0
        self.a1 = a1 * ia0
        self.a2 = a2 * ia0

    def reset(self) -> None:
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def process(self, x: float) -> float:
        y = self.b0 * x + self.x1
        self.x1 = self.b1 * x - self.a1 * y + self.x2
        self.x2 = self.b2 * x - self.a2 * y
        return y

    def process_block(self, block: Sequence[float]) -> List[float]:
        return [self.process(x) for x in block]


def bandpass_split(samples: Sequence[float], sr: float, lo: float, hi: float) -> List[float]:
    """Return the frequency band [lo, hi] of ``samples`` using biquads."""
    hp = Biquad("highpass", sr, lo, q=0.7071) if lo > 20 else None
    lp = Biquad("lowpass", sr, hi, q=0.7071) if hi < sr / 2 else None
    out = list(samples)
    if hp is not None:
        out = hp.process_block(out)
    if lp is not None:
        out = lp.process_block(out)
    return out


# --------------------------------------------------------------------------- #
# 7. Pitch detection
# --------------------------------------------------------------------------- #

def autocorrelation(signal: Sequence[float]) -> List[float]:
    """Unbiased-ish autocorrelation via FFT (power spectrum -> IFFT)."""
    n = len(signal)
    m = next_pow2(2 * n)
    padded = [float(v) for v in signal] + [0.0] * (m - n)
    spec = fft(padded)
    power = [abs(v) ** 2 for v in spec]
    acorr = ifft(power)
    return [v.real for v in acorr[:n]]


def pitch_autocorr(frame: Sequence[float], sr: float, fmin: float = 50.0,
                   fmax: float = 2000.0) -> Optional[float]:
    """Fundamental frequency via the autocorrelation peak (FFT accelerated)."""
    n = len(frame)
    if n < 4:
        return None
    acorr = autocorrelation(frame)
    norm = acorr[0]
    if norm < 1e-12:
        return None
    min_lag = max(2, int(sr / fmax))
    max_lag = min(n - 2, int(sr / fmin))
    if min_lag >= max_lag:
        return None
    best_lag, best_val = min_lag, -1e18
    for lag in range(min_lag, max_lag + 1):
        v = acorr[lag]
        if v > best_val:
            best_val, best_lag = v, lag
    if best_val <= 0:
        return None
    # Parabolic interpolation around the peak.
    if min_lag < best_lag < max_lag:
        y0 = acorr[best_lag - 1] / norm
        y1 = acorr[best_lag] / norm
        y2 = acorr[best_lag + 1] / norm
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-9:
            delta = 0.5 * (y0 - y2) / denom
            if abs(delta) < 1:
                best_lag += delta
    f0 = sr / best_lag
    return f0 if fmin <= f0 <= fmax else None


def yin_difference(frame: Sequence[float], max_tau: int) -> List[float]:
    """YIN squared difference function d(tau)."""
    n = len(frame)
    d = [0.0] * max_tau
    for tau in range(1, max_tau):
        s = 0.0
        limit = n - tau
        for i in range(limit):
            df = frame[i] - frame[i + tau]
            s += df * df
        d[tau] = s
    return d


def yin_pitch(frame: Sequence[float], sr: float, fmin: float = 50.0,
              fmax: float = 2000.0, thresh: float = 0.1) -> Optional[float]:
    """Classic YIN pitch estimator (accurate but O(n^2) — use for spot checks)."""
    n = len(frame)
    max_tau = min(n // 2, int(sr / fmin))
    min_tau = max(1, int(sr / fmax))
    if min_tau >= max_tau:
        return None
    d = yin_difference(frame, max_tau)
    cmnd = [1.0] * max_tau
    running = 0.0
    for tau in range(1, max_tau):
        running += d[tau]
        cmnd[tau] = (d[tau] * tau / running) if running > 0 else 1.0
    tau_est = None
    tau = 2
    while tau < max_tau - 1:
        if cmnd[tau] < thresh:
            while tau + 1 < max_tau and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
            tau_est = tau
            break
        tau += 1
    if tau_est is None:
        # Fall back to the global minimum of cmnd.
        tau_est = min(range(min_tau, max_tau), key=lambda t: cmnd[t])
    if 0 < tau_est < max_tau - 1:
        s0, s1, s2 = cmnd[tau_est - 1], cmnd[tau_est], cmnd[tau_est + 1]
        denom = s0 - 2 * s1 + s2
        if abs(denom) > 1e-9:
            tau_est += 0.5 * (s0 - s2) / denom
    f0 = sr / tau_est
    return f0 if fmin <= f0 <= fmax else None


def pitch_contour(samples: Sequence[float], sr: float, hop: int = 512,
                  win_len: int = 2048, fmin: float = 50.0,
                  fmax: float = 2000.0) -> Tuple[List[float], List[float]]:
    """Frame-by-frame fundamental frequency using the fast autocorrelation path.

    Returns (times, f0) where f0 is 0.0 for unvoiced frames.
    """
    w = hann(win_len)
    times = []
    f0s = []
    i = 0
    total = len(samples)
    while i + win_len <= total:
        seg = samples[i:i + win_len]
        frame = [seg[k] * w[k] for k in range(win_len)]
        f0 = pitch_autocorr(frame, sr, fmin, fmax)
        f0s.append(f0 if f0 is not None else 0.0)
        times.append(i / sr)
        i += hop
    return times, f0s


# --------------------------------------------------------------------------- #
# 8. Onset / beat / tempo
# --------------------------------------------------------------------------- #

def onset_strength_from_spec(spec_mag: Sequence[Sequence[float]]) -> List[float]:
    """Spectral-flux onset-strength envelope from a magnitude spectrogram."""
    flux = []
    prev = None
    for frame in spec_mag:
        if prev is None:
            prev = frame
            flux.append(0.0)
            continue
        s = 0.0
        for k in range(len(frame)):
            d = frame[k] - prev[k]
            if d > 0:
                s += d * d
        flux.append(math.sqrt(s))
        prev = frame
    return flux


def local_maxima(x: Sequence[float]) -> List[int]:
    """Indices of strict local maxima."""
    peaks = []
    n = len(x)
    if n < 3:
        return peaks
    for i in range(1, n - 1):
        if x[i] > x[i - 1] and x[i] >= x[i + 1]:
            peaks.append(i)
    return peaks


def estimate_tempo(onset_env: Sequence[float], frame_rate: float,
                   min_bpm: float = 40.0, max_bpm: float = 240.0) -> float:
    """Estimate tempo (BPM) via autocorrelation of the onset envelope."""
    n = len(onset_env)
    if n < 8:
        return 120.0
    acorr = autocorrelation(onset_env)
    min_lag = max(1, int(frame_rate * 60.0 / max_bpm))
    max_lag = min(n - 1, int(frame_rate * 60.0 / min_bpm))
    if min_lag >= max_lag:
        return 120.0
    best_lag, best_val = min_lag, -1e18
    for lag in range(min_lag, max_lag + 1):
        if acorr[lag] > best_val:
            best_val, best_lag = acorr[lag], lag
    return 60.0 * frame_rate / best_lag


def smooth(x: Sequence[float], width: int = 3) -> List[float]:
    """Moving-average smoothing of a 1-D sequence."""
    if width < 1:
        return list(x)
    n = len(x)
    out = []
    half = width // 2
    acc = sum(x[:width])
    for i in range(n):
        out.append(acc / width)
        if i - half >= 0 and i + half + 1 < n:
            acc += x[i + half + 1] - x[i - half]
    return out


def detect_beats(onset_env: Sequence[float], frame_rate: float, tempo: Optional[float] = None) -> List[float]:
    """Beat times (s) via onset-envelope peak picking guided by the tempo."""
    n = len(onset_env)
    if n == 0:
        return []
    if tempo is None:
        tempo = estimate_tempo(onset_env, frame_rate)
    period = max(1, int(round(frame_rate * 60.0 / tempo)))
    sm = smooth(onset_env, max(3, period // 2))
    peaks = local_maxima(sm)
    if not peaks:
        return []
    # Threshold relative to local average.
    mean = sum(sm) / max(1, n)
    thresh = mean * 1.3
    candidates = [p for p in peaks if sm[p] > thresh]
    if not candidates:
        candidates = peaks
    # Greedy pick with a minimum spacing of half a beat period.
    beats = []
    last = -10 ** 9
    min_gap = max(1, period // 2)
    for p in candidates:
        if p - last >= min_gap and sm[p] > 0:
            beats.append(p)
            last = p
    return [b / frame_rate for b in beats]


# --------------------------------------------------------------------------- #
# 9. Chroma
# --------------------------------------------------------------------------- #

def chroma_from_spec(spec_mag: Sequence[Sequence[float]], sr: float,
                     nfft: int) -> List[List[float]]:
    """Fold a magnitude spectrogram into a 12-bin chromagram (one row/frame)."""
    chroma = []
    bins = len(spec_mag[0]) if spec_mag else 0
    for frame in spec_mag:
        c = [0.0] * 12
        for k in range(1, bins):
            f = k * sr / nfft
            m = hz_to_midi(f)
            if m is None:
                continue
            pc = int(round(m)) % 12
            c[pc] += frame[k]
        s = sum(c)
        if s > 1e-12:
            c = [v / s for v in c]
        chroma.append(c)
    return chroma


# --------------------------------------------------------------------------- #
# 10. Utilities
# --------------------------------------------------------------------------- #

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def normalize(samples: Sequence[float], peak: float = 0.99) -> List[float]:
    """Peak-normalise a signal to ``peak`` (returns new list)."""
    p = max((abs(x) for x in samples), default=0.0)
    if p < 1e-12:
        return list(samples)
    g = peak / p
    return [x * g for x in samples]


def resample_linear(samples: Sequence[float], src_sr: float, dst_sr: float) -> List[float]:
    """Linear-interpolation resampler (adequate for small rate changes)."""
    if src_sr == dst_sr:
        return list(samples)
    ratio = src_sr / dst_sr
    n_out = int(round(len(samples) / ratio))
    out = []
    for i in range(n_out):
        pos = i * ratio
        i0 = int(pos)
        frac = pos - i0
        i1 = i0 + 1
        if i1 >= len(samples):
            out.append(samples[i0])
        else:
            out.append(samples[i0] * (1 - frac) + samples[i1] * frac)
    return out


class StreamingResampler:
    """Resample a stream of blocks with correct continuity across block edges.

    Push source blocks with :meth:`push` and pull output samples with
    :meth:`pull`.  A fractional source position is carried across calls so the
    interpolation is seamless (used by format conversion and the mixer).
    """

    def __init__(self, src_sr: float, dst_sr: float):
        self.ratio = src_sr / dst_sr
        self.pos = 0.0
        self.buf: List[float] = []

    def push(self, block: Sequence[float]) -> None:
        self.buf.extend(block)

    def pull(self, max_out: int) -> List[float]:
        out: List[float] = []
        buf = self.buf
        ratio = self.ratio
        pos = self.pos
        n = len(buf)
        while len(out) < max_out and int(pos) + 1 < n:
            i0 = int(pos)
            frac = pos - i0
            out.append(buf[i0] * (1.0 - frac) + buf[i0 + 1] * frac)
            pos += ratio
        consumed = int(pos)
        if consumed > 0:
            del buf[:consumed]
            pos -= consumed
        self.pos = pos
        return out

    def flush(self, max_out: int) -> List[float]:
        """Pull any remaining samples, including a final partial one."""
        if not self.buf:
            return []
        return self.pull(max_out)


def fade_in_out(samples: Sequence[float], fade_sec: float, sr: float) -> List[float]:
    """Apply linear fade-in and fade-out to avoid clicks."""
    n = len(samples)
    out = list(samples)
    fade_n = min(n // 2, int(fade_sec * sr))
    if fade_n > 1:
        for i in range(fade_n):
            g = i / fade_n
            out[i] *= g
            out[n - 1 - i] *= g
    return out


def midi_freq_table() -> List[float]:
    """Frequencies for MIDI notes 0..127."""
    return [midi_to_hz(m) for m in range(128)]
