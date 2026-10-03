"""
effects.py — A streaming audio effects chain (equaliser, compressor, reverb,
delay, distortion, chorus, …).

Design
------
An effect is an object with ``process_block(block) -> list`` that maintains its
own internal state, so the whole chain can be streamed over arbitrarily long
files with bounded memory: read a chunk -> run it through every effect in the
chain -> write the chunk.  Only the preview path materialises a short excerpt
in memory.

Every effect is self-implemented from first principles:

  * EQ      — cascaded RBJ biquads (peaking / shelving / low-high pass / notch)
  * Compressor — attack/release envelope follower + gain computer (soft knee)
  * Reverb  — Schroeder: parallel comb filters + series all-pass filters
  * Delay   — feedback delay line
  * Distortion — soft (tanh) / hard clip / bitcrusher
  * Chorus  — LFO-modulated delay line
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from . import audio_io, dsp


# --------------------------------------------------------------------------- #
# Delay-line helper
# --------------------------------------------------------------------------- #

class DelayLine:
    """Fractional (linearly-interpolated) delay line."""

    def __init__(self, max_samples: int):
        self.buf = [0.0] * max(1, max_samples)
        self.idx = 0
        self.size = len(self.buf)

    def write(self, x: float) -> None:
        self.buf[self.idx] = x
        self.idx = (self.idx + 1) % self.size

    def read(self, delay: float) -> float:
        pos = self.idx - delay
        while pos < 0:
            pos += self.size
        i0 = int(pos)
        frac = pos - i0
        i1 = (i0 + 1) % self.size
        return self.buf[i0] * (1.0 - frac) + self.buf[i1] * frac


# --------------------------------------------------------------------------- #
# Effects
# --------------------------------------------------------------------------- #

class Gain:
    def __init__(self, gain_db: float = 0.0):
        self.g = 10.0 ** (gain_db / 20.0)

    def process_block(self, block: Sequence[float]) -> List[float]:
        g = self.g
        return [x * g for x in block]


class Eq:
    def __init__(self, bands: List[Dict], sr: float):
        self.biquads = []
        for band in bands:
            b = dsp.Biquad(
                band.get("type", "peaking"),
                sr,
                band.get("freq", 1000.0),
                band.get("q", 1.0),
                band.get("gain", 0.0),
            )
            self.biquads.append(b)

    def process_block(self, block: Sequence[float]) -> List[float]:
        out = list(block)
        for bq in self.biquads:
            out = bq.process_block(out)
        return out


class Highpass:
    def __init__(self, freq: float, sr: float):
        self.bq = dsp.Biquad("highpass", sr, freq, 0.7071)

    def process_block(self, block: Sequence[float]) -> List[float]:
        return self.bq.process_block(block)


class Lowpass:
    def __init__(self, freq: float, sr: float):
        self.bq = dsp.Biquad("lowpass", sr, freq, 0.7071)

    def process_block(self, block: Sequence[float]) -> List[float]:
        return self.bq.process_block(block)


class Compressor:
    def __init__(self, threshold_db: float = -20.0, ratio: float = 4.0,
                 attack_s: float = 0.01, release_s: float = 0.2,
                 makeup_db: float = 0.0, knee_db: float = 6.0, sr: float = 44100):
        self.threshold_db = threshold_db
        self.ratio = max(1.0, ratio)
        self.knee = knee_db
        self.makeup = 10.0 ** (makeup_db / 20.0)
        self.attack_coef = math.exp(-1.0 / max(1e-4, attack_s * sr))
        self.release_coef = math.exp(-1.0 / max(1e-4, release_s * sr))
        self.env = 0.0

    def process_block(self, block: Sequence[float]) -> List[float]:
        out = []
        for x in block:
            level = abs(x)
            if level > self.env:
                self.env += self.attack_coef * (level - self.env)
            else:
                self.env += self.release_coef * (level - self.env)
            gain = 1.0
            if self.env > 1e-9:
                env_db = 20.0 * math.log10(self.env)
                over = env_db - self.threshold_db
                if over > 0:
                    # soft knee
                    if self.knee > 0 and over < self.knee:
                        over = (over * over) / (2.0 * self.knee)
                    gain_db = over * (1.0 / self.ratio - 1.0)
                    gain = 10.0 ** (gain_db / 20.0)
            out.append(x * gain * self.makeup)
        return out


class _Comb:
    def __init__(self, delay: int, feedback: float, damping: float):
        self.dl = DelayLine(delay + 1)
        self.delay = delay
        self.feedback = feedback
        self.damping = damping
        self.filterstore = 0.0

    def process(self, x: float) -> float:
        d = self.dl.read(self.delay)
        self.filterstore = d * (1.0 - self.damping) + self.filterstore * self.damping
        self.dl.write(x + self.filterstore * self.feedback)
        return d


class _Allpass:
    def __init__(self, delay: int, feedback: float):
        self.dl = DelayLine(delay + 1)
        self.delay = delay
        self.feedback = feedback

    def process(self, x: float) -> float:
        d = self.dl.read(self.delay)
        out = -x + d
        self.dl.write(x + d * self.feedback)
        return out


class Reverb:
    """Schroeder reverb: 4 parallel combs -> 2 series all-passes -> wet/dry mix."""

    def __init__(self, room_size: float = 0.7, damping: float = 0.4,
                 wet: float = 0.3, dry: float = 0.8, predelay_s: float = 0.0,
                 sr: float = 44100):
        fb = 0.7 + 0.2 * room_size
        self.combs = [
            _Comb(int(sr * 0.0297), fb, damping),
            _Comb(int(sr * 0.0371), fb, damping),
            _Comb(int(sr * 0.0411), fb, damping),
            _Comb(int(sr * 0.0437), fb, damping),
        ]
        self.allpasses = [
            _Allpass(int(sr * 0.0050), 0.5),
            _Allpass(int(sr * 0.0017), 0.5),
        ]
        self.predelay = DelayLine(int(sr * 0.1))
        self.predelay_s = predelay_s
        self.wet = wet
        self.dry = dry

    def process_block(self, block: Sequence[float]) -> List[float]:
        predelay_samples = self.predelay_s * 1  # sr unknown per-block; use time directly
        out = []
        for x in block:
            wet = 0.0
            for c in self.combs:
                wet += c.process(x)
            wet /= len(self.combs)
            for a in self.allpasses:
                wet = a.process(wet)
            out.append(x * self.dry + wet * self.wet)
        return out


class Delay:
    def __init__(self, time_s: float = 0.3, feedback: float = 0.4,
                 wet: float = 0.3, sr: float = 44100):
        self.dl = DelayLine(int(sr * 2.0))
        self.delay = int(time_s * sr)
        self.feedback = feedback
        self.wet = wet

    def process_block(self, block: Sequence[float]) -> List[float]:
        out = []
        for x in block:
            d = self.dl.read(self.delay)
            self.dl.write(x + d * self.feedback)
            out.append(x + d * self.wet)
        return out


class Distortion:
    def __init__(self, drive: float = 10.0, mode: str = "soft", bits: int = 8):
        self.drive = drive
        self.mode = mode
        self.bits = bits

    def process_block(self, block: Sequence[float]) -> List[float]:
        out = []
        if self.mode == "soft":
            t = math.tanh(self.drive)
            out = [math.tanh(self.drive * x) / t if t > 1e-9 else x for x in block]
        elif self.mode == "hard":
            out = [dsp.clamp(self.drive * x, -1.0, 1.0) for x in block]
        elif self.mode == "bitcrush":
            steps = 2 ** (self.bits - 1)
            out = [round(dsp.clamp(x, -1.0, 1.0) * steps) / steps for x in block]
        else:
            out = list(block)
        return out


class Chorus:
    def __init__(self, rate: float = 0.6, depth_ms: float = 8.0,
                 mix: float = 0.5, sr: float = 44100):
        self.dl = DelayLine(int(sr * 0.1))
        self.rate = rate
        self.depth = depth_ms / 1000.0 * sr
        self.mix = mix
        self.sr = sr
        self.phase = 0.0

    def process_block(self, block: Sequence[float]) -> List[float]:
        out = []
        for x in block:
            self.phase += 2.0 * math.pi * self.rate / self.sr
            delay = 1.0 + self.depth * (1.0 + math.sin(self.phase)) / 2.0
            wet = self.dl.read(delay)
            self.dl.write(x)
            out.append(x * (1.0 - self.mix) + wet * self.mix)
        return out


class Reverse:
    """Buffer the whole signal and reverse it (preview / in-memory only)."""

    def __init__(self):
        self.buf: List[float] = []

    def process_block(self, block: Sequence[float]) -> List[float]:
        self.buf.extend(block)
        return []

    def finish(self) -> List[float]:
        return list(reversed(self.buf))


# --------------------------------------------------------------------------- #
# Factory + chain
# --------------------------------------------------------------------------- #

def make_effect(spec: Dict, sr: float):
    etype = spec.get("type", "gain")
    params = spec.get("params", {})
    if etype == "gain":
        return Gain(params.get("gain_db", 0.0))
    if etype == "eq":
        return Eq(params.get("bands", []), sr)
    if etype == "highpass":
        return Highpass(params.get("freq", 200.0), sr)
    if etype == "lowpass":
        return Lowpass(params.get("freq", 8000.0), sr)
    if etype == "compressor":
        return Compressor(
            params.get("threshold_db", -20.0),
            params.get("ratio", 4.0),
            params.get("attack_s", 0.01),
            params.get("release_s", 0.2),
            params.get("makeup_db", 0.0),
            params.get("knee_db", 6.0),
            sr,
        )
    if etype == "reverb":
        return Reverb(
            params.get("room_size", 0.7),
            params.get("damping", 0.4),
            params.get("wet", 0.3),
            params.get("dry", 0.8),
            params.get("predelay_s", 0.0),
            sr,
        )
    if etype == "delay":
        return Delay(
            params.get("time_s", 0.3),
            params.get("feedback", 0.4),
            params.get("wet", 0.3),
            sr,
        )
    if etype == "distortion":
        return Distortion(
            params.get("drive", 10.0),
            params.get("mode", "soft"),
            params.get("bits", 8),
        )
    if etype == "chorus":
        return Chorus(
            params.get("rate", 0.6),
            params.get("depth_ms", 8.0),
            params.get("mix", 0.5),
            sr,
        )
    if etype == "reverse":
        return Reverse()
    raise ValueError(f"unknown effect type {etype!r}")


class EffectChain:
    def __init__(self, specs: List[Dict], sr: float):
        self.specs = [s for s in specs if s.get("enabled", True)]
        self.sr = sr
        self.effects = [make_effect(s, sr) for s in self.specs]

    def process_block(self, block: Sequence[float]) -> List[float]:
        out = list(block)
        for fx in self.effects:
            out = fx.process_block(out)
        return out

    def process_channels(self, channels: Sequence[Sequence[float]]) -> List[List[float]]:
        return [self.process_block(ch) for ch in channels]

    def finish_reverse(self) -> Optional[List[float]]:
        for fx in self.effects:
            if isinstance(fx, Reverse):
                return fx.finish()
        return None


def apply_chain_to_file(src_path: str, dst_path: str, specs: List[Dict]) -> Dict:
    """Stream the effect chain over a whole file and write the result."""
    with audio_io.WavReader(src_path) as r:
        sr = r.sr
        chain = EffectChain(specs, sr)
        reversed_bufs: List[Optional[List[float]]] = None
        has_reverse = any(e.get("type") == "reverse" and e.get("enabled", True) for e in specs)

        if has_reverse:
            # Reverse must buffer the entire signal — handled specially.
            bufs: List[List[float]] = [[] for _ in range(r.channels)]
            for chunk in r.iter_chunks():
                for c, ch in enumerate(chunk):
                    bufs[c].extend(ch)
            with audio_io.WavWriter(dst_path, sr, r.channels, 2) as w:
                w.write_chunk([bufs[c][::-1] for c in range(r.channels)])
            return {"frames": r.nframes, "reversed": True}

        with audio_io.WavWriter(dst_path, sr, r.channels, 2) as w:
            for chunk in r.iter_chunks():
                out_ch = chain.process_channels(chunk)
                w.write_chunk(out_ch)
        return {"frames": r.nframes, "reversed": False}


def render_preview(src_path: str, specs: List[Dict], start_s: float = 0.0,
                   duration_s: float = 6.0) -> bytes:
    """Render a short excerpt through the chain and return it as a WAV byte string."""
    with audio_io.WavReader(src_path) as r:
        sr = r.sr
        start = max(0, int(start_s * sr))
        n = int(duration_s * sr)
        if hasattr(r, "_w"):
            r._w.setpos(start)
        chunk = r.read_chunk(n)
        if chunk is None:
            chunk = [[0.0]]
        chain = EffectChain(specs, sr)
        out_ch = chain.process_channels(chunk)
        rev = chain.finish_reverse()
        if rev is not None:
            out_ch = [rev]

    # Encode the preview to a 16-bit WAV in memory.
    import io
    import struct
    buf = io.BytesIO()
    nframes = min(len(c) for c in out_ch) if out_ch else 0
    channels = len(out_ch)
    data = bytearray()
    for i in range(nframes):
        for c in out_ch:
            v = int(round(max(-1.0, min(1.0, c[i])) * 32767))
            data += struct.pack("<h", v)
    byte_rate = sr * channels * 2
    block_align = channels * 2
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(data)))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, channels, sr, byte_rate, block_align, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(data)))
    buf.write(bytes(data))
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #

PRESETS: Dict[str, Dict] = {
    "Vocal enhancer": {
        "description": "Presence boost + gentle compression",
        "chain": [
            {"type": "highpass", "params": {"freq": 100}, "enabled": True},
            {"type": "eq", "params": {"bands": [
                {"type": "peaking", "freq": 3000, "gain": 3.0, "q": 0.8},
                {"type": "highshelf", "freq": 8000, "gain": 2.0, "q": 0.7},
            ]}, "enabled": True},
            {"type": "compressor", "params": {"threshold_db": -18, "ratio": 3, "makeup_db": 2}, "enabled": True},
        ],
    },
    "Radio / lofi": {
        "description": "Band-pass EQ + warm overdrive",
        "chain": [
            {"type": "lowpass", "params": {"freq": 6000}, "enabled": True},
            {"type": "highpass", "params": {"freq": 250}, "enabled": True},
            {"type": "distortion", "params": {"drive": 6, "mode": "soft"}, "enabled": True},
            {"type": "compressor", "params": {"threshold_db": -14, "ratio": 5, "makeup_db": 3}, "enabled": True},
        ],
    },
    "Large hall reverb": {
        "description": "Spacious Schroeder reverb with predelay",
        "chain": [
            {"type": "reverb", "params": {"room_size": 0.9, "damping": 0.3, "wet": 0.4, "dry": 0.8, "predelay_s": 0.03}, "enabled": True},
        ],
    },
    "Punchy drums": {
        "description": "EQ scoop + hard-knee compression",
        "chain": [
            {"type": "lowpass", "params": {"freq": 14000}, "enabled": True},
            {"type": "eq", "params": {"bands": [
                {"type": "peaking", "freq": 60, "gain": 4, "q": 1.0},
                {"type": "peaking", "freq": 400, "gain": -4, "q": 1.4},
                {"type": "highshelf", "freq": 6000, "gain": 3, "q": 0.7},
            ]}, "enabled": True},
            {"type": "compressor", "params": {"threshold_db": -16, "ratio": 6, "attack_s": 0.003, "release_s": 0.1, "makeup_db": 3, "knee_db": 2}, "enabled": True},
        ],
    },
    "Telephone": {
        "description": "Narrow band-pass with hard clipping",
        "chain": [
            {"type": "highpass", "params": {"freq": 300}, "enabled": True},
            {"type": "lowpass", "params": {"freq": 3400}, "enabled": True},
            {"type": "distortion", "params": {"drive": 4, "mode": "hard"}, "enabled": True},
        ],
    },
}


def preset_list() -> List[Dict]:
    return [{"name": k, **v} for k, v in PRESETS.items()]


def effect_catalog() -> List[Dict]:
    """Describe every effect type and its parameters (for the UI)."""
    return [
        {"type": "gain", "label": "Gain", "params": [
            {"name": "gain_db", "label": "Gain (dB)", "min": -30, "max": 30, "step": 0.5, "default": 0.0}]},
        {"type": "eq", "label": "Equaliser", "params": [
            {"name": "bands", "label": "Bands", "complex": "eq_bands"}]},
        {"type": "highpass", "label": "High-pass", "params": [
            {"name": "freq", "label": "Cutoff (Hz)", "min": 20, "max": 2000, "step": 10, "default": 200, "log": True}]},
        {"type": "lowpass", "label": "Low-pass", "params": [
            {"name": "freq", "label": "Cutoff (Hz)", "min": 200, "max": 20000, "step": 10, "default": 8000, "log": True}]},
        {"type": "compressor", "label": "Compressor", "params": [
            {"name": "threshold_db", "label": "Threshold (dB)", "min": -60, "max": 0, "step": 1, "default": -20},
            {"name": "ratio", "label": "Ratio", "min": 1, "max": 20, "step": 0.5, "default": 4},
            {"name": "attack_s", "label": "Attack (s)", "min": 0.001, "max": 0.5, "step": 0.001, "default": 0.01},
            {"name": "release_s", "label": "Release (s)", "min": 0.01, "max": 2, "step": 0.01, "default": 0.2},
            {"name": "makeup_db", "label": "Makeup (dB)", "min": 0, "max": 24, "step": 0.5, "default": 0},
            {"name": "knee_db", "label": "Knee (dB)", "min": 0, "max": 24, "step": 1, "default": 6}]},
        {"type": "reverb", "label": "Reverb", "params": [
            {"name": "room_size", "label": "Room size", "min": 0, "max": 1, "step": 0.05, "default": 0.7},
            {"name": "damping", "label": "Damping", "min": 0, "max": 1, "step": 0.05, "default": 0.4},
            {"name": "wet", "label": "Wet", "min": 0, "max": 1, "step": 0.05, "default": 0.3},
            {"name": "dry", "label": "Dry", "min": 0, "max": 1, "step": 0.05, "default": 0.8},
            {"name": "predelay_s", "label": "Pre-delay (s)", "min": 0, "max": 0.2, "step": 0.01, "default": 0}]},
        {"type": "delay", "label": "Delay", "params": [
            {"name": "time_s", "label": "Time (s)", "min": 0.02, "max": 2, "step": 0.01, "default": 0.3},
            {"name": "feedback", "label": "Feedback", "min": 0, "max": 0.95, "step": 0.05, "default": 0.4},
            {"name": "wet", "label": "Wet", "min": 0, "max": 1, "step": 0.05, "default": 0.3}]},
        {"type": "distortion", "label": "Distortion", "params": [
            {"name": "drive", "label": "Drive", "min": 0, "max": 50, "step": 1, "default": 10},
            {"name": "mode", "label": "Mode", "enum": ["soft", "hard", "bitcrush"], "default": "soft"},
            {"name": "bits", "label": "Bits (bitcrush)", "min": 2, "max": 16, "step": 1, "default": 8}]},
        {"type": "chorus", "label": "Chorus", "params": [
            {"name": "rate", "label": "Rate (Hz)", "min": 0.05, "max": 5, "step": 0.05, "default": 0.6},
            {"name": "depth_ms", "label": "Depth (ms)", "min": 0, "max": 20, "step": 0.5, "default": 8},
            {"name": "mix", "label": "Mix", "min": 0, "max": 1, "step": 0.05, "default": 0.5}]},
        {"type": "reverse", "label": "Reverse", "params": []},
    ]
