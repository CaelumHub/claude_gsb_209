"""
audio_io.py — Audio file reading, writing and format conversion.

Design goals
------------
* **Streaming for large files.**  We never need to hold an entire track in
  memory.  ``WavReader`` yields fixed-size chunks and can extract a small
  excerpt, and ``WavWriter`` accepts a generator of chunks so effects and
  export can process multi-GB files with a bounded memory footprint.
* **Pure-stdlib PCM handling.**  WAV is decoded/encoded directly with ``wave``
  and ``struct``.  Compressed formats (mp3/ogg/flac/m4a) are delegated to
  ``ffmpeg`` when it is available.
* **Normalised float samples.**  Internally every sample is a float in
  [-1.0, 1.0], stored de-interleaved as one list per channel.

Supported formats
-----------------
  read : wav (PCM 8/16/24/32-bit, IEEE float 32/64), + anything ffmpeg decodes
  write: wav (PCM 8/16/24/32-bit, float 32), aiff, raw, + ffmpeg encoders
         (mp3, ogg/vorbis, flac, m4a/aac)
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Sequence, Tuple

from . import dsp

# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class AudioData:
    """De-interleaved float samples in [-1, 1] plus sample rate.

    ``samples`` is a list with one entry per channel; each entry is a list of
    floats.  Mono audio therefore has ``samples == [ch0]``.
    """

    samples: List[List[float]] = field(default_factory=list)
    sr: int = 44100

    @property
    def channels(self) -> int:
        return len(self.samples)

    @property
    def frames(self) -> int:
        return len(self.samples[0]) if self.samples else 0

    @property
    def duration(self) -> float:
        return self.frames / self.sr if self.sr else 0.0

    def to_mono(self) -> List[float]:
        return to_mono(self.samples)


def to_mono(channels: Sequence[Sequence[float]]) -> List[float]:
    """Average channels into a mono signal."""
    n = len(channels)
    if n == 0:
        return []
    if n == 1:
        return list(channels[0])
    length = min(len(c) for c in channels)
    inv = 1.0 / n
    return [sum(c[i] for c in channels) * inv for i in range(length)]


def interleave(channels: Sequence[Sequence[float]]) -> List[float]:
    """De-interleaved channel lists -> interleaved sample list."""
    if not channels:
        return []
    length = min(len(c) for c in channels)
    out = []
    for i in range(length):
        for c in channels:
            out.append(c[i])
    return out


def deinterleave(interleaved: Sequence[float], channels: int) -> List[List[float]]:
    """Interleaved sample list -> de-interleaved channel lists."""
    return [list(interleaved[i::channels]) for i in range(channels)]


# --------------------------------------------------------------------------- #
# WAV decode / encode primitives
# --------------------------------------------------------------------------- #

def _decode_pcm(raw: bytes, sample_width: int) -> List[float]:
    if sample_width == 1:  # unsigned 8-bit
        return [(b - 128) / 128.0 for b in raw]
    if sample_width == 2:  # signed 16-bit LE
        vals = struct.unpack("<%dh" % (len(raw) // 2), raw)
        return [v / 32768.0 for v in vals]
    if sample_width == 3:  # signed 24-bit LE
        out = []
        for i in range(0, len(raw) - 2, 3):
            b0, b1, b2 = raw[i], raw[i + 1], raw[i + 2]
            v = b0 | (b1 << 8) | (b2 << 16)
            if v & 0x800000:
                v -= 0x1000000
            out.append(v / 8388608.0)
        return out
    if sample_width == 4:  # signed 32-bit LE
        vals = struct.unpack("<%di" % (len(raw) // 4), raw)
        return [v / 2147483648.0 for v in vals]
    raise ValueError(f"unsupported sample width {sample_width}")


def _decode_float(raw: bytes, sample_width: int) -> List[float]:
    if sample_width == 4:
        vals = struct.unpack("<%df" % (len(raw) // 4), raw)
        return [float(v) for v in vals]
    if sample_width == 8:
        vals = struct.unpack("<%dd" % (len(raw) // 8), raw)
        return [float(v) for v in vals]
    raise ValueError(f"unsupported float width {sample_width}")


def _encode_pcm_16(samples: Sequence[float]) -> bytes:
    out = bytearray(len(samples) * 2)
    pos = 0
    for x in samples:
        v = int(round(max(-1.0, min(1.0, x)) * 32767))
        out[pos] = v & 0xFF
        out[pos + 1] = (v >> 8) & 0xFF
        pos += 2
    return bytes(out)


def _encode_pcm_24(samples: Sequence[float]) -> bytes:
    out = bytearray(len(samples) * 3)
    pos = 0
    for x in samples:
        v = int(round(max(-1.0, min(1.0, x)) * 8388607))
        out[pos] = v & 0xFF
        out[pos + 1] = (v >> 8) & 0xFF
        out[pos + 2] = (v >> 16) & 0xFF
        pos += 3
    return bytes(out)


def _encode_float32(samples: Sequence[float]) -> bytes:
    out = bytearray(len(samples) * 4)
    pos = 0
    for x in samples:
        out[pos:pos + 4] = struct.pack("<f", max(-1.0, min(1.0, x)))
        pos += 4
    return bytes(out)


def _encode_pcm(samples: Sequence[float], sample_width: int) -> bytes:
    if sample_width == 2:
        return _encode_pcm_16(samples)
    if sample_width == 3:
        return _encode_pcm_24(samples)
    if sample_width == 4:
        return _encode_float32(samples)
    if sample_width == 1:
        out = bytearray(len(samples))
        for i, x in enumerate(samples):
            out[i] = int(round(max(-1.0, min(1.0, x)) * 127 + 128)) & 0xFF
        return bytes(out)
    raise ValueError(f"unsupported sample width {sample_width}")


# --------------------------------------------------------------------------- #
# Streaming reader / writer
# --------------------------------------------------------------------------- #

class WavReader:
    """Streaming reader for PCM/float WAV files.

    Supports chunked iteration (bounded memory) and random-access excerpt
    extraction (used for waveform previews)."""

    def __init__(self, path: str):
        self.path = path
        self._w = wave.open(path, "rb")
        self.channels = self._w.getnchannels()
        self.sr = self._w.getframerate()
        self.sample_width = self._w.getsampwidth()
        self.comptype = self._w.getcomptype()
        self.nframes = self._w.getnframes()
        self._pos = 0

    @property
    def duration(self) -> float:
        return self.nframes / self.sr if self.sr else 0.0

    def close(self) -> None:
        self._w.close()

    def __enter__(self) -> "WavReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def read_chunk(self, nframes: int) -> Optional[List[List[float]]]:
        """Read up to ``nframes`` frames as de-interleaved channels (None at EOF)."""
        raw = self._w.readframes(nframes)
        if not raw:
            return None
        n = len(raw) // (self.sample_width * self.channels)
        inter = _decode_float(raw, self.sample_width) if self.comptype == 'NONE' and self.sample_width > 2 and self._is_float() \
            else _decode_pcm(raw, self.sample_width)
        self._pos += n
        return deinterleave(inter, self.channels)

    def iter_chunks(self, nframes: int = 1 << 16) -> Generator[List[List[float]], None, None]:
        """Yield consecutive chunks of up to ``nframes`` frames."""
        while True:
            chunk = self.read_chunk(nframes)
            if chunk is None:
                break
            yield chunk

    def _is_float(self) -> bool:
        # Heuristic: IEEE-float WAV is signalled by the 'WAVE_FORMAT_IEEE_FLOAT'
        # fmt tag (3).  wave exposes it only through comptype in some builds, so
        # we read the fmt tag directly from the header.
        try:
            with open(self.path, "rb") as f:
                f.seek(20)
                tag = struct.unpack("<H", f.read(2))[0]
                return tag == 3
        except Exception:
            return False

    def read_excerpt(self, start_frame: int, nframes: int) -> AudioData:
        """Random-access excerpt (used for waveform preview rendering)."""
        self._w.setpos(start_frame)
        chunk = self.read_chunk(nframes)
        return AudioData(chunk if chunk is not None else [[] for _ in range(self.channels)], self.sr)


def wav_pcm_header(sr: int, channels: int, sample_width: int,
                   data_size: int, fmt_tag: int = 1) -> bytes:
    """Build a 44-byte canonical PCM WAV header.

    ``data_size`` is the size of the PCM payload only (0 is legal while the
    size is still unknown).  ``fmt_tag`` 1 = PCM, 3 = IEEE float.
    """
    byte_rate = sr * channels * sample_width
    block_align = channels * sample_width
    bits = sample_width * 8
    return (b"RIFF"
            + struct.pack("<I", 36 + data_size)
            + b"WAVE"
            + b"fmt "
            + struct.pack("<IHHIIHH", 16, fmt_tag, channels, sr,
                          byte_rate, block_align, bits)
            + b"data"
            + struct.pack("<I", data_size))


class WavWriter:
    """Streaming PCM WAV writer that patches the header on close.

    Sample widths 1 (u8), 2 (s16), 3 (s24) and 4 (f32) are supported."""

    def __init__(self, path: str, sr: int, channels: int, sample_width: int = 2):
        self.path = path
        self.sr = int(sr)
        self.channels = channels
        self.sample_width = sample_width
        self._data_size = 0
        self._f = open(path, "wb")
        self._write_header()

    def _write_header(self) -> None:
        # Placeholder sizes (0); patched on close().
        fmt_tag = 3 if self.sample_width == 4 else 1
        self._f.write(wav_pcm_header(self.sr, self.channels, self.sample_width,
                                     0, fmt_tag=fmt_tag))

    def write_chunk(self, channels: Sequence[Sequence[float]]) -> None:
        if not channels:
            return
        n = min(len(c) for c in channels)
        if n == 0:
            return
        inter = []
        for i in range(n):
            for c in channels:
                inter.append(c[i])
        if self.sample_width == 4:
            raw = _encode_float32(inter)
        else:
            raw = _encode_pcm(inter, self.sample_width)
        self._f.write(raw)
        self._data_size += len(raw)

    def close(self) -> None:
        # Patch RIFF chunk size (offset 4) and data chunk size (offset 40).
        self._f.flush()
        self._f.seek(4)
        self._f.write(struct.pack("<I", 36 + self._data_size))
        self._f.seek(40)
        self._f.write(struct.pack("<I", self._data_size))
        self._f.close()

    def __enter__(self) -> "WavWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class RawReader:
    """Streaming reader for headerless PCM files (RAW export).

    A raw file carries no metadata of its own, so sample rate / channels /
    sample width must be supplied by the caller (they are stored in the
    library entry).  Exposes the same interface as :class:`WavReader` so the
    two are interchangeable in streaming/analysis code.
    """

    def __init__(self, path: str, sr: int, channels: int, sample_width: int = 2):
        self.path = path
        self.sr = int(sr)
        self.channels = int(channels)
        self.sample_width = int(sample_width)
        self.comptype = "NONE"
        self._f = open(path, "rb")
        size = os.fstat(self._f.fileno()).st_size
        frame_size = self.sample_width * self.channels
        self.nframes = size // frame_size if frame_size else 0
        self._pos = 0

    @property
    def duration(self) -> float:
        return self.nframes / self.sr if self.sr else 0.0

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "RawReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def read_chunk(self, nframes: int) -> Optional[List[List[float]]]:
        """Read up to ``nframes`` frames as de-interleaved channels (None at EOF)."""
        raw = self._f.read(nframes * self.sample_width * self.channels)
        if not raw:
            return None
        inter = _decode_pcm(raw, self.sample_width)
        n = len(inter) // self.channels
        self._pos += n
        return deinterleave(inter, self.channels)

    def iter_chunks(self, nframes: int = 1 << 16) -> Generator[List[List[float]], None, None]:
        """Yield consecutive chunks of up to ``nframes`` frames."""
        while True:
            chunk = self.read_chunk(nframes)
            if chunk is None:
                break
            yield chunk

    def read_excerpt(self, start_frame: int, nframes: int) -> AudioData:
        """Random-access excerpt (used for waveform preview rendering)."""
        frame_size = self.sample_width * self.channels
        self._f.seek(start_frame * frame_size)
        self._pos = start_frame
        chunk = self.read_chunk(nframes)
        return AudioData(chunk if chunk is not None else [[] for _ in range(self.channels)], self.sr)


def open_audio(path: str, sr: Optional[int] = None, channels: Optional[int] = None,
               sample_width: int = 2):
    """Open a local audio file for streaming.

    ``.raw`` files are headerless PCM: the caller must pass ``sr`` (and
    optionally ``channels`` / ``sample_width``, both defaulting to 16-bit and
    the source channel count).  Everything else is opened as WAV.
    """
    if path.lower().endswith(".raw"):
        if not sr:
            raise ValueError("raw PCM files require an explicit sample rate")
        return RawReader(path, sr, channels or 1, sample_width or 2)
    return WavReader(path)


# --------------------------------------------------------------------------- #
# Whole-file helpers
# --------------------------------------------------------------------------- #

MAX_IN_MEMORY_FRAMES = 4_000_000  # ~90 s stereo @44.1k — guard for whole-file loads


def load(path: str, max_frames: int = MAX_IN_MEMORY_FRAMES) -> AudioData:
    """Load an entire audio file into memory (WAV or ffmpeg-decodable)."""
    if not path.lower().endswith(".wav"):
        path = decode_with_ffmpeg(path)
    with WavReader(path) as r:
        if r.nframes > max_frames:
            raise MemoryError(
                f"file has {r.nframes} frames; exceeds in-memory limit {max_frames}. "
                "Use streaming processing instead."
            )
        chans = [[] for _ in range(r.channels)]
        for chunk in r.iter_chunks():
            for c, ch in enumerate(chunk):
                chans[c].extend(ch)
        return AudioData(chans, r.sr)


def save(path: str, audio: AudioData, sample_width: int = 2) -> None:
    """Write an AudioData object to a WAV file."""
    with WavWriter(path, audio.sr, audio.channels, sample_width) as w:
        w.write_chunk(audio.samples)


def resample(audio: AudioData, dst_sr: int) -> AudioData:
    if audio.sr == dst_sr:
        return audio
    ratio = audio.sr / dst_sr
    n_out = int(round(audio.frames / ratio))
    out = []
    for ch in audio.samples:
        out.append(dsp.resample_linear(ch, audio.sr, dst_sr)[:n_out])
    return AudioData(out, dst_sr)


# --------------------------------------------------------------------------- #
# ffmpeg interop for compressed formats
# --------------------------------------------------------------------------- #

def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def decode_with_ffmpeg(path: str, tmpdir: Optional[str] = None) -> str:
    """Decode any ffmpeg-supported audio file to a temporary 16-bit PCM WAV."""
    if not has_ffmpeg():
        raise RuntimeError("ffmpeg not found — cannot decode this format")
    fd, out = tempfile.mkstemp(suffix=".wav", dir=tmpdir)
    os.close(fd)
    try:
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-i", path, "-acodec", "pcm_s16le", out]
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        return out
    except subprocess.CalledProcessError as e:
        os.unlink(out)
        raise RuntimeError(f"ffmpeg decode failed: {e.stderr.decode(errors='replace')[:400]}")


def encode_with_ffmpeg(src_wav: str, dst_path: str, fmt: str,
                       bitrate: str = "192k") -> str:
    """Encode a WAV file to a compressed container via ffmpeg."""
    if not has_ffmpeg():
        raise RuntimeError("ffmpeg not found — cannot encode this format")
    codec_map = {
        "mp3": ("-acodec", "libmp3lame", "-b:a", bitrate),
        "ogg": ("-acodec", "libvorbis", "-q:a", "4"),
        "flac": ("-acodec", "flac",),
        "m4a": ("-acodec", "aac", "-b:a", bitrate),
    }
    if fmt not in codec_map:
        raise ValueError(f"unsupported ffmpeg format {fmt}")
    args = list(codec_map[fmt])
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", src_wav]
    cmd += args
    cmd += [dst_path]
    subprocess.run(cmd, check=True, capture_output=True, timeout=1200)
    return dst_path


# --------------------------------------------------------------------------- #
# Format conversion (export)
# --------------------------------------------------------------------------- #

WAV_SAMPLE_WIDTHS = {"pcm16": 2, "pcm24": 3, "pcm8": 1, "float32": 4}


def _probe_wav_header(path: str) -> Tuple[int, int, int, int]:
    """Parse ``(sr, channels, sample_width, frames)`` from a WAV header.

    A small chunk parser that, unlike stdlib :mod:`wave`, accepts IEEE-float
    files (fmt tag 3).  Only the canonical RIFF/WAVE layout emitted by this
    project is needed.
    """
    with open(path, "rb") as f:
        head = f.read(12)
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise ValueError("not a WAV file")
        sr = ch = sw = data_size = 0
        while True:
            cid = f.read(4)
            if len(cid) < 4:
                break
            size = struct.unpack("<I", f.read(4))[0]
            if cid == b"fmt ":
                body = f.read(min(size, 16))
                _, ch, sr, _, _, bits = struct.unpack("<HHIIHH", body[:16])
                sw = bits // 8
                if size > 16:
                    f.seek(size - 16, 1)
            elif cid == b"data":
                data_size = size
                break
            else:
                f.seek(size + (size & 1), 1)  # skip + pad byte
    frames = data_size // (sw * ch) if sw and ch else 0
    return sr, ch, sw, frames


def convert(src_path: str, dst_path: str, fmt: str, dst_sr: Optional[int] = None,
            sample_width: str = "pcm16", channels: Optional[int] = None,
            bitrate: str = "192k") -> Dict[str, object]:
    """Convert ``src_path`` to ``dst_path`` in the requested format.

    ``fmt`` is one of: wav, aiff, raw, mp3, ogg, flac, m4a.

    Returns an ``info`` dict describing the output (``sr``, ``channels``,
    ``frames`` when known, ``sample_width`` in bytes).  The info is what the
    library index stores as metadata — essential for ``raw`` files, which
    carry no header that could be probed later.
    """
    # Normalise the source to a WAV first (in case it is compressed).
    src_wav = src_path if src_path.lower().endswith(".wav") else decode_with_ffmpeg(src_path)

    if fmt == "wav":
        _convert_wav(src_wav, dst_path, dst_sr, sample_width, channels)
        try:
            with WavReader(dst_path) as r:
                return {"sr": r.sr, "channels": r.channels, "frames": r.nframes,
                        "sample_width": WAV_SAMPLE_WIDTHS.get(sample_width, 2)}
        except wave.Error:
            # stdlib ``wave`` rejects IEEE-float WAVs (fmt tag 3); parse the
            # canonical header manually instead.
            sr, ch, sw, frames = _probe_wav_header(dst_path)
            return {"sr": sr, "channels": ch, "frames": frames,
                    "sample_width": sw or WAV_SAMPLE_WIDTHS.get(sample_width, 2)}
    if fmt == "aiff":
        _convert_aiff(src_wav, dst_path, dst_sr, channels)
        with WavReader(src_wav) as r:
            return {"sr": dst_sr or r.sr, "channels": channels or r.channels,
                    "frames": 0, "sample_width": 2}
    if fmt == "raw":
        return _convert_raw(src_wav, dst_path, dst_sr, channels)
    if fmt in ("mp3", "ogg", "flac", "m4a"):
        # ffmpeg handles resample/channel mixdown itself via the output spec.
        tmp = src_wav
        if dst_sr or channels:
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            _convert_wav(src_wav, tmp, dst_sr, "pcm16", channels)
        encode_with_ffmpeg(tmp, dst_path, fmt, bitrate)
        with WavReader(tmp) as r:
            return {"sr": r.sr, "channels": r.channels, "frames": 0,
                    "sample_width": 2}
    raise ValueError(f"unknown format {fmt}")


def _adapt_channels(chunk: List[List[float]], dst_ch: int) -> List[List[float]]:
    """Match a decoded chunk to ``dst_ch`` channels (mono->stereo duplicate,
    multi->mono mixdown, extra channels dropped)."""
    src_ch = len(chunk)
    if src_ch == dst_ch:
        return chunk
    if dst_ch == 1:
        return [to_mono(chunk)]
    if src_ch == 1:
        return [list(chunk[0]) for _ in range(dst_ch)]
    return [list(chunk[i]) if i < src_ch else list(chunk[0]) for i in range(dst_ch)]


def _convert_wav(src_wav: str, dst_path: str, dst_sr: Optional[int],
                 sample_width: str, channels: Optional[int]) -> str:
    sw = WAV_SAMPLE_WIDTHS.get(sample_width, 2)
    with WavReader(src_wav) as r:
        sr = dst_sr or r.sr
        ch = channels or r.channels
        need_resample = bool(dst_sr and dst_sr != r.sr)
        resamplers = [dsp.StreamingResampler(r.sr, sr) for _ in range(ch)]
        with WavWriter(dst_path, sr, ch, sw) as out:
            for chunk in r.iter_chunks():
                c = _adapt_channels([list(x) for x in chunk], ch)
                if need_resample:
                    out_ch = []
                    for i, rs in enumerate(resamplers):
                        rs.push(c[i])
                        out_ch.append(rs.pull(len(c[i])))
                    c = out_ch
                out.write_chunk(c)
            # Flush the resampler's trailing samples.
            if need_resample:
                tail = [rs.flush(1 << 20) for rs in resamplers]
                if any(tail):
                    m = min(len(t) for t in tail)
                    out.write_chunk([t[:m] for t in tail])
    return dst_path


def _convert_aiff(src_wav: str, dst_path: str, dst_sr: Optional[int],
                  channels: Optional[int]) -> str:
    # ffmpeg is the simplest reliable AIFF encoder; fall back to copying via
    # a temporary WAV first if ffmpeg is missing.
    if has_ffmpeg():
        extra = []
        if dst_sr:
            extra += ["-ar", str(dst_sr)]
        if channels:
            extra += ["-ac", str(channels)]
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", src_wav]
                       + extra + [dst_path], check=True, capture_output=True, timeout=600)
        return dst_path
    raise RuntimeError("ffmpeg required for AIFF export")


def _convert_raw(src_wav: str, dst_path: str, dst_sr: Optional[int],
                 channels: Optional[int]) -> Dict[str, object]:
    """Write headerless 16-bit little-endian PCM.

    Because no metadata survives on disk, the output parameters are returned
    to the caller for library indexing.
    """
    frames_out = 0
    with WavReader(src_wav) as r:
        sr = dst_sr or r.sr
        ch = channels or r.channels
        need_resample = bool(dst_sr and dst_sr != r.sr)
        resamplers = [dsp.StreamingResampler(r.sr, sr) for _ in range(ch)]
        with open(dst_path, "wb") as f:
            def _emit(c: Sequence[Sequence[float]]) -> None:
                nonlocal frames_out
                inter = []
                n = min(len(x) for x in c)
                for i in range(n):
                    for x in c:
                        inter.append(x[i])
                f.write(_encode_pcm_16(inter))
                frames_out += n

            for chunk in r.iter_chunks():
                c = _adapt_channels([list(x) for x in chunk], ch)
                if need_resample:
                    out_ch = []
                    for i, rs in enumerate(resamplers):
                        rs.push(c[i])
                        out_ch.append(rs.pull(len(c[i])))
                    c = out_ch
                _emit(c)
            # Flush the resampler's trailing samples.
            if need_resample:
                tail = [rs.flush(1 << 20) for rs in resamplers]
                if any(tail):
                    m = min(len(t) for t in tail)
                    _emit([t[:m] for t in tail])
    return {"sr": sr, "channels": ch, "frames": frames_out, "sample_width": 2}
