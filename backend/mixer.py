"""
mixer.py — Multi-track mixdown.

Mixes any number of tracks (each an audio file with a gain, pan and mute flag)
into a single stereo WAV.  Tracks may have different sample rates and lengths;
the mixer resamples on the fly with a seamless streaming resampler and pads
shorter tracks with silence.  Memory stays bounded because every track is read
a fixed-size chunk at a time.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence

from . import audio_io, dsp


def _pan_gains(pan: float) -> tuple:
    """Constant-power pan gains for a mono source (pan in [-1, 1])."""
    pan = max(-1.0, min(1.0, pan))
    angle = (pan + 1.0) * math.pi / 4.0
    return math.cos(angle), math.sin(angle)


def _balance_gains(pan: float, gain: float) -> tuple:
    """Stereo balance gains: pan < 0 attenuates right, pan > 0 attenuates left."""
    pan = max(-1.0, min(1.0, pan))
    if pan <= 0:
        return gain, gain * (1.0 + pan)
    return gain * (1.0 - pan), gain


def mixdown(tracks: Sequence[Dict], out_path: str, target_sr: Optional[int] = None,
            master_gain: float = 1.0, chunk: int = 1 << 15) -> Dict:
    """Mix ``tracks`` into ``out_path``.

    Each track is a dict: ``{"path", "gain", "pan", "muted"}``.
    """
    active = [t for t in tracks if t.get("path") and not t.get("muted")
              and os.path.isfile(t["path"])]
    if not active:
        raise ValueError("no active tracks to mix")

    readers = []
    for t in active:
        r = audio_io.WavReader(t["path"])
        readers.append((r, t))

    sr = target_sr or max(r.sr for r, _ in readers)
    # Streaming resamplers for tracks that need rate conversion.
    resamplers: List[Optional[dsp.StreamingResampler]] = []
    for r, _ in readers:
        resamplers.append(dsp.StreamingResampler(r.sr, sr) if r.sr != sr else None)

    total_frames = 0
    with audio_io.WavWriter(out_path, sr, 2, 2) as w:
        done = [False] * len(readers)
        while not all(done):
            out_l = [0.0] * chunk
            out_r = [0.0] * chunk
            actual = 0
            for i, (r, t) in enumerate(readers):
                if done[i]:
                    continue
                rs = resamplers[i]
                # Read enough source samples to yield ~chunk output frames.
                src_frames = max(1, int(chunk * r.sr / sr)) if rs else chunk
                raw = r.read_chunk(src_frames)
                if raw is None:
                    # Drain a resampler's trailing samples, if any.
                    if rs is not None:
                        tail = [rs.pull(chunk) for _ in range(r.channels)]
                        if any(tail):
                            raw = tail
                        else:
                            done[i] = True
                            continue
                    else:
                        done[i] = True
                        continue

                if rs is not None:
                    out_ch = []
                    for c, ch_data in enumerate(raw):
                        rs.push(ch_data)
                        out_ch.append(rs.pull(chunk))
                else:
                    out_ch = [list(x) for x in raw]

                gain = t.get("gain", 1.0)
                pan = t.get("pan", 0.0)
                n = min(len(x) for x in out_ch)
                actual = max(actual, n)
                if len(out_ch) == 1:
                    lg, rg = _pan_gains(pan)
                    mono = out_ch[0]
                    for j in range(n):
                        out_l[j] += mono[j] * gain * lg
                        out_r[j] += mono[j] * gain * rg
                else:
                    lg, rg = _balance_gains(pan, gain)
                    left, right = out_ch[0], out_ch[1]
                    for j in range(n):
                        out_l[j] += left[j] * lg
                        out_r[j] += right[j] * rg

            if actual == 0:
                break
            # Master gain + soft clipping to guard against overload.
            total_frames += actual
            out_l = [math.tanh(x * master_gain) for x in out_l[:actual]]
            out_r = [math.tanh(x * master_gain) for x in out_r[:actual]]
            w.write_chunk([out_l, out_r])

    for r, _ in readers:
        r.close()

    return {
        "tracks": len(active),
        "sr": sr,
        "duration": total_frames / sr if sr else 0.0,
        "frames": total_frames,
    }
