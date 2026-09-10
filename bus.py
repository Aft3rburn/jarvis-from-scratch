"""
Tiny status bus for Jarvis's own visualizer (visualizer/server.py, a
separate copy of Mary's ai-visualizer engine - see Local Mary Clone
Session 32 in the vault). Same plain-text-files contract Mary's own
backtalk already uses, just pointed at Jarvis's own bus folder instead
of shared with anything else.

Deliberately dumb: each write is a small, fast, best-effort file write.
Never let a visualizer glitch break an actual conversation turn - every
function here swallows its own errors.
"""

import json
import pathlib
import time

BUS_DIR = pathlib.Path(__file__).parent / "bus"

VALID_STATES = {"idle", "listening", "thinking", "speaking"}


def _write(name: str, content: str) -> None:
    try:
        BUS_DIR.mkdir(parents=True, exist_ok=True)
        (BUS_DIR / name).write_text(content, encoding="utf-8")
    except OSError:
        pass  # the visualizer is a nice-to-have, never worth crashing a turn over


def set_state(state: str) -> None:
    if state not in VALID_STATES:
        state = "idle"
    _write(".voice_state", state)


def set_caption(text: str) -> None:
    """What Jarvis is currently saying - shown on the face while speaking."""
    _write(".voice_caption", json.dumps({"text": text}))


def set_usertext(text: str) -> None:
    """What Mark just said/typed - shown on the face while thinking."""
    _write(".voice_usertext", json.dumps({"text": text}))


def set_waveform(samples) -> None:
    """A snapshot of recent audio amplitude (up to 64 floats) for the
    face's audio-reactive animation. Stale after WAVEFORM_STALE_S
    (server.py's own constant) so a crashed speak() doesn't leave the
    face looking like it's talking forever."""
    _write(".voice_waveform", json.dumps({"ts": time.time(), "samples": list(samples)}))
