"""
Voice front end for Jarvis From Scratch — v1 addition on top of the v0 text
loop. Local only: faster-whisper on the Nvidia RTX A1000 (CUDA) for
speech-to-text, Kokoro on CPU for text-to-speech. No cloud calls.

listen()  -> records from the default mic until a beat of silence, then
             transcribes and returns the text.
speak(text) -> synthesizes text with Kokoro and plays it back.
"""

import os
import queue
import re
import threading
import time
from pathlib import Path

# faster-whisper's CUDA backend (ctranslate2) needs cuBLAS/cuDNN, which
# aren't in this machine's system PATH (no full CUDA Toolkit installed,
# just the driver). The nvidia-cublas-cu12/nvidia-cudnn-cu12 pip packages
# ship the DLLs inside the venv itself - Windows' DLL search order checks
# PATH, so prepend their bin folders before anything CUDA-related loads.
# (os.add_dll_directory alone does NOT work here - ctranslate2's own
# native LoadLibrary calls don't go through Python's DLL-directory
# registry, only through the OS-level PATH search. Proven the hard way,
# 2026-09-08.)
_site_packages = Path(__file__).parent / ".venv" / "Lib" / "site-packages"
_cuda_dll_dirs = [
    _site_packages / "nvidia" / pkg / "bin"
    for pkg in ("cublas", "cudnn", "cuda_nvrtc")
]
os.environ["PATH"] = (
    os.pathsep.join(str(d) for d in _cuda_dll_dirs if d.exists())
    + os.pathsep
    + os.environ["PATH"]
)

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard as pynput_keyboard

# Kokoro phonemizes through espeak-ng. espeakng-loader ships its own DLL +
# data dir (no system eSpeak NG install needed, unlike Piper's approach) -
# point phonemizer at it before kokoro is ever imported. Proven live
# 2026-09-08.
import espeakng_loader
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeakng_loader.get_library_path()
os.environ["PHONEMIZER_ESPEAK_PATH"] = espeakng_loader.get_data_path()
from kokoro import KPipeline

SAMPLE_RATE = 16000  # whisper's native rate
BLOCK_MS = 30
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_MS / 1000)
SILENCE_RMS = 0.01       # below this = silence
SILENCE_HANG_MS = 1200   # stop after this much trailing silence
MAX_RECORD_SECONDS = 30

# Push-to-talk key. Not F10 (Mary's own backtalk PTT key), not F9
# (already bound to something else), not Scroll Lock (missing on a lot of
# laptop keyboards). Mark's pick: Left Ctrl.
PTT_KEY = pynput_keyboard.Key.ctrl_l

# bm_fable - Kokoro British male voice. Chosen 2026-09-09 after Mark asked
# for something more youthful, mid-30s British male; he A/B'd all four
# Kokoro British male voices (daniel/fable/george/lewis) played back to
# back through the speakers and picked fable by ear - Kokoro publishes no
# age/character metadata, so this wasn't guessed from documentation.
# Previously bm_lewis (Mary's own backtalk voice, swapped in 2026-09-08);
# before that, Piper's en_GB-alan-medium (still on disk in voices/, unused).
KOKORO_VOICE = "bm_fable"
KOKORO_LANG = KOKORO_VOICE[0]  # voice name's first letter = language pipeline
KOKORO_RATE = 24000

_whisper_model = None
_kokoro_pipe = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel("base", device="cuda", compute_type="float16")
    return _whisper_model


def _get_kokoro():
    global _kokoro_pipe
    if _kokoro_pipe is None:
        _kokoro_pipe = KPipeline(lang_code=KOKORO_LANG)
    return _kokoro_pipe


def _record_until_silence() -> np.ndarray:
    """Record from the default mic until SILENCE_HANG_MS of quiet follows
    speech, or MAX_RECORD_SECONDS is hit. Returns float32 mono audio."""
    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(indata.copy())

    chunks = []
    speech_started = False
    silence_ms = 0
    start = time.time()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=callback,
    ):
        print("listening...")
        while True:
            block = q.get()
            chunks.append(block)
            rms = float(np.sqrt(np.mean(block**2)))

            if rms > SILENCE_RMS:
                speech_started = True
                silence_ms = 0
            elif speech_started:
                silence_ms += BLOCK_MS
                if silence_ms >= SILENCE_HANG_MS:
                    break

            if time.time() - start > MAX_RECORD_SECONDS:
                break

    audio = np.concatenate(chunks, axis=0).flatten()
    return audio


def listen() -> str:
    """Record from the mic (auto-stop on silence) and return the
    transcribed text."""
    audio = _record_until_silence()
    if audio.size < SAMPLE_RATE * 0.3:  # less than ~0.3s, nothing said
        return ""
    return _transcribe(audio)


def listen_ptt(key=PTT_KEY) -> str:
    """Push-to-talk: block until `key` is pressed, record while it's held,
    stop and transcribe on release."""
    print(f"hold {key} to talk...")

    pressed = threading.Event()
    released = threading.Event()
    frames = []

    def on_press(k):
        if k == key:
            pressed.set()

    def on_release(k):
        if k == key:
            released.set()
            return False  # stop this listener once the key comes back up

    listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    pressed.wait()

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=callback,
    )
    stream.start()
    print("recording (release key to stop)...")
    released.wait()
    stream.stop()
    stream.close()
    listener.join()

    if not frames:
        return ""
    audio = np.concatenate(frames, axis=0).flatten()
    if audio.size < SAMPLE_RATE * 0.2:
        return ""
    print("transcribing...")
    return _transcribe(audio)


def _transcribe(audio: np.ndarray) -> str:
    model = _get_whisper()
    segments, _info = model.transcribe(audio, language="en")
    return " ".join(seg.text.strip() for seg in segments).strip()


# Markdown syntax the model sometimes writes even when the answer is
# headed straight to Piper - none of it means anything spoken aloud, and
# left in, Piper reads the literal symbols (asterisks, pipes, backticks)
# as if they were words. Stripped here, once, so every call site that
# speaks gets clean text without having to remember to clean it itself.
_MD_CODE_FENCE_RE = re.compile(r"```.*?```", re.S)
_MD_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_MD_BOLD_ITALIC_RE = re.compile(r"(\*\*\*|\*\*|\*|__)(.*?)\1")
_MD_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_MD_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.M)
_MD_NUMBERED_RE = re.compile(r"^\s*\d+\.\s+", re.M)
_MD_TABLE_SEP_RE = re.compile(r"^[\s|:-]+$", re.M)
_MD_TABLE_ROW_RE = re.compile(r"^\|(.*)\|\s*$", re.M)


def strip_for_speech(text: str) -> str:
    """Clean model output down to plain conversational text before it
    reaches Piper - strip markdown that would otherwise get read aloud
    as literal symbols, and collapse whitespace into natural sentence
    spacing instead of a run-on."""
    text = _MD_CODE_FENCE_RE.sub(" (code omitted) ", text)
    text = _MD_TABLE_SEP_RE.sub("", text)
    text = _MD_TABLE_ROW_RE.sub(lambda m: m.group(1).replace("|", ", "), text)
    text = _MD_INLINE_CODE_RE.sub(r"\1", text)
    text = _MD_BOLD_ITALIC_RE.sub(r"\2", text)
    text = _MD_HEADER_RE.sub("", text)
    text = _MD_BULLET_RE.sub("", text)
    text = _MD_NUMBERED_RE.sub("", text)
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n", ". ", text)
    # Stripping symbols above tends to leave doubled/dangling punctuation
    # (empty table cells, a header's trailing colon meeting a sentence's
    # period) - clean that up so pauses land where a sentence actually
    # ends, not after every stray leftover mark.
    text = re.sub(r"\s+([,.])", r"\1", text)
    text = re.sub(r"([,.])\1+", r"\1", text)
    text = re.sub(r",\s*\.", ".", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def speak(text: str) -> None:
    """Clean and synthesize text with Kokoro, then play it back."""
    text = strip_for_speech(text)
    if not text.strip():
        return

    pipe = _get_kokoro()
    audio_pieces = [
        np.asarray(audio, dtype=np.float32)
        for _, _, audio in pipe(text, voice=KOKORO_VOICE, speed=1.0)
    ]
    if not audio_pieces:
        return
    audio = np.concatenate(audio_pieces)

    sd.play(audio, samplerate=KOKORO_RATE)
    sd.wait()
