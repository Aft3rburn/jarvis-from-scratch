"""
Standalone popup for live-tuning Jarvis's tone. Separate from agent.py on
purpose - Mark wants a window he can pop open, nudge a slider, and close,
without touching a running chat/voice session at all.

Five sliders, not the full twenty-attribute rubric this was scoped from
(see Local Mary Clone Session 28/29 in the vault for why): a local 30B
model doesn't reliably produce different output for twenty overlapping
"slightly X" distinctions, so each slider here collapses several
overlapping rubric axes into one lever that actually changes something
audible. Each slider writes straight to tone_config.json on release -
agent.py rereads that file and rebuilds the tone portion of the system
prompt on every turn, so a change here takes effect on Jarvis's very
next reply, no restart needed.

Jarvis can also adjust these himself now (Session 31: a set_tone tool,
so "dial the sarcasm down by 50%" doesn't need Mark to touch this
window at all) - this popup polls the config file for outside changes
and updates its own slider positions to match, so it never shows a
stale value if Jarvis moved something while this was already open.
"""

import json
import pathlib
import tkinter as tk
from tkinter import ttk

CONFIG_PATH = pathlib.Path(__file__).parent / "tone_config.json"

DEFAULTS = {
    "formality": 6,
    "warmth": 6,
    "conciseness": 8,
    "humor": 2,
    "confidence": 7,
}

# (key, display name, low-pole label, high-pole label)
SLIDERS = [
    ("formality", "Formality", "Casual", "Formal"),
    ("warmth", "Warmth", "Detached", "Warm"),
    ("conciseness", "Conciseness", "Elaborate", "Concise"),
    ("humor", "Humor / Sarcasm", "Serious", "Playful"),
    ("confidence", "Confidence", "Suggestive", "Authoritative"),
]


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return {**DEFAULTS, **{k: v for k, v in data.items() if k in DEFAULTS}}
        except (json.JSONDecodeError, OSError):
            pass
    return dict(DEFAULTS)


def save_config(values: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


class ToneControl(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Jarvis Tone Control")
        self.resizable(False, False)
        self.values = load_config()
        self.vars = {}
        self._suppress_write = False
        self._last_mtime = self._config_mtime()

        pad = {"padx": 12, "pady": (10, 0)}
        tk.Label(
            self,
            text="Adjust live - takes effect on Jarvis's next reply, no restart.",
            fg="#555",
        ).grid(row=0, column=0, columnspan=3, padx=12, pady=(10, 4), sticky="w")

        row = 1
        for key, name, low_label, high_label in SLIDERS:
            var = tk.DoubleVar(value=self.values.get(key, DEFAULTS[key]))
            self.vars[key] = var

            tk.Label(self, text=name, font=("Segoe UI", 10, "bold")).grid(
                row=row, column=0, columnspan=3, sticky="w", **pad
            )
            row += 1

            tk.Label(self, text=low_label, fg="#555").grid(
                row=row, column=0, padx=(12, 0), sticky="w"
            )
            scale = ttk.Scale(
                self,
                from_=0,
                to=10,
                orient="horizontal",
                variable=var,
                length=220,
                command=lambda _v, k=key: self._on_change(k),
            )
            scale.grid(row=row, column=1, pady=4)
            tk.Label(self, text=high_label, fg="#555").grid(
                row=row, column=2, padx=(0, 12), sticky="e"
            )
            row += 1

        btns = tk.Frame(self)
        btns.grid(row=row, column=0, columnspan=3, pady=12)
        tk.Button(btns, text="Reset to defaults", command=self._reset).pack(
            side="left", padx=6
        )
        tk.Button(btns, text="Close", command=self.destroy).pack(side="left", padx=6)

        self.after(1000, self._poll_external_changes)

    def _config_mtime(self):
        try:
            return CONFIG_PATH.stat().st_mtime
        except OSError:
            return None

    def _on_change(self, key: str) -> None:
        if self._suppress_write:
            return  # this move came from _poll_external_changes, not the user
        # Sliders are continuous but agent.py buckets each into
        # low/mid/high before turning it into prompt text anyway (a
        # local model can't reliably act on fine gradations), so round
        # here to keep tone_config.json itself readable.
        self.values[key] = round(self.vars[key].get())
        save_config(self.values)
        self._last_mtime = self._config_mtime()

    def _reset(self) -> None:
        self.values = dict(DEFAULTS)
        for key, var in self.vars.items():
            var.set(self.values[key])
        save_config(self.values)
        self._last_mtime = self._config_mtime()

    def _poll_external_changes(self) -> None:
        # Jarvis's own set_tone tool writes this same file. Without this,
        # the popup would keep showing whatever position it last drew
        # even after he moved a slider himself - a stale display Mark
        # would have no reason to distrust.
        mtime = self._config_mtime()
        if mtime is not None and mtime != self._last_mtime:
            self._last_mtime = mtime
            self.values = load_config()
            self._suppress_write = True
            for key, var in self.vars.items():
                var.set(self.values[key])
            self._suppress_write = False
        self.after(1000, self._poll_external_changes)


if __name__ == "__main__":
    ToneControl().mainloop()
