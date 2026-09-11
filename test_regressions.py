"""
Regression suite for known past Jarvis failure modes — item 2 of the
2026-09-10 four-part roadmap (see 'Local Mary Clone' Session 38 in Mark's
Mary vault, 06 - Resources).

Every incident this week got fixed one at a time, live, after it broke -
the PTT crash, the memory-self-poisoning bug, and the fabrication give-up
gate each ate a real debugging session. This file exists so a future code
change gets checked against every one of those failure shapes *before* it
ships, instead of relying on someone noticing a regression live again.

Stdlib only (unittest + unittest.mock), matching this project's existing
"no framework" style. Run with:

    python test_regressions.py

Deterministic and offline throughout - every scenario that would
otherwise need a real Ollama call is monkeypatched with scripted
responses instead, same pattern already proven in the (deleted)
Session 37 proof scripts this suite makes permanent.
"""

import ast
import inspect
import unittest
from unittest.mock import patch

import agent
import tools


# ---------------------------------------------------------------------------
# Regression 1: voice.py's listen_ptt scope bug (Local Mary Clone Session 36,
# 2026-09-09). Root cause: on_press declared `global press_processed` when
# no module-level press_processed exists - Python silently bound the name to
# an empty global namespace instead of the enclosing function's real
# variable, raising NameError the first time a key was actually pressed.
# Fix was `global` -> `nonlocal`. Can't exercise the real PTT flow headless
# (needs a live keyboard listener + mic), so this checks the actual AST of
# the shipped function instead of re-implementing the closure and testing
# the copy - a static check of the real source is what actually catches a
# reintroduced bug, not the same bug in a hand-copied stand-in.
# ---------------------------------------------------------------------------
class TestPTTScopeBug(unittest.TestCase):
    def test_on_press_uses_nonlocal_not_global_for_press_processed(self):
        import voice  # imported lazily: heavy audio/CUDA deps, only needed here

        source = inspect.getsource(voice.listen_ptt)
        tree = ast.parse(source)

        on_press = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "on_press":
                on_press = node
                break
        self.assertIsNotNone(
            on_press, "listen_ptt no longer defines a nested on_press - "
            "update this test if the function was renamed or restructured."
        )

        globals_declared = {
            name
            for n in ast.walk(on_press)
            if isinstance(n, ast.Global)
            for name in n.names
        }
        nonlocals_declared = {
            name
            for n in ast.walk(on_press)
            if isinstance(n, ast.Nonlocal)
            for name in n.names
        }

        self.assertNotIn(
            "press_processed", globals_declared,
            "on_press declares 'global press_processed' again - this is "
            "the exact 2026-09-09 bug (NameError: no module-level "
            "press_processed exists, listen_ptt's local gets shadowed). "
            "Must be 'nonlocal', not 'global'."
        )
        self.assertIn(
            "press_processed", nonlocals_declared,
            "on_press no longer declares 'nonlocal press_processed' - "
            "without it, press_processed re-binds a fresh local instead "
            "of updating listen_ptt's real flag, silently breaking the "
            "single-press guard."
        )


# ---------------------------------------------------------------------------
# Regression 2: the LESSONS.md / daily-note self-poisoning bug (Local Mary
# Clone Session 26 addendum, 2026-09-09). Root cause: stuck making no real
# progress, the model repeatedly logged near-identical "still blocked"
# status entries - 20+ times in one case - which then got auto-loaded back
# into its own system prompt and parroted as fact. Fix: tools._write_guard,
# a near-duplicate check plus a rate-limit backstop on
# append_daily_note/append_lesson. These tests call _write_guard directly -
# a pure function, no file I/O - so they're fast and don't touch the real
# vault.
# ---------------------------------------------------------------------------
class TestMemorySelfPoisoningGuard(unittest.TestCase):
    def _parse_ts_always_none(self, header):
        # Rate-limit path isn't under test here; return None so only the
        # near-duplicate check is exercised.
        return None

    def test_blocks_near_duplicate_entry(self):
        existing = (
            "# Lessons Learned\n\n"
            "## 2026-09-09 10:00 — ollama\n"
            "Still blocked on the same tool call, not sure why it keeps "
            "failing.\n"
        )
        new_text = "Still blocked on the same tool call, unsure why it fails."
        result = tools._write_guard(existing, new_text, self._parse_ts_always_none)
        self.assertTrue(
            result.startswith("BLOCKED"),
            "a near-duplicate entry should be refused, matching the real "
            "2026-09-09 spam incident - got: " + repr(result),
        )

    def test_allows_genuinely_distinct_entry(self):
        existing = (
            "# Lessons Learned\n\n"
            "## 2026-09-09 10:00 — ollama\n"
            "ollama list auto-launches the Ollama server if it's not "
            "already running.\n"
        )
        new_text = "schedule_task must route through a generated .bat wrapper - an inline schtasks command line silently fails under Task Scheduler on this box."
        result = tools._write_guard(existing, new_text, self._parse_ts_always_none)
        self.assertEqual(
            result, "",
            "a genuinely distinct lesson must not be blocked - the guard "
            "exists to stop spam, not legitimate new entries. Got: " + repr(result),
        )

    def test_blocks_rate_limit_burst(self):
        import datetime

        now = datetime.datetime.now()
        # Three distinct-worded entries, all within the last 5 minutes -
        # distinct wording alone must not be enough to bypass the burst
        # limiter (the real incident's entries weren't textually identical
        # either, just repetitive in substance).
        headers = [
            (now - datetime.timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M"),
            (now - datetime.timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M"),
            (now - datetime.timedelta(minutes=3)).strftime("%Y-%m-%d %H:%M"),
        ]
        existing = "# Lessons Learned\n\n" + "".join(
            f"## {h} — topic{i}\nDistinct entry body number {i}, nothing alike.\n\n"
            for i, h in enumerate(headers)
        )

        def parse_ts(header):
            m = header.strip()
            try:
                return datetime.datetime.strptime(m.split(" — ")[0], "%Y-%m-%d %H:%M")
            except ValueError:
                return None

        new_text = "A fourth, also totally unrelated entry about something else entirely."
        result = tools._write_guard(existing, new_text, parse_ts)
        self.assertTrue(
            result.startswith("BLOCKED"),
            "3+ entries in 5 minutes should trip the rate-limit backstop "
            "even when wording differs each time - got: " + repr(result),
        )


# ---------------------------------------------------------------------------
# Regression 2b: self-incapacity claims sneaking into permanent memory
# through a normal, successful tool call (2026-09-10). A model turn logged
# "persistent failure to execute tool calls due to interface formatting
# limitations" straight into LESSONS.md via a completely ordinary
# append_lesson call - no malformed tool call happened first, so the
# give-up gate below (which only watches the spoken answer) never saw it.
# The claim was false. Fix: append_lesson/append_daily_note now reject a
# claim shaped like "tool calls / the interface can't do X" unless the
# call also passes verified=true.
# ---------------------------------------------------------------------------
class TestCapabilityClaimGuard(unittest.TestCase):
    REAL_FABRICATION = (
        "Encountered persistent failure to execute any tool calls due to "
        "interface formatting limitations. This is a limitation in how "
        "the interface handles tool execution rather than an issue with "
        "the tools themselves."
    )

    def test_blocks_unverified_self_incapacity_claim(self):
        result = tools._capability_claim_guard(self.REAL_FABRICATION, False)
        self.assertTrue(
            result.startswith("BLOCKED"),
            "the exact real 2026-09-10 fabrication text must be blocked "
            "when unverified - got: " + repr(result),
        )

    def test_allows_same_claim_when_verified(self):
        result = tools._capability_claim_guard(self.REAL_FABRICATION, True)
        self.assertEqual(
            result, "",
            "verified=true must let a capability claim through - the gate "
            "requires evidence, not a permanent ban on the topic. Got: "
            + repr(result),
        )

    def test_allows_unrelated_technical_lesson(self):
        # Real, legitimate lesson from this same file - not about the
        # agent's own tool-calling ability, must never be caught.
        text = (
            "rocm-smi doesn't exist on Windows at all - rewrote "
            "check_gpu_amd around Windows perf counters instead."
        )
        result = tools._capability_claim_guard(text, False)
        self.assertEqual(
            result, "",
            "a real hardware/tooling lesson unrelated to the agent's own "
            "tool-calling must not be blocked - got: " + repr(result),
        )

    def test_append_lesson_rejects_unverified_claim_end_to_end(self):
        with patch.object(tools, "_ensure_lessons_file") as mock_ensure, \
             patch("pathlib.Path.read_text", return_value="# Lessons Learned\n"), \
             patch("pathlib.Path.open") as mock_open:
            mock_ensure.return_value = tools.LESSONS_FILE
            result = tools.append_lesson({"lesson": self.REAL_FABRICATION})
        self.assertTrue(result.startswith("BLOCKED"))
        mock_open.assert_not_called()

    # Real 2026-09-11 incident: a false "Unable to execute tool calls..."
    # lesson got past the guard above (negation word BEFORE the capability
    # phrase, the original regex only matched the other order), poisoned
    # LESSONS.md, and triggered a real fabrication spiral on Mark's very
    # next message. Fixed by matching both word orders - this is the exact
    # text that slipped through before the fix.
    REAL_FABRICATION_NEGATION_FIRST = (
        "Unable to execute tool calls for show_face and list_faces due to "
        "a persistent formatting issue in the agent's response parsing. "
        "The tool calls are properly formatted but fail during execution."
    )

    def test_blocks_negation_first_phrasing(self):
        result = tools._capability_claim_guard(
            self.REAL_FABRICATION_NEGATION_FIRST, False
        )
        self.assertTrue(
            result.startswith("BLOCKED"),
            "the exact real 2026-09-11 fabrication text (negation word "
            "before the capability phrase) must be blocked when unverified "
            "- got: " + repr(result),
        )

    def test_allows_negation_first_phrasing_when_verified(self):
        result = tools._capability_claim_guard(
            self.REAL_FABRICATION_NEGATION_FIRST, True
        )
        self.assertEqual(result, "")


# ---------------------------------------------------------------------------
# Regression 3: the fabrication give-up gate (Local Mary Clone Session 37,
# 2026-09-09). Root cause: asked to bring up its face, the model garbled
# show_face's tool call three times, then gave up and invented an
# "interface limitations" excuse instead of retrying or admitting it didn't
# know why. Fix: agent._agentic_turn now forces up to 2 extra retries after
# a malformed attempt, then overwrites both the spoken answer AND stored
# message history with an honest fallback if it still can't produce a real
# tool call. These three scenarios are the same three Session 37 proved
# live with throwaway scripts - made permanent here instead of deleted
# after one pass.
# ---------------------------------------------------------------------------
class _ScriptedOllama:
    """Returns each response in `responses` in order; raises loudly (not
    silently) if called more times than scripted, same fail-loud property
    Session 37's own proof scripts had."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def __call__(self, messages, allowed_tools=None, model=None, url=None):
        if not self._responses:
            raise AssertionError(
                f"_ScriptedOllama exhausted after {self.calls} calls - "
                "the agentic loop made more model calls than this "
                "scenario expected."
            )
        self.calls += 1
        return {"message": self._responses.pop(0)}


def _malformed_text_message(tool_name="show_face"):
    return {"role": "assistant", "content": f"<function={tool_name}></function>"}


def _plain_text_message(text):
    return {"role": "assistant", "content": text}


def _real_tool_call_message(tool_name, args=None):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": tool_name, "arguments": args or {}}}
        ],
    }


class TestFabricationGiveUpGate(unittest.TestCase):
    def test_fabricated_giveup_gets_overwritten_honestly(self):
        """One garbled attempt, then it gives up and fabricates an excuse
        - MAX_GIVE_UP_RETRIES (2) more chances get forced, and only once
        that budget is spent does the honest fallback take over. Matches
        agent.py's actual retry math (verified by running this test
        against a wrong call count first - see the failed first draft):
        malformed -> continue; 1st give-up plain text -> retries=1,
        continue; 2nd give-up plain text -> retries=2, continue; 3rd
        give-up plain text -> retries>=MAX, overwrite. 4 calls total."""
        excuse = (
            "I cannot execute the tool call due to interface "
            "limitations in this environment."
        )
        scripted = _ScriptedOllama([
            _malformed_text_message(),   # attempt 1: garbled tool-call syntax
            _plain_text_message(excuse),  # give-up 1/2: forced to retry
            _plain_text_message(excuse),  # give-up 2/2: forced to retry
            _plain_text_message(excuse),  # give-up budget spent: overwritten
        ])
        messages = [{"role": "user", "content": "bring up your face"}]
        with patch.object(agent, "call_ollama", scripted):
            agent._agentic_turn(messages, speak_answer=False)

        final = messages[-1]["content"]
        self.assertNotIn(
            "interface limitations", final,
            "the fabricated excuse must not survive into stored history - "
            "this is the exact split-brain bug Session 37 caught (spoken "
            "answer honest, stored history still held the lie)."
        )
        self.assertIn(
            "don't know why", final.lower(),
            "expected the honest fallback line, got: " + repr(final),
        )
        self.assertEqual(scripted.calls, 4)

    def test_self_correction_after_one_garble_passes_through(self):
        """Garbles once, then self-corrects into a REAL tool call - must
        sail through untouched, not get caught by the give-up gate."""
        scripted = _ScriptedOllama([
            _malformed_text_message(),                       # attempt 1: garbled
            _real_tool_call_message("show_face"),              # retry: real call this time
            _plain_text_message("Your face is up now."),        # normal summary after
        ])
        messages = [{"role": "user", "content": "bring up your face"}]
        with patch.object(agent, "call_ollama", scripted), \
             patch.dict(tools.REGISTRY, {"show_face": lambda args: "OK: face shown"}):
            agent._agentic_turn(messages, speak_answer=False)

        final = messages[-1]["content"]
        self.assertEqual(
            final, "Your face is up now.",
            "a genuine self-correction followed by a normal summary must "
            "not be touched by the give-up gate - got: " + repr(final),
        )

    def test_normal_answer_with_zero_malformed_attempts_untouched(self):
        """No malformed attempts at all - the gate must never fire."""
        scripted = _ScriptedOllama([
            _plain_text_message("It's currently 3:15 PM."),
        ])
        messages = [{"role": "user", "content": "what time is it"}]
        with patch.object(agent, "call_ollama", scripted):
            agent._agentic_turn(messages, speak_answer=False)

        self.assertEqual(messages[-1]["content"], "It's currently 3:15 PM.")


# ---------------------------------------------------------------------------
# Regression 4: the startup-sitrep tool-call spiral (Local Mary Clone
# Session 23, 2026-09-09). Root cause: with every tool still on the table,
# the sitrep turn spiraled through 15+ tool calls (re-searching the vault,
# logging redundant lessons) instead of answering directly. Fix:
# _startup_sitrep() now calls _agentic_turn with allowed_tools=set() - no
# tools offered at all. This test covers the defense-in-depth half of that
# fix: even if a local model hallucinates a tool call despite an empty
# schema list (a real local-model quirk, not hypothetical - see the
# malformed-call incidents above), dispatch must reject it via the
# existing "not permitted" path instead of crashing or silently running it.
# ---------------------------------------------------------------------------
class TestZeroToolsEnforcement(unittest.TestCase):
    def test_disallowed_tool_call_rejected_not_executed(self):
        run_marker = {"called": False}

        def _spy_search_vault(args):
            run_marker["called"] = True
            return "should never run"

        scripted = _ScriptedOllama([
            _real_tool_call_message("search_vault", {"query": "test"}),
            _plain_text_message("Everything's quiet, nothing open."),
        ])
        messages = [{"role": "user", "content": "give me a sitrep"}]
        with patch.object(agent, "call_ollama", scripted), \
             patch.dict(tools.REGISTRY, {"search_vault": _spy_search_vault}):
            agent._agentic_turn(messages, speak_answer=False, allowed_tools=set())

        self.assertFalse(
            run_marker["called"],
            "a tool call outside allowed_tools must never actually "
            "execute, even if the model produces one - it should be "
            "rejected with an error result and fed back, not run.",
        )
        self.assertEqual(messages[-1]["content"], "Everything's quiet, nothing open.")


class TestOpsMonitorInputHardening(unittest.TestCase):
    """Two real bugs caught live 2026-09-10 wiring the ops-monitor tools
    into the fast-lane router: the small model garbled a tool argument,
    and misread an unlabeled tool result. Both are model-reliability
    failures, not one-off flukes - worth locking down so a future change
    to either function doesn't quietly bring them back."""

    def test_check_disk_normalizes_garbled_drive_arg(self):
        """llama3.2:3b, asked to check disk space, called check_disk with
        drive='C\\' (letter + backslash, no colon) - it garbled the
        schema's escaped 'C:\\\\' example. Must resolve to the real C:
        drive instead of erroring out."""
        import ops_monitor
        result = ops_monitor.check_disk({"drive": "C\\"})
        self.assertNotIn("ERROR", result)
        self.assertIn("free", result)

    def test_check_gpu_nvidia_output_is_labeled_not_bare_csv(self):
        """The same model, given nvidia-smi's bare CSV output with no
        field labels, answered 'running at 75% temperature' - it mixed
        up the utilization and temperature columns. check_gpu_nvidia's
        real output must never regress back to bare unlabeled numbers a
        weak model can misread."""
        import ops_monitor
        with patch.object(
            ops_monitor, "_run_ps",
            return_value="NVIDIA RTX A1000, 44, 87, 4732, 8188",
        ):
            result = ops_monitor.check_gpu_nvidia({})
        self.assertIn("temperature 44C", result)
        self.assertIn("utilization 87%", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
