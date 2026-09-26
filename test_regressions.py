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
import tempfile
import pathlib
import json
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
        - the retry budget gets forced, and once spent the honest fallback
        takes over. Matches agent.py's actual retry math: malformed ->
        nudge (retries=1); excuse -> forced retry (retries=2); excuse ->
        budget spent (2>=MAX_GIVE_UP_RETRIES), overwrite. 3 calls total."""
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
            "show_face", final,
            "the honest fallback names the tool it was trying to call, so "
            "Mark knows what to retry - got: " + repr(final),
        )
        self.assertIn(
            "try asking me again", final.lower(),
            "the fallback must suggest a retry instead of dead-ending at "
            "'needs a person' - got: " + repr(final),
        )
        self.assertEqual(scripted.calls, 3)

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
# Regression 3b: the "explaining the format" false-positive spiral
# (2026-09-12). Mark asked "what is the correct way to format a tool
# call?" - a meta-question, not a request to do anything. The model
# answered correctly in prose, illustrating the shape with a generic
# placeholder ("<function=tool_name>..."). _FAKE_TOOL_CALL_RE fired on
# that placeholder anyway, telling the model its correct answer "wasn't a
# valid tool call" and demanding a retry it had no reason to make. It had
# nothing real to retry, so it spiraled for 30 steps inventing fake calls
# to real tools trying to satisfy a demand that made no sense, then gave
# up with a fabricated "I can't execute tool calls" excuse - and because
# every single step re-triggered the fake-call regex, give_up_retries
# never advanced past 0, so the existing give-up-overwrite gate never
# fired either (see TestUniversalCapabilityNet below for that half).
# Fix: only treat "<function=...>" as a real malformed attempt when the
# extracted name resolves to an actual registered tool - a placeholder
# like "tool_name" never does, because an explanation invents a name
# while a real attempt uses one that exists.
# ---------------------------------------------------------------------------
class TestExplanatoryAnswerNotNudged(unittest.TestCase):
    def test_placeholder_function_example_passes_through(self):
        """The exact 2026-09-12 shape: a correct, prose explanation of
        tool-call format using a placeholder name must be accepted as the
        final answer, not mistaken for a real malformed attempt."""
        explanation = (
            "The correct way to format a tool call is with an XML block "
            "that starts with <function=, followed by the tool name, and "
            "then its parameters within <parameter=> tags. It should "
            "look like this:\n\n<function=tool_name>\n"
            "<parameter=parameter_1>\nvalue_1\n</parameter>\n</function>\n\n"
            "Each parameter must be on its own line."
        )
        scripted = _ScriptedOllama([_plain_text_message(explanation)])
        messages = [
            {"role": "user", "content": "what is the correct way to format a tool call?"}
        ]
        with patch.object(agent, "call_ollama", scripted):
            agent._agentic_turn(messages, speak_answer=False)

        self.assertEqual(
            messages[-1]["content"], explanation,
            "an explanatory answer using a placeholder tool name must "
            "pass through untouched, not get nudged into a retry spiral "
            "it has no real tool call to retry - got: "
            + repr(messages[-1]["content"]),
        )
        self.assertEqual(
            scripted.calls, 1,
            "should answer in a single step - no retry was ever warranted",
        )

    def test_placeholder_name_not_confused_with_real_tool(self):
        """A real malformed attempt (genuine tool name) still gets nudged
        - this fix narrows the trigger, it doesn't disable it."""
        scripted = _ScriptedOllama([
            _malformed_text_message("show_face"),
            _real_tool_call_message("show_face"),
            _plain_text_message("Your face is up now."),
        ])
        messages = [{"role": "user", "content": "bring up your face"}]
        with patch.object(agent, "call_ollama", scripted), \
             patch.dict(tools.REGISTRY, {"show_face": lambda args: "OK: face shown"}):
            agent._agentic_turn(messages, speak_answer=False)

        self.assertEqual(messages[-1]["content"], "Your face is up now.")
        self.assertEqual(
            scripted.calls, 3,
            "a genuine malformed attempt (real tool name) must still be "
            "nudged to retry, not waved through like an explanation",
        )


# ---------------------------------------------------------------------------
# Regression 3c: the universal capability-claim net (2026-09-12), the other
# half of the same incident. Once the model spiraled (see above), every one
# of its 30 steps re-emitted fake-call syntax naming a real tool, so
# had_malformed_attempt stayed true but give_up_retries never left 0 - the
# specific state that has to exist for the give-up-overwrite check to fire.
# The loop hit MAX_STEPS still holding a fabricated "I can't execute tool
# calls" answer and spoke it verbatim, unfiltered. Fix: right before any
# final answer is printed/spoken, run it through the same
# tools._capability_claim_guard already trusted for append_lesson/
# append_daily_note - independent of whatever state the retry machinery
# above is in.
# ---------------------------------------------------------------------------
class TestUniversalCapabilityNet(unittest.TestCase):
    def test_maxsteps_exhausted_with_fabricated_claim_gets_overwritten(self):
        """Every step re-triggers the malformed-attempt path (so
        give_up_retries never advances) all the way to MAX_STEPS, and the
        very last step's content is a fabricated capability claim. Must
        still be replaced with the honest fallback, not spoken as-is."""
        fabrication = (
            "Cannot execute tool calls despite understanding the correct "
            "format. System interprets all attempts as plain text rather "
            "than actual executions."
        )
        scripted = _ScriptedOllama(
            [_malformed_text_message("run_shell") for _ in range(agent.MAX_STEPS - 1)]
            + [_plain_text_message(fabrication)]
        )
        messages = [{"role": "user", "content": "what is the correct way to format a tool call?"}]
        with patch.object(agent, "call_ollama", scripted):
            agent._agentic_turn(messages, speak_answer=False)

        final = messages[-1]["content"]
        self.assertNotIn(
            "Cannot execute tool calls", final,
            "the fabricated claim must never survive as the final answer, "
            "even when it arrives on the very last step of the loop's "
            "step budget - got: " + repr(final),
        )
        self.assertIn("run_shell", final)
        self.assertIn("try asking me again", final.lower())

    def test_honest_final_answer_at_maxsteps_untouched(self):
        """An honest, non-fabricating final answer must pass through the
        capability-claim net as-is - the net rewrites fabrications, not
        everything. (The 'fake-call spam all the way to MAX_STEPS' shape
        this test originally scripted can't occur anymore: the shared
        retry budget from the 2026-09-12 fix caps the spiral at 3 model
        calls, so a step-30 honest answer is unreachable. This keeps the
        test's intent - the net's precision - in a reachable scenario.)"""
        scripted = _ScriptedOllama(
            [_plain_text_message("Here's the disk usage you asked for.")]
        )
        messages = [{"role": "user", "content": "check disk space"}]
        with patch.object(agent, "call_ollama", scripted):
            agent._agentic_turn(messages, speak_answer=False)

        self.assertEqual(
            messages[-1]["content"], "Here's the disk usage you asked for."
        )


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


class TestToolErrorStreakGuard(unittest.TestCase):
    """Real incident, 2026-09-23: asked to add GPU utilization lines to a
    face, set_face_tunable kept returning an ERROR telling the model to
    ask Mark which option he meant, and the model just retried with
    near-identical bad arguments 10 straight times instead of ever
    asking. No guard existed for a well-formed call that keeps failing,
    only for malformed-format calls (see TestFabricationGiveUpGate).
    Fix: a tool that errors MAX_TOOL_ERROR_RETRIES+1 times in a row gets
    pulled from allowed_tools for the rest of the turn."""

    def test_repeated_tool_error_pulls_tool_and_forces_a_different_path(self):
        error_msg = (
            "ERROR: nothing here clearly matches what Mark said. "
            "Options: SQUASH = .32; MIC_N = 32. Nothing changed. "
            "Ask Mark which one he means."
        )
        scripted = _ScriptedOllama([
            _real_tool_call_message("set_face_tunable", {"value": "half"}),  # streak 1
            _real_tool_call_message("set_face_tunable", {"value": "half"}),  # streak 2
            _real_tool_call_message("set_face_tunable", {"value": "half"}),  # streak 3: tool pulled
            _plain_text_message(
                "I'm not sure which setting you mean - could you say "
                "which one specifically?"
            ),
        ])
        messages = [
            {"role": "user", "content": "add GPU utilization lines to this face"}
        ]
        with patch.object(agent, "call_ollama", scripted), \
             patch.dict(tools.REGISTRY, {"set_face_tunable": lambda args: error_msg}):
            result = agent._agentic_turn(messages, speak_answer=False)

        self.assertEqual(
            scripted.calls, 4,
            "expected exactly 3 failed tool calls then one plain-text "
            "answer - not 10 straight retries like the real incident.",
        )
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        self.assertEqual(
            len(tool_messages), 3,
            "the tool must stop being called after its error budget is "
            "spent, not keep executing indefinitely.",
        )
        self.assertIn(
            "which one", result["answer"].lower(),
            "once the tool's pulled, the model's final answer should "
            "surface the real ambiguity instead of another blind guess - "
            "got: " + repr(result["answer"]),
        )


class TestReadBeforeFaceEditGate(unittest.TestCase):
    """Real pattern across nearly every face-editing incident on record
    (the circuit clobber 2026-09-18, the aether->blue_board
    restyle-vs-switch 2026-09-17, the swarm-doubling edit that hit the
    wrong file entirely 2026-09-20): the model acted on a face's file
    without ever having actually read it this turn. Fixed 2026-09-25:
    write_file/edit_file on a faces/<name>/... path is refused until
    read_file has actually succeeded on that exact face's path this
    turn - grounding enforced before the edit, not a claim checked after."""

    def test_edit_without_reading_the_face_first_is_refused_then_succeeds_after_reading(self):
        scripted = _ScriptedOllama([
            _real_tool_call_message(
                "edit_file",
                {"path": "visualizer/faces/aether/index.html",
                 "old_string": "blue", "new_string": "teal"},
            ),  # blocked - never read aether first
            _real_tool_call_message(
                "read_file", {"path": "visualizer/faces/aether/index.html"},
            ),  # now reads the real file
            _real_tool_call_message(
                "edit_file",
                {"path": "visualizer/faces/aether/index.html",
                 "old_string": "blue", "new_string": "teal"},
            ),  # retries - now allowed
            _plain_text_message("Aether's color has been changed to teal."),
        ])
        messages = [{"role": "user", "content": "give aether a teal motif"}]
        with patch.object(agent, "call_ollama", scripted), \
             patch.dict(tools.REGISTRY, {
                 "read_file": lambda args: "some html with blue in it",
                 "edit_file": lambda args: "OK: replaced 1 occurrence(s)",
             }):
            agent._agentic_turn(messages, speak_answer=False)

        tool_results = [m["content"] for m in messages if m.get("role") == "tool"]
        self.assertTrue(
            tool_results[0].startswith("ERROR") and "haven't actually read" in tool_results[0],
            "first edit attempt (no read_file yet) must be refused: " + repr(tool_results[0]),
        )
        self.assertEqual(tool_results[2], "OK: replaced 1 occurrence(s)",
                          "second edit attempt, after a real read_file on the same face, must go through")

    def test_reading_a_different_face_does_not_authorize_editing_this_one(self):
        """Real incident shape, 2026-09-17: asked to restyle aether, the
        model acted on blue_board instead. Reading circuit's file must
        never authorize an edit to aether's."""
        scripted = _ScriptedOllama([
            _real_tool_call_message("read_file", {"path": "visualizer/faces/circuit/index.html"}),
            _real_tool_call_message(
                "edit_file",
                {"path": "visualizer/faces/aether/index.html",
                 "old_string": "blue", "new_string": "teal"},
            ),
            _plain_text_message("I haven't actually changed aether yet - which file should I edit?"),
        ])
        messages = [{"role": "user", "content": "give aether a teal motif"}]
        with patch.object(agent, "call_ollama", scripted), \
             patch.dict(tools.REGISTRY, {
                 "read_file": lambda args: "some html content",
                 "edit_file": lambda args: "OK: replaced 1 occurrence(s)",
             }):
            agent._agentic_turn(messages, speak_answer=False)

        tool_results = [m["content"] for m in messages if m.get("role") == "tool"]
        self.assertTrue(
            tool_results[1].startswith("ERROR") and "haven't actually read" in tool_results[1],
            "reading circuit's file must not authorize editing aether's - "
            "got: " + repr(tool_results[1]),
        )


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


# ---------------------------------------------------------------------------
# Regression 7: fast-lane give-up must escalate with exactly one spoken
# answer (2026-09-13). run_task used to hand speak_answer=True straight
# into the fast lane's _agentic_turn, so when the small model gave up,
# the turn SPOKE the give-up text - and then run_task escalated to the
# full model, which spoke its own answer. Mark heard two responses back
# to back for one request. Fix: the fast lane always runs silent; its
# turn returns the answer text, and run_task does the single speak only
# when it's NOT escalating.
# ---------------------------------------------------------------------------
class _ModelRoutingOllama:
    """One mock serving both of run_task's lanes: picks its scripted
    responses by the model kwarg _agentic_turn passes through."""
    def __init__(self, scripts):
        self._scripts = {m: list(rs) for m, rs in scripts.items()}
        self.calls = []

    def __call__(self, messages, allowed_tools=None, model=None, url=None):
        self.calls.append(model)
        queue = self._scripts.get(model, [])
        if not queue:
            raise AssertionError(
                f"no scripted responses left for model {model!r} "
                f"after {len(self.calls)} calls"
            )
        return {"message": queue.pop(0)}


def _give_up_script(tool_name="show_face"):
    """The exact shape the 2026-09-13 fix handles: one garbled attempt,
    then excuses until the retry budget is spent and the honest fallback
    takes over. 3 model calls, matching TestFabricationGiveUpGate's math."""
    excuse = (
        "I cannot execute the tool call due to interface "
        "limitations in this environment."
    )
    return [
        _malformed_text_message(tool_name),  # attempt 1: garbled syntax
        _plain_text_message(excuse),          # give-up 1/2: forced retry
        _plain_text_message(excuse),          # give-up 2/2: budget spent
    ]


class TestFastLaneEscalationSpeaksOnce(unittest.TestCase):
    def _run(self, scripts, task="bring up your face"):
        ollama = _ModelRoutingOllama(scripts)
        with patch.object(agent.router, "classify", return_value=agent.router.FAST), \
             patch.object(agent, "call_ollama", ollama), \
             patch.object(agent, "_build_system_prompt", return_value="stub"), \
             patch.object(agent.voice, "speak") as speak_mock:
            agent.run_task(task, speak_answer=True)
        return ollama, speak_mock

    def test_give_up_escalates_and_speaks_only_the_full_answer(self):
        """Fast lane garbles its tool call and gives up - run_task must
        escalate to the full model, and Mark must hear exactly ONE
        spoken answer: the full model's, never the give-up text."""
        full_answer = "Your face is up now."
        ollama, speak_mock = self._run({
            agent.FAST_MODEL: _give_up_script(),
            agent.MODEL: [_plain_text_message(full_answer)],
        })

        self.assertIn(
            agent.MODEL, ollama.calls,
            "the full model must be asked after the fast lane gives up - "
            f"calls went to: {ollama.calls}",
        )
        self.assertEqual(
            speak_mock.call_count, 1,
            "exactly one thing may be spoken per task - the old code "
            f"spoke the give-up AND the full answer: {speak_mock.call_args_list}",
        )
        spoken = speak_mock.call_args_list[0][0][0]
        self.assertEqual(
            spoken, full_answer,
            "the single spoken answer must be the full model's, not the "
            f"fast lane's give-up text - got: {spoken!r}",
        )

    def test_fast_lane_success_speaks_fast_answer_once_no_escalation(self):
        """Fast lane answers cleanly - its answer is spoken exactly once
        and the full model is never woken up."""
        fast_answer = "It's currently 3:15 PM."
        ollama, speak_mock = self._run({
            agent.FAST_MODEL: [_plain_text_message(fast_answer)],
            agent.MODEL: [_plain_text_message("SHOULD NEVER BE CALLED")],
        }, task="what time is it")

        self.assertNotIn(
            agent.MODEL, ollama.calls,
            "a clean fast-lane answer must not escalate - "
            f"calls went to: {ollama.calls}",
        )
        self.assertEqual(speak_mock.call_count, 1)
        self.assertEqual(
            speak_mock.call_args_list[0][0][0], fast_answer,
            "the fast lane's own answer is what gets spoken - got: "
            f"{speak_mock.call_args_list[0][0][0]!r}",
        )

    def test_silent_task_never_speaks_even_on_escalation(self):
        """run_task(speak_answer=False) - e.g. relay/one-shot callers that
        only want the side effects - must not speak at all, even when the
        fast lane gives up and the full model answers."""
        ollama = _ModelRoutingOllama({
            agent.FAST_MODEL: _give_up_script(),
            agent.MODEL: [_plain_text_message("Full answer.")],
        })
        with patch.object(agent.router, "classify", return_value=agent.router.FAST), \
             patch.object(agent, "call_ollama", ollama), \
             patch.object(agent, "_build_system_prompt", return_value="stub"), \
             patch.object(agent.voice, "speak") as speak_mock:
            agent.run_task("bring up your face", speak_answer=False)

        self.assertEqual(
            speak_mock.call_count, 0,
            "a silent task must stay silent through escalation - "
            f"spoken: {speak_mock.call_args_list}",
        )
        self.assertIn(agent.MODEL, ollama.calls)


class TestFaceEditRoutesToQwenCoder(unittest.TestCase):
    """2026-09-25: face-edit requests (router.EDIT) route to
    agent.EDIT_MODEL (qwen3-coder:30b) instead of granite - the incident
    history on this exact task shape (circuit clobber, aether->blue_board,
    swarm-doubling) is granite's worst of any category, so precision
    file edits go to a model that's actually good at them."""

    def test_edit_classification_calls_edit_model_not_granite(self):
        answer = "Aether's motif is now blue."
        ollama = _ModelRoutingOllama({
            agent.EDIT_MODEL: [_plain_text_message(answer)],
            agent.MODEL: [_plain_text_message("SHOULD NEVER BE CALLED")],
        })
        with patch.object(agent.router, "classify", return_value=agent.router.EDIT), \
             patch.object(agent, "call_ollama", ollama), \
             patch.object(agent, "_build_system_prompt", return_value="stub"), \
             patch.object(agent.voice, "speak") as speak_mock:
            agent.run_task("give aether a blue motif", speak_answer=True)

        self.assertIn(agent.EDIT_MODEL, ollama.calls)
        self.assertNotIn(
            agent.MODEL, ollama.calls,
            "a clean EDIT-model answer must not also wake granite - "
            f"calls went to: {ollama.calls}",
        )
        self.assertEqual(speak_mock.call_count, 1)
        self.assertEqual(speak_mock.call_args_list[0][0][0], answer)

    def test_edit_model_unreachable_falls_back_to_granite(self):
        answer = "Aether's motif is now blue."
        ollama = _ModelRoutingOllama({
            agent.MODEL: [_plain_text_message(answer)],
        })

        def flaky(messages, allowed_tools=None, model=None, url=None):
            if model == agent.EDIT_MODEL:
                raise agent.OllamaUnavailable("connection refused")
            return ollama(messages, allowed_tools=allowed_tools, model=model, url=url)

        with patch.object(agent.router, "classify", return_value=agent.router.EDIT), \
             patch.object(agent, "call_ollama", flaky), \
             patch.object(agent, "_build_system_prompt", return_value="stub"), \
             patch.object(agent.voice, "speak") as speak_mock:
            agent.run_task("give aether a blue motif", speak_answer=True)

        self.assertIn(agent.MODEL, ollama.calls)
        self.assertEqual(speak_mock.call_count, 1)
        self.assertEqual(speak_mock.call_args_list[0][0][0], answer)


class FaceLookClaimRegexTests(unittest.TestCase):
    """2026-09-18: 'updated circuit face with double the traces' claims
    slipped past the original color/motif-only regex."""

    def test_trace_and_update_claims_are_flagged(self):
        for text in (
            "Here is the updated 'circuit' face with double the amount of traces.",
            "The circuit face has been updated to include four times more traces.",
            "The background image now includes double the amount of traces.",
            "I gave the aether face a blue motif.",
        ):
            self.assertTrue(agent._FACE_LOOK_CLAIM_RE.search(text), text)

    def test_plain_switch_and_status_answers_pass(self):
        for text in (
            "Your active visual face is currently set to 'circuit'.",
            "Switched the face to circuit.",
            "Your face window is now in front, centered on screen.",
        ):
            self.assertFalse(agent._FACE_LOOK_CLAIM_RE.search(text), text)


class FaceCreateClaimRegexTests(unittest.TestCase):
    """Sibling to FaceLookClaimRegexTests above, but for a face-CREATION
    claim rather than a restyle. See _FACE_CREATE_CLAIM_RE."""

    def test_creation_claims_are_flagged(self):
        for text in (
            "I've created a new face for you, fully original.",
            "I built a brand new face with a swirling gradient.",
            "Your new face is ready.",
            "The face is done - take a look.",
        ):
            self.assertTrue(agent._FACE_CREATE_CLAIM_RE.search(text), text)

    def test_switch_and_restyle_claims_do_not_false_positive(self):
        for text in (
            "Switched the face to circuit.",
            "I gave the aether face a blue motif.",
            "Your active visual face is currently set to 'circuit'.",
        ):
            self.assertFalse(agent._FACE_CREATE_CLAIM_RE.search(text), text)


class FaceCreateClaimGuardTests(unittest.TestCase):
    """2026-09-17: asked for '100% original face,' the model wrote a
    static webpage with an invented schema - no core.js include, no
    AV.tick(dt) call, the two things every real face in
    visualizer/faces/ actually has - then reported it as done. Built and
    tested in isolation that night, then deliberately reverted unfinished
    (Mark's call, to fold it into the queued bench work instead of
    shipping a fifth guard at 12:30 AM). Built for real 2026-09-25,
    alongside the fix (see the read-before-edit gate's exists() check)
    for the gate this scenario ran into: a brand-new face has nothing to
    read_file first, so that gate must not block its initial write_file
    the way it correctly blocks a blind edit of an EXISTING face."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.tmp.name)
        (root / "faces" / "newface").mkdir(parents=True)
        self.root = root
        self.patch = patch.object(tools, "VISUALIZER_DIR", root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def _write_face(self, content):
        (self.root / "faces" / "newface" / "index.html").write_text(content, encoding="utf-8")

    def test_fabricated_face_created_claim_gets_retried_then_replaced(self):
        # The file must not exist before the scripted write_file call -
        # the read-before-edit gate only exempts a face that doesn't
        # exist yet (see agent.py's exists() check), and this test is
        # specifically exercising a brand-new-face creation. The mocked
        # handler writes the (structurally invalid) content itself, same
        # as the real write_file tool would on this exact incident shape.
        def fake_write_file(args):
            self._write_face("<html><body>a plain static page, no core, no tick</body></html>")
            return "OK: wrote 1 file"

        scripted = _ScriptedOllama([
            _real_tool_call_message(
                "write_file",
                {"path": "visualizer/faces/newface/index.html", "content": "irrelevant"},
            ),
            _plain_text_message("I've created a brand new face for you, fully original."),
            _plain_text_message("I've created a brand new face for you, fully original."),
        ])
        messages = [{"role": "user", "content": "make me a 100% original face"}]
        with patch.object(agent, "call_ollama", scripted), \
             patch.dict(tools.REGISTRY, {"write_file": fake_write_file}):
            result = agent._agentic_turn(messages, speak_answer=False)

        self.assertEqual(scripted.calls, 3, "must force exactly one retry before giving up")
        self.assertIn("not actually a working one", result["answer"])
        self.assertNotIn("fully original", result["answer"])

    def test_genuinely_valid_face_creation_claim_passes_through(self):
        def fake_write_file(args):
            self._write_face(
                '<html><body>\n<script src="../../core.js"></script>\n'
                '<script>function tick(dt){AV.tick(dt);}</script>\n</body></html>'
            )
            return "OK: wrote 1 file"

        scripted = _ScriptedOllama([
            _real_tool_call_message(
                "write_file",
                {"path": "visualizer/faces/newface/index.html", "content": "irrelevant"},
            ),
            _plain_text_message("I've created a brand new face for you."),
        ])
        messages = [{"role": "user", "content": "make me a new face"}]
        with patch.object(agent, "call_ollama", scripted), \
             patch.dict(tools.REGISTRY, {"write_file": fake_write_file}):
            result = agent._agentic_turn(messages, speak_answer=False)

        self.assertEqual(scripted.calls, 2, "a genuinely valid creation must not be retried")
        self.assertEqual(result["answer"], "I've created a brand new face for you.")

    def test_write_file_on_a_brand_new_face_is_not_blocked_by_the_read_before_edit_gate(self):
        # The read-before-edit gate (agent.py, same block this test's
        # sibling class exercises) must only refuse a write/edit to a face
        # whose index.html already exists - a face that doesn't exist yet
        # has nothing to read_file first, so demanding one would make
        # creating any new face impossible.
        self._write_face("placeholder - overwritten by the scripted write_file call below")
        (self.root / "faces" / "newface" / "index.html").unlink()
        scripted = _ScriptedOllama([
            _real_tool_call_message(
                "write_file",
                {"path": "visualizer/faces/newface/index.html", "content": "irrelevant"},
            ),
            _plain_text_message("Done."),
        ])
        messages = [{"role": "user", "content": "make me a new face"}]
        with patch.object(agent, "call_ollama", scripted), \
             patch.dict(tools.REGISTRY, {"write_file": lambda args: "OK: wrote 1 file"}):
            agent._agentic_turn(messages, speak_answer=False)

        tool_results = [m["content"] for m in messages if m.get("role") == "tool"]
        self.assertEqual(tool_results[0], "OK: wrote 1 file",
                          "a write to a face that doesn't exist yet must go through unblocked: "
                          + repr(tool_results[0]))


class SetFaceFuzzyMatchTests(unittest.TestCase):
    """Flagged as an open bench item in September, never closed until
    2026-09-25: face_tunables/set_face_tunable already suggest a close
    match on a speech-recognition slip ('acer' for 'aether'), but set_face
    itself - the tool that actually switches the visible face - never
    got the same fix and just flatly errored with no suggestion."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.tmp.name)
        for name in ("aether", "circuit"):
            (root / "faces" / name).mkdir(parents=True)
            (root / "faces" / name / "index.html").write_text("<html></html>", encoding="utf-8")
        (root / "ai-visualizer.json").write_text('{"face": "circuit"}', encoding="utf-8")
        self.root = root
        self.patches = [
            patch.object(tools, "VISUALIZER_DIR", root),
            patch.object(tools, "VISUALIZER_CONFIG_PATH", root / "ai-visualizer.json"),
        ]
        for pt in self.patches:
            pt.start()

    def tearDown(self):
        for pt in self.patches:
            pt.stop()
        self.tmp.cleanup()

    def test_near_miss_name_gets_a_suggestion_not_a_bare_error(self):
        for typo in ("ether", "aeter", "acer"):
            with self.subTest(typo=typo):
                result = tools.set_face({"face": typo})
                self.assertTrue(result.startswith("ERROR"), result)
                self.assertIn("aether", result)

    def test_config_is_untouched_on_a_near_miss(self):
        before = (self.root / "ai-visualizer.json").read_text(encoding="utf-8")
        tools.set_face({"face": "aeter"})
        self.assertEqual((self.root / "ai-visualizer.json").read_text(encoding="utf-8"), before)

    def test_a_name_with_nothing_close_gets_no_hint_but_still_errors(self):
        result = tools.set_face({"face": "xyzzy_totally_unrelated"})
        self.assertTrue(result.startswith("ERROR"), result)
        self.assertNotIn("Did you mean", result)


class RemoteGraniteRoutingTests(unittest.TestCase):
    """2026-09-20: granite can be served from ADLAPTOP over the LAN
    (JARVIS_REMOTE_GRANITE_URL), off by default, with a fallback to local
    granite when the laptop isn't answering. Offline: _post_chat and the
    reachability check are patched, nothing touches a network."""

    REMOTE = "http://10.0.0.9:11434/api/chat"

    def setUp(self):
        agent._remote_dead_until = 0.0
        agent._last_chat_target = None

    def _run(self, remote=REMOTE, reachable=True, fail=(), **kw):
        seen = []

        def fake_post(url, data):
            seen.append(url)
            if url in fail:
                if url == agent.OLLAMA_URL:
                    raise agent.urllib.error.URLError("local down")
                raise OSError("remote down")
            return {"message": {"role": "assistant", "content": "ok"}}

        with patch.object(agent, "REMOTE_GRANITE_URL", remote),                 patch.object(agent, "_remote_reachable", return_value=reachable),                 patch.object(agent, "_post_chat", side_effect=fake_post):
            agent.call_ollama([{"role": "user", "content": "hi"}], **kw)
        return seen

    def test_remote_payload_asks_for_the_big_context_and_local_does_not(self):
        bodies = {}

        def fake_post(url, data):
            bodies[url] = json.loads(data.decode("utf-8"))
            return {"message": {"role": "assistant", "content": "ok"}}

        with patch.object(agent, "REMOTE_GRANITE_URL", self.REMOTE),                 patch.object(agent, "_remote_reachable", return_value=True),                 patch.object(agent, "_post_chat", side_effect=fake_post):
            agent.call_ollama([{"role": "user", "content": "hi"}])
        self.assertEqual(
            bodies[self.REMOTE]["options"], {"num_ctx": agent.REMOTE_NUM_CTX}
        )
        self.assertGreaterEqual(agent.REMOTE_NUM_CTX, 16384)

        bodies.clear()
        with patch.object(agent, "REMOTE_GRANITE_URL", None),                 patch.object(agent, "_post_chat", side_effect=fake_post):
            agent.call_ollama([{"role": "user", "content": "hi"}])
        self.assertNotIn("options", bodies[agent.OLLAMA_URL])

    def test_off_by_default_uses_local(self):
        self.assertEqual(self._run(remote=None), [agent.OLLAMA_URL])

    def test_uses_remote_when_configured_and_reachable(self):
        self.assertEqual(self._run(), [self.REMOTE])

    def test_falls_back_to_local_when_unreachable(self):
        self.assertEqual(self._run(reachable=False), [agent.OLLAMA_URL])

    def test_falls_back_when_remote_request_fails_then_skips_it_for_a_while(self):
        self.assertEqual(
            self._run(fail=(self.REMOTE,)), [self.REMOTE, agent.OLLAMA_URL]
        )
        self.assertGreater(agent._remote_dead_until, agent.time.monotonic())
        # Second call with the REAL reachability check: the recent failure
        # short-circuits it, so the remote is never tried again this window.
        seen = []
        with patch.object(agent, "REMOTE_GRANITE_URL", self.REMOTE),                 patch.object(
                    agent, "_post_chat",
                    side_effect=lambda u, d: seen.append(u) or {"message": {}},
                ):
            agent.call_ollama([{"role": "user", "content": "hi"}])
        self.assertEqual(seen, [agent.OLLAMA_URL])

    def test_explicit_url_is_never_redirected(self):
        seen = self._run(url=agent.FAST_OLLAMA_URL, model=agent.FAST_MODEL)
        self.assertEqual(seen, [agent.FAST_OLLAMA_URL])

    def test_other_models_are_never_redirected(self):
        self.assertEqual(
            self._run(model="qwen3-coder:30b"), [agent.OLLAMA_URL]
        )

    def test_local_failure_after_remote_fallback_still_raises_unavailable(self):
        with self.assertRaises(agent.OllamaUnavailable):
            self._run(fail=(self.REMOTE, agent.OLLAMA_URL))

    def test_real_reachability_check_reports_false_and_marks_dead(self):
        with patch.object(agent, "REMOTE_GRANITE_URL", "http://127.0.0.1:9/api/chat"),                 patch.object(agent, "REMOTE_CONNECT_TIMEOUT_S", 1):
            self.assertFalse(agent._remote_reachable())
        self.assertGreater(agent._remote_dead_until, agent.time.monotonic())


class MemoryFileWriteBlockTests(unittest.TestCase):
    """2026-09-20: asked to double the orbit face's swarm, the model
    replaced a whole LESSONS.md paragraph via edit_file and claimed
    success. The generic write/edit tools must never touch the memory
    files, with no confirmed=true way around it. The block returns before
    any file IO, so these tests never touch the real vault."""

    def _blocked(self, result):
        self.assertTrue(result.startswith("BLOCKED:"), result)
        self.assertIn("memory files", result)

    def test_edit_and_write_are_blocked_on_every_memory_file(self):
        daily = str(tools.DAILY_DIR / "2026-09-20.md")
        for target in (str(tools.LESSONS_FILE), str(tools.TASKS_FILE), daily):
            self._blocked(tools.edit_file(
                {"path": target, "old_string": "a", "new_string": "b"}))
            self._blocked(tools.write_file({"path": target, "content": "x"}))

    def test_relative_path_and_dotdot_traversal_are_blocked(self):
        self._blocked(tools.edit_file(
            {"path": "vault/LESSONS.md", "old_string": "a", "new_string": "b"}))
        self._blocked(tools.edit_file(
            {"path": "vault/daily/../LESSONS.md", "old_string": "a", "new_string": "b"}))
        self._blocked(tools.write_file(
            {"path": "vault/LESSONS.md".upper().replace("VAULT", "vault"), "content": "x"}))

    def test_confirmed_true_is_not_an_override(self):
        self._blocked(tools.edit_file({
            "path": str(tools.LESSONS_FILE), "old_string": "a",
            "new_string": "b", "confirmed": True}))
        self._blocked(tools.write_file({
            "path": str(tools.LESSONS_FILE), "content": "x", "confirmed": True}))

    def test_block_message_names_the_right_tool(self):
        self.assertIn("append_lesson", tools._memory_file_block_message(tools.LESSONS_FILE))
        self.assertIn("complete_task", tools._memory_file_block_message(tools.TASKS_FILE))
        self.assertIn("append_daily_note",
                      tools._memory_file_block_message(tools.DAILY_DIR / "2026-09-20.md"))

    def test_ordinary_files_and_index_are_not_blocked(self):
        self.assertIsNone(tools._memory_file_block_message(tools.INDEX_FILE))
        self.assertIsNone(tools._memory_file_block_message(
            tools.WORKSPACE / "visualizer" / "faces" / "orbit" / "index.html"))
        # Rooted inside WORKSPACE deliberately, not a bare system temp dir -
        # see TestOutsideWorkspaceBlock below: 2026-09-24 closed a real gap
        # where write_file/edit_file could write anywhere on disk, so an
        # ordinary-file test now has to live inside the sandbox too.
        with tempfile.TemporaryDirectory(dir=tools.WORKSPACE) as d:
            f = pathlib.Path(d) / "scratch.txt"
            self.assertTrue(tools.write_file({"path": str(f), "content": "hi"}).startswith("OK:"))
            self.assertTrue(tools.edit_file(
                {"path": str(f), "old_string": "hi", "new_string": "yo"}).startswith("OK:"))
            self.assertEqual(f.read_text(encoding="utf-8"), "yo")


class TestOutsideWorkspaceBlock(unittest.TestCase):
    """Real gap found 2026-09-24 while scoping search_vault to also read
    Mary's real Active Priorities.md: _resolve() passed any absolute path
    straight through untouched, so nothing stopped write_file/edit_file
    from writing anywhere on disk the model named - including Mary's real
    vault, which isn't git-tracked and has no clean checkout-based undo
    like Jarvis's own tools.py/agent.py get when he clobbers them."""

    def test_write_file_blocks_absolute_path_outside_workspace(self):
        with tempfile.TemporaryDirectory() as d:
            target = pathlib.Path(d) / "not_mine.md"
            target.write_text("original content", encoding="utf-8")
            result = tools.write_file({"path": str(target), "content": "CLOBBERED"})
            self.assertTrue(result.startswith("BLOCKED:"), result)
            self.assertNotIn("confirmed", result.lower().split("no ")[0])
            self.assertEqual(target.read_text(encoding="utf-8"), "original content")

    def test_write_file_blocks_even_with_confirmed_true(self):
        """No confirmed=true override - the model sets that flag itself,
        same reasoning as _memory_file_block_message."""
        with tempfile.TemporaryDirectory() as d:
            target = pathlib.Path(d) / "not_mine.md"
            target.write_text("original content", encoding="utf-8")
            result = tools.write_file(
                {"path": str(target), "content": "CLOBBERED", "confirmed": True}
            )
            self.assertTrue(result.startswith("BLOCKED:"), result)
            self.assertEqual(target.read_text(encoding="utf-8"), "original content")

    def test_edit_file_blocks_absolute_path_outside_workspace(self):
        with tempfile.TemporaryDirectory() as d:
            target = pathlib.Path(d) / "not_mine.md"
            target.write_text("original content", encoding="utf-8")
            result = tools.edit_file(
                {"path": str(target), "old_string": "original", "new_string": "CLOBBERED"}
            )
            self.assertTrue(result.startswith("BLOCKED:"), result)
            self.assertEqual(target.read_text(encoding="utf-8"), "original content")

    def test_files_inside_workspace_still_writable(self):
        with tempfile.TemporaryDirectory(dir=tools.WORKSPACE) as d:
            target = pathlib.Path(d) / "scratch.txt"
            result = tools.write_file({"path": str(target), "content": "fine"})
            self.assertTrue(result.startswith("OK:"), result)
            self.assertEqual(target.read_text(encoding="utf-8"), "fine")


class UnbackedChangeClaimGuardTests(unittest.TestCase):
    """2026-09-20: asked to double the orbit swarm, the model edited the
    wrong file and answered 'The swarm scaling parameter in the lessons has
    been doubled.' A claim of a change with no state-changing tool success
    behind it gets one forced retry, then an honest replacement. Known
    limit (not tested because it can't be caught): a successful edit of the
    WRONG target still passes."""

    CLAIM = "The swarm scaling parameter has been doubled."

    def test_detector_flags_claims_and_clears_denials_and_plain_answers(self):
        for text in (
            self.CLAIM, "I've doubled the swarm.",
            "I have successfully edited the file.",
            "The file has now been replaced.", "I deleted the old entry.",
            "I've adjusted the swarm objects by doubling their count.",
            "The change has been made - refresh the face.",
            "The orbit face has been updated.",
        ):
            self.assertTrue(agent._is_change_claim(text), text)
        for text in (
            "The lesson was updated yesterday.",
            "Here are the faces currently available.",
            "The circuit face has been activated.", "I'm ready to assist!",
            "Your active face is aether2.",
            "I can double it if you tell me which file.",
            "The orbit face is now active.", "Nothing has been edited yet.",
            "I haven't changed anything.", "The swarm has not been doubled.",
            "It can't be doubled from here.",
        ):
            self.assertFalse(agent._is_change_claim(text), text)

    def _run(self, script, registry):
        scripted = _ScriptedOllama(script)
        messages = [{"role": "user", "content": "double the swarm"}]
        with patch.object(agent, "call_ollama", scripted), \
             patch.dict(tools.REGISTRY, registry):
            agent._agentic_turn(messages, speak_answer=False)
        return messages[-1]["content"], scripted.calls

    def test_claim_after_only_reading_is_retried_then_replaced(self):
        final, calls = self._run(
            [_real_tool_call_message("read_file", {"path": "x"}),
             _plain_text_message(self.CLAIM),
             _plain_text_message(self.CLAIM)],
            {"read_file": lambda a: "some contents"},
        )
        self.assertNotIn("doubled", final)
        self.assertIn("nothing was actually changed", final)
        self.assertEqual(calls, 3)

    def test_claim_after_a_failed_edit_is_still_caught(self):
        final, calls = self._run(
            [_real_tool_call_message("edit_file", {"path": "x"}),
             _plain_text_message(self.CLAIM),
             _plain_text_message(self.CLAIM)],
            {"edit_file": lambda a: "ERROR: old_string not found in x"},
        )
        self.assertNotIn("doubled", final)
        self.assertEqual(calls, 3)

    def test_honest_correction_after_the_retry_passes_through(self):
        honest = "I haven't changed anything yet. Which file do you mean?"
        final, calls = self._run(
            [_real_tool_call_message("read_file", {"path": "x"}),
             _plain_text_message(self.CLAIM),
             _plain_text_message(honest)],
            {"read_file": lambda a: "some contents"},
        )
        self.assertEqual(final, honest)
        self.assertEqual(calls, 3)

    def test_claim_backed_by_a_successful_change_is_untouched(self):
        final, calls = self._run(
            [_real_tool_call_message("edit_file", {"path": "x"}),
             _plain_text_message(self.CLAIM)],
            {"edit_file": lambda a: "OK: replaced 1 occurrence(s) in x"},
        )
        self.assertEqual(final, self.CLAIM)
        self.assertEqual(calls, 2)


class FaceTunablesTests(unittest.TestCase):
    """2026-09-20: neither granite (0/8) nor qwen3-coder:30b (0/4) could
    'double the swarm objects' on the orbit face by hunting through its
    HTML, so face_tunables / set_face_tunable do the grounding in code.
    Every test runs against a scratch copy of the visualizer folder."""

    FACE_HTML = (
        "<style>\n:root{\n  --ink:#eaf3f6; --cool:#43a9d0;\n}\n</style>\n"
        "<script>\n"
        "const SQUASH=.32;   // shared vertical-axis squash\n"
        "/* ---------------- ambient swarm: fixed at load ---------------- */\n"
        "const AMB_N=14;\n"
        "const TASKLOG_MAX=40;\n"
        "</script>\n"
    )

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.tmp.name)
        for name in ("orbit", "aether"):
            (root / "faces" / name).mkdir(parents=True)
            (root / "faces" / name / "index.html").write_text(self.FACE_HTML, encoding="utf-8")
        (root / "ai-visualizer.json").write_text('{"face": "orbit"}', encoding="utf-8")
        self.root = root
        self.patches = [
            patch.object(tools, "VISUALIZER_DIR", root),
            patch.object(tools, "VISUALIZER_CONFIG_PATH", root / "ai-visualizer.json"),
        ]
        for pt in self.patches:
            pt.start()

    def tearDown(self):
        for pt in self.patches:
            pt.stop()
        self.tmp.cleanup()

    def _html(self, face="orbit"):
        return (self.root / "faces" / face / "index.html").read_bytes().decode("utf-8")

    def test_lists_values_with_plain_descriptions_for_the_active_face(self):
        out = tools.face_tunables({})
        self.assertIn("AMB_N = 14", out)
        self.assertIn("ambient swarm", out)
        self.assertIn("SQUASH = .32", out)
        self.assertIn("--cool = #43a9d0", out)

    def test_parses_the_real_orbit_face_and_finds_the_swarm(self):
        real = pathlib.Path(__file__).parent / "visualizer" / "faces" / "orbit" / "index.html"
        if not real.exists():
            self.skipTest("no real orbit face on this machine")
        items = {i["name"]: i for i in tools._parse_tunables(real.read_text(encoding="utf-8"))}
        self.assertIn("AMB_N", items)
        self.assertIn("swarm", items["AMB_N"]["desc"].lower())

    def test_doubling_the_swarm_changes_only_that_value(self):
        before = self._html()
        result = tools.set_face_tunable({"name": "AMB_N", "value": "28"})
        self.assertTrue(result.startswith("OK:"), result)
        after = self._html()
        self.assertEqual(after, before.replace("const AMB_N=14;", "const AMB_N=28;"))
        self.assertEqual(self._html("aether"), before, "other faces must be untouched")

    def test_line_endings_are_preserved(self):
        crlf_html = self.FACE_HTML.replace("\n", "\r\n")
        (self.root / "faces" / "orbit" / "index.html").write_bytes(crlf_html.encode("utf-8"))
        tools.set_face_tunable({"name": "AMB_N", "value": "28"})
        raw = (self.root / "faces" / "orbit" / "index.html").read_bytes()
        self.assertEqual(raw.count(b"\r\n"), crlf_html.count("\r\n"))
        self.assertEqual(raw.count(b"\n"), raw.count(b"\r\n"))

    def test_absurd_and_bad_values_are_rejected_and_nothing_changes(self):
        before = self._html()
        for bad in ("100000", "-5", "lots", "nan", "inf"):
            result = tools.set_face_tunable({"name": "AMB_N", "value": bad})
            self.assertTrue(result.startswith("ERROR"), (bad, result))
        self.assertEqual(self._html(), before)

    def test_colors_take_hex_only(self):
        ok = tools.set_face_tunable({"name": "--cool", "value": "#0000ff"})
        self.assertTrue(ok.startswith("OK:"), ok)
        self.assertIn("--cool:#0000ff", self._html())
        before = self._html()
        self.assertTrue(tools.set_face_tunable({"name": "cool", "value": "blue"}).startswith("ERROR"))
        self.assertEqual(self._html(), before)

    def test_exact_names_are_forgiving_about_case_and_dashes(self):
        self.assertTrue(tools.set_face_tunable({"name": "amb_n", "value": "20"}).startswith("OK:"))
        self.assertTrue(tools.set_face_tunable({"name": "cool", "value": "#111111"}).startswith("OK:"))

    def test_mark_s_own_words_find_the_right_value_by_description(self):
        for phrase in ("swarm objects", "swarm_count", "swarmObjectsCount", "the swarm"):
            (self.root / "faces" / "orbit" / "index.html").write_text(self.FACE_HTML, encoding="utf-8")
            result = tools.set_face_tunable({"what": phrase, "value": "double"})
            self.assertTrue(result.startswith("OK:"), (phrase, result))
            self.assertIn("AMB_N 14 -> 28", result)
            self.assertIn("const AMB_N=28;", self._html())
            self.assertIn("const SQUASH=.32;", self._html(), "must not touch the wrong value")

    def test_the_request_overrides_a_model_pick_that_ignores_what_mark_said(self):
        # the model grabbed SQUASH, but Mark said 'swarm'
        result = tools.set_face_tunable({
            "what": "SQUASH", "value": "double",
            "_request": "make an edit to this space and double the swarm objects",
        })
        self.assertTrue(result.startswith("OK:"), result)
        self.assertIn("AMB_N 14 -> 28", result)
        self.assertIn("const SQUASH=.32;", self._html())

    def test_a_model_pick_that_matches_the_request_is_kept(self):
        result = tools.set_face_tunable({
            "what": "AMB_N", "value": "double",
            "_request": "double the swarm",
        })
        self.assertIn("AMB_N 14 -> 28", result)
        self.assertNotIn("[used", result)

    def test_a_scale_word_in_the_request_beats_a_wrong_multiplier(self):
        result = tools.set_face_tunable({
            "what": "swarm", "value": "x3", "_request": "double the swarm objects",
        })
        self.assertIn("AMB_N 14 -> 28", result)
        (self.root / "faces" / "orbit" / "index.html").write_text(self.FACE_HTML, encoding="utf-8")
        result = tools.set_face_tunable({
            "what": "swarm", "value": "20", "_request": "double the swarm objects",
        })
        self.assertIn("AMB_N 14 -> 28", result)

    def test_an_explicit_number_in_the_request_is_respected(self):
        result = tools.set_face_tunable({
            "what": "swarm", "value": "40", "_request": "set the swarm to 40",
        })
        self.assertIn("AMB_N 14 -> 40", result)

    def test_a_face_mark_never_named_is_replaced_by_the_active_face(self):
        before_aether = self._html("aether")
        result = tools.set_face_tunable({
            "face": "aether", "what": "swarm", "value": "double",
            "_request": "double the swarm objects",
        })
        self.assertTrue(result.startswith("OK:"), result)
        self.assertEqual(self._html("aether"), before_aether)
        self.assertIn("const AMB_N=28;", self._html("orbit"))

    def test_a_face_mark_did_name_is_used(self):
        result = tools.set_face_tunable({
            "face": "aether", "what": "swarm", "value": "double",
            "_request": "double the swarm on the aether face",
        })
        self.assertIn("const AMB_N=28;", self._html("aether"))
        self.assertIn("aether", result)

    def test_a_request_that_matches_nothing_refuses_instead_of_guessing(self):
        before = self._html()
        result = tools.set_face_tunable({
            "what": "AMB_N", "value": "40",
            "_request": "make the rainbow louder",
        })
        self.assertTrue(result.startswith("ERROR"), result)
        self.assertEqual(self._html(), before)

    def test_a_repeat_call_in_the_same_turn_does_not_compound(self):
        first = tools.set_face_tunable({"what": "swarm", "value": "double", "_turn": "t1"})
        self.assertIn("AMB_N 14 -> 28", first)
        again = tools.set_face_tunable({"what": "swarm", "value": "double", "_turn": "t1"})
        self.assertTrue(again.startswith("ALREADY DONE"), again)
        self.assertIn("const AMB_N=28;", self._html())
        # a new turn may change it again
        later = tools.set_face_tunable({"what": "swarm", "value": "double", "_turn": "t2"})
        self.assertIn("AMB_N 28 -> 56", later)

    def test_a_phrase_that_matches_nothing_lists_candidates_and_changes_nothing(self):
        before = self._html()
        result = tools.set_face_tunable({"what": "speed", "value": "2"})
        self.assertTrue(result.startswith("ERROR"), result)
        self.assertIn("AMB_N", result)
        self.assertEqual(self._html(), before)

    def test_scale_words_use_the_current_value(self):
        for raw, expected in (("double", "28"), ("x3", "42"), ("3x", "42"), ("half", "7"), ("20", "20")):
            (self.root / "faces" / "orbit" / "index.html").write_text(self.FACE_HTML, encoding="utf-8")
            result = tools.set_face_tunable({"what": "swarm", "value": raw})
            self.assertTrue(result.startswith("OK:"), (raw, result))
            self.assertIn(f"const AMB_N={expected};", self._html(), raw)

    def test_near_miss_face_name_gets_a_suggestion_not_a_guess(self):
        before = self._html("aether")
        result = tools.set_face_tunable({"face": "aeter", "name": "AMB_N", "value": "28"})
        self.assertTrue(result.startswith("ERROR"), result)
        self.assertIn("aether", result)
        self.assertEqual(self._html("aether"), before)

    def test_registered_and_counted_as_a_real_change_by_the_claim_guards(self):
        self.assertIn("face_tunables", tools.REGISTRY)
        self.assertIn("set_face_tunable", tools.REGISTRY)
        names = {s["function"]["name"] for s in tools.SCHEMAS}
        self.assertLessEqual({"face_tunables", "set_face_tunable"}, names)
        self.assertIn("set_face_tunable", agent._STATE_CHANGING_TOOLS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
