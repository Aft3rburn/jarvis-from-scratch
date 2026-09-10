"""
Tests for router.py's classify() heuristic - item 3 of the 2026-09-10
four-part roadmap. Real example requests, not synthetic edge cases, so a
future change to the heuristic gets checked against realistic Jarvis
traffic before it ships. Run with:

    python test_router.py
"""

import unittest

from router import classify, FAST, FULL


class TestRouterClassify(unittest.TestCase):
    def test_simple_commands_are_fast(self):
        simple = [
            "what time is it",
            "turn off the lights",
            "set a timer for 5 minutes",
            "check the gpu temp",
            "check your backup status",
            "restart the visualizer",
            "stop listening",
            "list open tasks",
            "show your face",
            "how much disk space is left",
            "what's the weather",
        ]
        for text in simple:
            with self.subTest(text=text):
                self.assertEqual(
                    classify(text), FAST,
                    f"{text!r} should fast-path - it's a bounded simple command",
                )

    def test_complex_requests_are_full(self):
        complex_requests = [
            "write a python function that finds the second largest unique number in a list",
            "debug this crash, here's the traceback",
            "can you explain why the model keeps fabricating answers",
            "refactor tools.py to split the vault functions into their own module",
            "plan out how to migrate the vault to a new folder structure",
            "compare qwen3-coder and qwen3.8 for this use case and give a recommendation",
            "summarize what happened in today's daily note",
            "design a database schema for tracking scheduled tasks",
        ]
        for text in complex_requests:
            with self.subTest(text=text):
                self.assertEqual(
                    classify(text), FULL,
                    f"{text!r} should stay on the full model - it needs real reasoning",
                )

    def test_long_message_defaults_full_even_without_keywords(self):
        long_text = (
            "I was thinking about the whole situation with the backup task "
            "and the scheduled restart and whether it actually ran last "
            "night or not because I did not see a log entry for it anywhere"
        )
        self.assertEqual(classify(long_text), FULL)

    def test_short_ambiguous_message_defaults_fast(self):
        self.assertEqual(classify("hey Jarvis"), FAST)
        self.assertEqual(classify("good morning"), FAST)

    def test_medium_ambiguous_message_errs_full(self):
        # 7-20 words, no explicit simple pattern, no complex keyword -
        # deliberately errs toward FULL per the stated risk tradeoff
        # (a wrongly-fast-tracked complex request risks a bad answer,
        # a wrongly-escalated simple one only costs a little latency).
        text = "hey so I was wondering about something from earlier today"
        self.assertEqual(len(text.split()), 10)
        self.assertEqual(classify(text), FULL)

    def test_empty_input_is_fast_not_an_error(self):
        self.assertEqual(classify(""), FAST)
        self.assertEqual(classify("   "), FAST)


if __name__ == "__main__":
    unittest.main(verbosity=2)
