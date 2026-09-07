"""Tests for the shared configuration helper.

The bug these guard against: the sidebar reported a key as configured while it
still held the placeholder text from .env.example, because the check compared
against one exact string.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402


class TestIsConfigured(unittest.TestCase):

    def test_real_looking_keys_are_accepted(self):
        for value in [
            # Obviously synthetic, but shaped like the real thing.
            "0123456789abcdef0123456789abcdef",          # OpenWeatherMap style
            "gsk_EXAMPLEEXAMPLEEXAMPLEEXAMPLE0000",      # Groq style
            "sk-proj-EXAMPLEEXAMPLE0000",                # OpenAI style
            "ollama",
        ]:
            with self.subTest(value=value):
                self.assertTrue(config.is_configured(value))

    def test_placeholders_are_rejected(self):
        for value in [
            "your_key_here",
            "paste_your_groq_key_here",
            "your_groq_key_here",
            "YOUR_KEY_HERE",
            "changeme",
            "xxxxxxxx",
        ]:
            with self.subTest(value=value):
                self.assertFalse(config.is_configured(value))

    def test_blank_values_are_rejected(self):
        for value in [None, "", "   ", "\t\n"]:
            with self.subTest(value=value):
                self.assertFalse(config.is_configured(value))

    def test_surrounding_quotes_are_ignored(self):
        self.assertFalse(config.is_configured('"your_key_here"'))
        self.assertTrue(config.is_configured('"0123456789abcdef"'))

    def test_all_env_example_placeholders_are_rejected(self):
        """Whatever ships in .env.example must never count as configured."""
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            ".env.example")
        checked = 0
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                if name.strip().endswith("API_KEY"):
                    checked += 1
                    self.assertFalse(config.is_configured(value),
                                     f"{name.strip()} placeholder was accepted: {value}")
        self.assertGreater(checked, 0, "no API key lines found in .env.example")


if __name__ == "__main__":
    unittest.main(verbosity=2)
