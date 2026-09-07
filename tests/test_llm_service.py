"""Offline tests for the conversation layer.

The OpenAI client is replaced with a scripted fake, so these assert on the
parts we actually wrote: tool dispatch, argument validation, multi-tool turns,
error handling, and the invariant that conversation history is never left in a
state the OpenAI API would reject.

Run with:  python -m unittest discover -s tests -v
"""

import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm_service  # noqa: E402


def make_tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def make_response(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class ScriptedClient:
    """Returns queued responses; records the messages it was called with."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        completions = SimpleNamespace(create=self._create)
        self.chat = SimpleNamespace(completions=completions)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("ScriptedClient ran out of responses")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class FailingClient:
    def __init__(self, exc):
        self._exc = exc
        completions = SimpleNamespace(create=self._create)
        self.chat = SimpleNamespace(completions=completions)

    def _create(self, **kwargs):
        raise self._exc


def assert_history_valid(testcase, history):
    """Every assistant tool_calls message must be answered by matching tool messages.

    This is the exact shape the OpenAI API rejects when it is wrong, and a
    corrupted history breaks every subsequent turn, not just the current one.
    """
    index = 0
    while index < len(history):
        message = history[index]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            expected_ids = [call["id"] for call in message["tool_calls"]]
            replies = history[index + 1: index + 1 + len(expected_ids)]
            testcase.assertEqual(
                [r.get("role") for r in replies], ["tool"] * len(expected_ids),
                f"tool_calls at index {index} were not followed by tool replies",
            )
            testcase.assertEqual(
                sorted(r.get("tool_call_id") for r in replies), sorted(expected_ids),
                "tool reply ids do not match the requested tool_call ids",
            )
            index += 1 + len(expected_ids)
        else:
            testcase.assertNotEqual(
                message.get("role"), "tool",
                f"orphaned tool message at index {index}",
            )
            index += 1


class TestRunTool(unittest.TestCase):

    def test_dispatches_to_weather_service(self):
        with patch.object(llm_service.weather_service, "get_current_weather",
                          return_value={"location": "Delhi", "temperature": 29}) as mock:
            result = llm_service.run_tool(
                "get_current_weather", json.dumps({"location": "Delhi"}))
        mock.assert_called_once()
        self.assertEqual(result["temperature"], 29)

    def test_forecast_passes_days_through(self):
        with patch.object(llm_service.weather_service, "get_forecast",
                          return_value={"forecast": []}) as mock:
            llm_service.run_tool("get_forecast",
                                 json.dumps({"location": "Mumbai", "days": 5}))
        self.assertEqual(mock.call_args.kwargs["days"], 5)

    def test_forecast_defaults_to_three_days(self):
        with patch.object(llm_service.weather_service, "get_forecast",
                          return_value={"forecast": []}) as mock:
            llm_service.run_tool("get_forecast", json.dumps({"location": "Mumbai"}))
        self.assertEqual(mock.call_args.kwargs["days"], 3)

    def test_units_fall_back_to_caller_default(self):
        with patch.object(llm_service.weather_service, "get_current_weather",
                          return_value={}) as mock:
            llm_service.run_tool("get_current_weather",
                                 json.dumps({"location": "Delhi"}),
                                 default_units="imperial")
        self.assertEqual(mock.call_args.kwargs["units"], "imperial")

    def test_invalid_units_are_rejected(self):
        with patch.object(llm_service.weather_service, "get_current_weather",
                          return_value={}) as mock:
            llm_service.run_tool("get_current_weather",
                                 json.dumps({"location": "Delhi", "units": "kelvin"}))
        self.assertEqual(mock.call_args.kwargs["units"], "metric")

    def test_alerts_tool_takes_no_units(self):
        with patch.object(llm_service.weather_service, "get_alerts",
                          return_value={"alerts": []}) as mock:
            llm_service.run_tool("get_alerts", json.dumps({"location": "Chennai"}))
        self.assertNotIn("units", mock.call_args.kwargs)

    def test_unknown_tool_returns_error(self):
        result = llm_service.run_tool("launch_rocket", json.dumps({"location": "Delhi"}))
        self.assertIn("Unknown tool", result["error"])

    def test_malformed_json_arguments_return_error(self):
        result = llm_service.run_tool("get_current_weather", "{not valid json")
        self.assertIn("not valid JSON", result["error"])

    def test_missing_location_returns_error(self):
        result = llm_service.run_tool("get_current_weather", json.dumps({}))
        self.assertIn("error", result)

    def test_non_object_arguments_return_error(self):
        result = llm_service.run_tool("get_current_weather", json.dumps(["Delhi"]))
        self.assertIn("error", result)

    def test_exception_in_weather_layer_is_contained(self):
        with patch.object(llm_service.weather_service, "get_current_weather",
                          side_effect=RuntimeError("boom")):
            result = llm_service.run_tool("get_current_weather",
                                          json.dumps({"location": "Delhi"}))
        self.assertIn("RuntimeError", result["error"])


class ChatTestCase(unittest.TestCase):

    def run_chat(self, responses, message="What's the weather in Delhi?", **kwargs):
        client = ScriptedClient(responses)
        with patch.object(llm_service, "get_client", return_value=(client, None)):
            result = llm_service.chat(message, **kwargs)
        return result, client


class TestPlainConversation(ChatTestCase):

    def test_non_weather_question_needs_no_tool(self):
        result, client = self.run_chat([make_response(content="Hello! How can I help?")],
                                       message="Hello")
        self.assertFalse(result["error"])
        self.assertEqual(result["reply"], "Hello! How can I help?")
        self.assertEqual(result["tool_calls"], [])
        self.assertEqual(len(client.calls), 1)
        assert_history_valid(self, result["history"])

    def test_history_records_user_then_assistant(self):
        result, _ = self.run_chat([make_response(content="Hi there")], message="Hello")
        roles = [m["role"] for m in result["history"]]
        self.assertEqual(roles, ["user", "assistant"])

    def test_system_prompt_is_sent_but_not_stored(self):
        result, client = self.run_chat([make_response(content="Hi")], message="Hello")
        self.assertEqual(client.calls[0]["messages"][0]["role"], "system")
        self.assertNotIn("system", [m["role"] for m in result["history"]])

    def test_empty_message_is_rejected_without_calling_the_model(self):
        client = ScriptedClient([])
        with patch.object(llm_service, "get_client", return_value=(client, None)):
            result = llm_service.chat("   ")
        self.assertTrue(result["error"])
        self.assertEqual(client.calls, [])
        self.assertEqual(result["history"], [])

    def test_blank_model_reply_falls_back_to_a_message(self):
        result, _ = self.run_chat([make_response(content="  ")], message="Hello")
        self.assertTrue(result["reply"])


class TestToolCalling(ChatTestCase):

    def test_single_tool_call_round_trip(self):
        responses = [
            make_response(tool_calls=[
                make_tool_call("call_1", "get_current_weather", {"location": "Delhi"})]),
            make_response(content="Delhi is 29°C with clear skies."),
        ]
        with patch.object(llm_service.weather_service, "get_current_weather",
                          return_value={"location": "Delhi", "temperature": 29}):
            result, client = self.run_chat(responses)

        self.assertFalse(result["error"])
        self.assertEqual(result["reply"], "Delhi is 29°C with clear skies.")
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["tool_calls"][0]["name"], "get_current_weather")
        assert_history_valid(self, result["history"])
        self.assertEqual([m["role"] for m in result["history"]],
                         ["user", "assistant", "tool", "assistant"])

    def test_tool_result_is_forwarded_to_the_model(self):
        responses = [
            make_response(tool_calls=[
                make_tool_call("call_1", "get_current_weather", {"location": "Delhi"})]),
            make_response(content="Delhi is 29°C."),
        ]
        with patch.object(llm_service.weather_service, "get_current_weather",
                          return_value={"location": "Delhi", "temperature": 29}):
            result, client = self.run_chat(responses)

        second_request = client.calls[1]["messages"]
        tool_message = [m for m in second_request if m.get("role") == "tool"][0]
        self.assertIn("29", tool_message["content"])

    def test_two_tool_calls_in_one_turn(self):
        """'Weather in Delhi and Mumbai?' - both calls must be answered."""
        responses = [
            make_response(tool_calls=[
                make_tool_call("call_1", "get_current_weather", {"location": "Delhi"}),
                make_tool_call("call_2", "get_current_weather", {"location": "Mumbai"}),
            ]),
            make_response(content="Delhi is 29°C, Mumbai is 31°C."),
        ]
        with patch.object(llm_service.weather_service, "get_current_weather",
                          side_effect=[{"temperature": 29}, {"temperature": 31}]):
            result, _ = self.run_chat(responses,
                                      message="What's the weather in Delhi and Mumbai?")

        self.assertEqual(len(result["tool_calls"]), 2)
        assert_history_valid(self, result["history"])
        tool_ids = [m["tool_call_id"] for m in result["history"] if m["role"] == "tool"]
        self.assertEqual(sorted(tool_ids), ["call_1", "call_2"])

    def test_tool_error_is_passed_to_model_not_raised(self):
        responses = [
            make_response(tool_calls=[
                make_tool_call("call_1", "get_current_weather", {"location": "Xyzabc"})]),
            make_response(content="I couldn't find that location."),
        ]
        with patch.object(llm_service.weather_service, "get_current_weather",
                          return_value={"error": "Could not find location: Xyzabc"}):
            result, client = self.run_chat(responses, message="Weather in Xyzabc?")

        self.assertFalse(result["error"])
        tool_message = [m for m in client.calls[1]["messages"] if m.get("role") == "tool"][0]
        self.assertIn("Could not find location", tool_message["content"])
        assert_history_valid(self, result["history"])

    def test_runaway_tool_loop_is_stopped(self):
        """A model that only ever calls tools must not loop forever."""
        looping = [
            make_response(tool_calls=[
                make_tool_call(f"call_{i}", "get_current_weather", {"location": "Delhi"})])
            for i in range(llm_service.MAX_TOOL_ROUNDS + 2)
        ]
        with patch.object(llm_service.weather_service, "get_current_weather",
                          return_value={"temperature": 29}):
            result, client = self.run_chat(looping)

        self.assertTrue(result["error"])
        self.assertEqual(len(client.calls), llm_service.MAX_TOOL_ROUNDS)
        assert_history_valid(self, result["history"])


class TestConversationMemory(ChatTestCase):

    def test_prior_history_is_replayed_to_the_model(self):
        history = [
            {"role": "user", "content": "What's the weather in Delhi?"},
            {"role": "assistant", "content": "Delhi is 29°C."},
        ]
        result, client = self.run_chat([make_response(content="Tomorrow looks similar.")],
                                       message="What about tomorrow?", history=history)
        sent = client.calls[0]["messages"]
        self.assertEqual(sent[1]["content"], "What's the weather in Delhi?")
        self.assertEqual(sent[-1]["content"], "What about tomorrow?")
        self.assertEqual(len(result["history"]), 4)

    def test_caller_history_is_not_mutated(self):
        history = [{"role": "user", "content": "Hello"},
                   {"role": "assistant", "content": "Hi"}]
        original = [dict(m) for m in history]
        self.run_chat([make_response(content="Sure")], history=history)
        self.assertEqual(history, original)

    def test_history_from_a_tool_turn_can_be_reused(self):
        """The history handed back must be accepted as input to the next turn."""
        first = [
            make_response(tool_calls=[
                make_tool_call("call_1", "get_current_weather", {"location": "Delhi"})]),
            make_response(content="Delhi is 29°C."),
        ]
        with patch.object(llm_service.weather_service, "get_current_weather",
                          return_value={"temperature": 29}):
            result_one, _ = self.run_chat(first)

        result_two, client = self.run_chat([make_response(content="Warmer tomorrow.")],
                                           message="And tomorrow?",
                                           history=result_one["history"])
        assert_history_valid(self, result_two["history"])
        assert_history_valid(self, client.calls[0]["messages"][1:])


class TestFailureHandling(ChatTestCase):

    def _failing(self, exc, history=None):
        client = FailingClient(exc)
        with patch.object(llm_service, "get_client", return_value=(client, None)):
            return llm_service.chat("What's the weather in Delhi?", history=history)

    def test_api_error_returns_friendly_message(self):
        result = self._failing(RuntimeError("connection reset"))
        self.assertTrue(result["error"])
        self.assertIn("unexpected error", result["reply"])

    def test_missing_key_is_reported_clearly(self):
        with patch.object(llm_service, "get_client",
                          return_value=(None, "OpenAI API key is not configured.")):
            result = llm_service.chat("Hello")
        self.assertTrue(result["error"])
        self.assertIn("not configured", result["reply"])

    def test_failure_leaves_history_usable(self):
        """A failed turn must not leave a dangling tool_calls message behind."""
        history = [{"role": "user", "content": "Hello"},
                   {"role": "assistant", "content": "Hi"}]
        result = self._failing(RuntimeError("boom"), history=history)
        assert_history_valid(self, result["history"])
        self.assertEqual([m["role"] for m in result["history"]],
                         ["user", "assistant", "user", "assistant"])

    def test_failure_midway_through_tools_does_not_corrupt_history(self):
        """Model asks for a tool, then the follow-up request fails."""
        responses = [
            make_response(tool_calls=[
                make_tool_call("call_1", "get_current_weather", {"location": "Delhi"})]),
            RuntimeError("network died"),
        ]
        with patch.object(llm_service.weather_service, "get_current_weather",
                          return_value={"temperature": 29}):
            result, _ = self.run_chat(responses)

        self.assertTrue(result["error"])
        assert_history_valid(self, result["history"])
        self.assertNotIn("tool", [m["role"] for m in result["history"]])

    def test_error_messages_map_to_actionable_text(self):
        cases = {
            "AuthenticationError": "rejected",
            "RateLimitError": "Too many requests",
            "NotFoundError": "not available",
            "APIConnectionError": "Could not reach OpenAI",
        }
        # Pin the provider: these messages name it, and a developer's real .env
        # must never change the outcome of a test.
        for name, expected in cases.items():
            with self.subTest(error=name):
                exc = type(name, (Exception,), {})("failure")
                with patch.dict(os.environ, {"OPENAI_BASE_URL": ""}):
                    message = llm_service._friendly_api_error(exc)
                self.assertIn(expected, message)

    def test_exhausted_credits_is_not_reported_as_a_rate_limit(self):
        """Waiting never fixes an empty balance, so it must not say 'try again'."""
        exc = type("RateLimitError", (Exception,), {})(
            "Error code: 429 - {'error': {'type': 'insufficient_quota', "
            "'code': 'credit_balance_exhausted'}}")
        message = llm_service._friendly_api_error(exc)
        self.assertIn("no API credits remaining", message)
        self.assertNotIn("Too many requests", message)


class TestProviderConfiguration(unittest.TestCase):
    """Any OpenAI-compatible provider must work via OPENAI_BASE_URL."""

    def test_defaults_to_openai_when_no_base_url(self):
        with patch.dict(os.environ, {"OPENAI_BASE_URL": ""}):
            self.assertEqual(llm_service.provider_name(), "OpenAI")

    def test_known_hosts_are_named(self):
        cases = {
            "https://api.groq.com/openai/v1": "Groq",
            "https://generativelanguage.googleapis.com/v1beta/openai/": "Google Gemini",
            "http://localhost:11434/v1": "Ollama",
        }
        for url, expected in cases.items():
            with self.subTest(url=url):
                with patch.dict(os.environ, {"OPENAI_BASE_URL": url}):
                    self.assertEqual(llm_service.provider_name(), expected)

    def test_base_url_is_passed_to_the_client(self):
        captured = {}

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "k",
                                     "OPENAI_BASE_URL": "https://api.groq.com/openai/v1"}):
            with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}):
                client, error = llm_service.get_client()
        self.assertIsNone(error)
        self.assertEqual(captured["base_url"], "https://api.groq.com/openai/v1")

    def test_base_url_is_omitted_when_unset(self):
        captured = {}

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "k", "OPENAI_BASE_URL": ""}):
            with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}):
                llm_service.get_client()
        self.assertNotIn("base_url", captured)

    def test_errors_name_the_configured_provider(self):
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://api.groq.com/openai/v1"}):
            exc = type("RateLimitError", (Exception,), {})("slow down")
            self.assertIn("Groq", llm_service._friendly_api_error(exc))

    def test_openai_billing_link_only_shown_for_openai(self):
        quota = type("RateLimitError", (Exception,), {})("insufficient_quota")
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://api.groq.com/openai/v1"}):
            self.assertNotIn("platform.openai.com",
                             llm_service._friendly_api_error(quota))


class TestToolSchema(unittest.TestCase):

    def test_every_declared_tool_is_dispatchable(self):
        declared = {tool["function"]["name"] for tool in llm_service.TOOLS}
        self.assertEqual(declared, set(llm_service.TOOL_NAMES))
        for name in llm_service.TOOL_NAMES:
            self.assertTrue(callable(llm_service._resolve_tool(name)), name)

    def test_tools_require_a_location(self):
        for tool in llm_service.TOOLS:
            with self.subTest(tool=tool["function"]["name"]):
                self.assertEqual(tool["function"]["parameters"]["required"], ["location"])

    def test_system_prompt_forbids_fabrication(self):
        prompt = llm_service.build_system_prompt()
        self.assertIn("Never fabricate", prompt)
        self.assertIn("alerts_available", prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
