"""Conversational layer for WeatherGPT.

Owns the OpenAI chat loop and the tool definitions that expose
``weather_service`` to the model. Weather numbers always originate from a tool
result; the model's job is to phrase them, not to produce them.

The public entry point is :func:`chat`. It treats conversation history as
immutable: a turn either completes and returns an extended history, or fails
and returns the original history untouched. A partially-built turn (an
assistant message carrying ``tool_calls`` with no matching ``tool`` replies)
would make every later request fail, so it is never committed.
"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv

import config
import weather_service

load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
MAX_TOOL_ROUNDS = 4  # guards against a model looping on tool calls forever

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": (
                "Get the current weather conditions for a city or place. "
                "Use for questions about weather right now, today, or what to wear."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Delhi' or 'Paris, FR'.",
                    },
                    "units": {
                        "type": "string",
                        "enum": ["metric", "imperial"],
                        "description": "metric for Celsius, imperial for Fahrenheit.",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": (
                "Get a multi-day weather forecast for a city. Use for questions "
                "about tomorrow, this week, or whether it will rain later."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Mumbai'.",
                    },
                    "days": {
                        "type": "integer",
                        "description": (
                            "How many days to return, counting from today. "
                            "Day 1 is today, so use 2 to cover tomorrow and 3 "
                            "for a three-day outlook."
                        ),
                        "minimum": 1,
                        "maximum": 5,
                    },
                    "units": {
                        "type": "string",
                        "enum": ["metric", "imperial"],
                        "description": "metric for Celsius, imperial for Fahrenheit.",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_alerts",
            "description": (
                "Get active severe-weather alerts and warnings for a city. "
                "Use for questions about warnings, alerts, storms or hazards."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Chennai'.",
                    },
                },
                "required": ["location"],
            },
        },
    },
]

# Names only. The function is looked up on the module at call time rather than
# bound here at import, so the dispatch table cannot drift from what
# weather_service actually exposes.
TOOL_NAMES = ("get_current_weather", "get_forecast", "get_alerts")


def _resolve_tool(name: str):
    if name not in TOOL_NAMES:
        return None
    return getattr(weather_service, name, None)


def build_system_prompt(units: str = "metric") -> str:
    unit_name = "Fahrenheit and miles per hour" if units == "imperial" else "Celsius and metres per second"
    return (
        "You are WeatherGPT, a helpful conversational weather assistant.\n\n"
        f"Today's date is {date.today().isoformat()}.\n\n"
        "Use the available weather tools whenever current weather, forecasts, "
        "or weather alerts are requested. If the user asks about several places, "
        "call the tool once per place.\n\n"
        "Never fabricate weather measurements. Every number you state must come "
        "from a tool result. If a tool returns an 'error' field, explain the "
        "problem to the user in plain language and suggest what they can do; do "
        "not guess the weather instead.\n\n"
        "If an alerts result has 'alerts_available': false, say clearly that "
        "alerts could not be checked. Never report this as 'no alerts'.\n\n"
        "Forecast entries start with today and each carries a 'day_label' "
        "('today', 'tomorrow', or a weekday). Always use that label to decide "
        "which day you are describing - never infer it from position or date.\n\n"
        f"Report temperatures in {unit_name} unless the user asks otherwise.\n\n"
        "Keep answers short and conversational: two or three sentences, or a "
        "compact list for multi-day forecasts. Add a brief practical suggestion "
        "(clothing, umbrella, timing) when it is genuinely useful. Do not dump "
        "raw JSON at the user.\n\n"
        "For questions unrelated to weather, reply naturally without calling a "
        "weather tool."
    )


def _base_url() -> str:
    """Optional OpenAI-compatible endpoint (Groq, Gemini, Ollama, ...)."""
    return os.getenv("OPENAI_BASE_URL", "").strip()


def provider_name() -> str:
    """Human label for whichever provider is configured, for error messages."""
    base = _base_url()
    if not base:
        return "OpenAI"
    host = urlparse(base).hostname or base
    known = {
        "api.groq.com": "Groq",
        "generativelanguage.googleapis.com": "Google Gemini",
        "openrouter.ai": "OpenRouter",
        "localhost": "Ollama",
        "127.0.0.1": "Ollama",
    }
    return known.get(host, host)


def get_client() -> Tuple[Optional[Any], Optional[str]]:
    """Build the chat client, or explain why one cannot be built.

    Any OpenAI-compatible provider works: set OPENAI_BASE_URL to its endpoint.
    """
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not config.is_configured(key):
        return None, ("API key is not configured. "
                      "Add OPENAI_API_KEY to your .env file.")
    try:
        from openai import OpenAI
    except ImportError:
        return None, "The 'openai' package is not installed. Run: pip install -r requirements.txt"

    kwargs: Dict[str, Any] = {"api_key": key}
    base = _base_url()
    if base:
        kwargs["base_url"] = base
    try:
        return OpenAI(**kwargs), None
    except Exception as exc:  # pragma: no cover - client construction rarely fails
        return None, f"Could not initialise the API client: {type(exc).__name__}"


def run_tool(name: str, raw_arguments: str, default_units: str = "metric") -> Dict[str, Any]:
    """Execute one tool call and always return a JSON-serialisable dict."""
    func = _resolve_tool(name)
    if func is None:
        return {"error": f"Unknown tool: {name}"}

    try:
        args = json.loads(raw_arguments) if raw_arguments else {}
    except (json.JSONDecodeError, TypeError):
        return {"error": "The tool arguments were not valid JSON. "
                         "Please retry with a simple location name."}
    if not isinstance(args, dict):
        return {"error": "The tool arguments were not an object."}

    location = args.get("location")
    if not isinstance(location, str) or not location.strip():
        return {"error": "No location was provided. Please name a city."}

    kwargs: Dict[str, Any] = {"location": location}
    if name in ("get_current_weather", "get_forecast"):
        units = args.get("units")
        kwargs["units"] = units if units in ("metric", "imperial") else default_units
    if name == "get_forecast":
        kwargs["days"] = args.get("days", 3)

    try:
        return func(**kwargs)
    except Exception as exc:
        # weather_service is written not to raise; this is the last-resort net
        # so a bug there degrades to a message instead of killing the turn.
        return {"error": f"The weather lookup failed unexpectedly ({type(exc).__name__})."}


def _friendly_api_error(exc: Exception) -> str:
    """Turn an OpenAI exception into something a user can act on."""
    name = type(exc).__name__
    text = str(exc)
    lowered = text.lower()
    provider = provider_name()
    if "AuthenticationError" in name or "invalid_api_key" in lowered:
        return (f"The {provider} API key was rejected. Check that OPENAI_API_KEY "
                "in your .env file is valid.")
    # An exhausted balance also arrives as RateLimitError, but waiting will
    # never fix it, so it must not be reported as a rate limit.
    if any(marker in lowered for marker in
           ("insufficient_quota", "credit_balance_exhausted", "no credits")):
        billing = (" Add credits at "
                   "https://platform.openai.com/settings/organization/billing "
                   "to continue." if provider == "OpenAI" else
                   " Add credits or switch to a free provider.")
        return f"Your {provider} account has no API credits remaining.{billing}"
    if "RateLimitError" in name:
        return (f"Too many requests to {provider} right now. Wait a few seconds "
                "and try again.")
    if "NotFoundError" in name or "model_not_found" in text:
        return (f"The model '{DEFAULT_MODEL}' is not available on this API key. "
                "Set OPENAI_MODEL in your .env file to a model you have access to.")
    if "APIConnectionError" in name or "APITimeoutError" in name:
        return (f"Could not reach {provider}. Check your internet connection "
                "and try again.")
    if "BadRequestError" in name:
        return f"OpenAI rejected the request: {text[:200]}"
    return f"The assistant hit an unexpected error ({name}). Please try again."


def chat(
    user_message: str,
    history: Optional[List[Dict[str, Any]]] = None,
    units: str = "metric",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one conversational turn.

    ``history`` holds only prior turns in OpenAI message format (no system
    message; it is prepended fresh each call so the date and unit preference
    stay current).

    Returns ``{"reply", "history", "tool_calls", "error"}``. On failure the
    returned history is the caller's history plus the user message and a plain
    assistant reply, never a half-finished tool exchange.
    """
    history = list(history or [])

    if not isinstance(user_message, str) or not user_message.strip():
        return {
            "reply": "Please type a question first, for example: "
                     "\"What's the weather in Delhi?\"",
            "history": history,
            "tool_calls": [],
            "error": True,
        }

    user_message = user_message.strip()
    model = (model or DEFAULT_MODEL).strip()

    def _fail(message: str) -> Dict[str, Any]:
        return {
            "reply": message,
            "history": history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": message},
            ],
            "tool_calls": [],
            "error": True,
        }

    client, error = get_client()
    if client is None:
        return _fail(error or "The assistant is not configured.")

    # Build the turn in a scratch list; only merge into history on success.
    working: List[Dict[str, Any]] = history + [{"role": "user", "content": user_message}]
    tool_calls_made: List[Dict[str, Any]] = []

    for _ in range(MAX_TOOL_ROUNDS):
        messages = [{"role": "system", "content": build_system_prompt(units)}] + working
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.3,
            )
        except Exception as exc:
            return _fail(_friendly_api_error(exc))

        message = response.choices[0].message
        calls = message.tool_calls or []

        if not calls:
            reply = (message.content or "").strip() or "I'm not sure how to answer that."
            working.append({"role": "assistant", "content": reply})
            return {"reply": reply, "history": working,
                    "tool_calls": tool_calls_made, "error": False}

        # Record the assistant's tool request, then answer every call it made.
        working.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in calls
            ],
        })

        for call in calls:
            result = run_tool(call.function.name, call.function.arguments, units)
            tool_calls_made.append({
                "name": call.function.name,
                "arguments": call.function.arguments,
                "result": result,
            })
            working.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.function.name,
                "content": json.dumps(result, default=str),
            })

    return _fail("I got stuck looking that up. Could you rephrase the question?")
