"""WeatherGPT - a conversational weather assistant.

Streamlit front end. This module owns presentation only: it collects input,
renders the conversation, and delegates every decision to ``llm_service.chat``.

Two parallel records of the conversation are kept in session state:
  * ``messages``  - what the user sees (user/assistant text only)
  * ``history``   - full OpenAI message list including tool calls and results

They are kept separate so tool plumbing never leaks into the UI, and so the
transcript the model sees stays valid for the API.
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

import config
import llm_service

load_dotenv()

st.set_page_config(
    page_title="WeatherGPT",
    page_icon="⛅",
    layout="centered",
    initial_sidebar_state="expanded",
)

EXAMPLE_PROMPTS = [
    "What's the weather in Delhi right now?",
    "Will it rain in Mumbai tomorrow?",
    "Give me a 3-day forecast for Bangalore.",
    "Are there any weather alerts for Chennai?",
    "What should I wear in Pune today?",
]

CSS = """
<style>
  /* Clears Streamlit's fixed top toolbar, which otherwise crops the title. */
  .block-container { padding-top: 4.5rem; max-width: 46rem; }
  .wx-title {
      font-size: 2.1rem; font-weight: 700; letter-spacing: -0.02em;
      margin-bottom: 0.15rem;
  }
  .wx-subtitle {
      color: #6b7280; font-size: 0.97rem; margin-bottom: 1.4rem;
  }
  .wx-empty {
      border: 1px solid rgba(128, 128, 128, 0.25); border-radius: 12px;
      padding: 1.1rem 1.25rem; margin-bottom: 0.5rem;
      background: rgba(128, 128, 128, 0.06);
  }
  .wx-empty h4 { margin: 0 0 0.5rem 0; font-size: 0.95rem; }
  .wx-empty ul { margin: 0; padding-left: 1.1rem; color: #6b7280; }
  .wx-empty li { margin-bottom: 0.28rem; font-size: 0.92rem; }
  [data-testid="stSidebar"] .stCaption { line-height: 1.35; }
</style>
"""


def key_status() -> tuple[bool, bool]:
    """Report whether each key is configured. Never returns the values."""
    def configured(name: str) -> bool:
        return config.is_configured(os.getenv(name, ""))

    return configured("OPENWEATHER_API_KEY"), configured("OPENAI_API_KEY")


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("pending", None)


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("### ⛅ WeatherGPT")
        st.caption("Live weather, asked in plain English.")

        st.divider()
        unit_label = st.radio(
            "Units",
            ["Celsius (°C)", "Fahrenheit (°F)"],
            index=0,
            help="Applies to new questions.",
        )
        units = "imperial" if unit_label.startswith("Fahrenheit") else "metric"

        st.divider()
        st.markdown("**Setup**")
        weather_ok, openai_ok = key_status()
        provider = llm_service.provider_name()
        st.caption(("✅ " if weather_ok else "❌ ") + "OpenWeatherMap key")
        st.caption(("✅ " if openai_ok else "❌ ") + f"{provider} key")
        if not (weather_ok and openai_ok):
            st.caption("Add the missing key(s) to your `.env` file, then restart the app.")
        st.caption(f"Model: `{llm_service.DEFAULT_MODEL}`")

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()

    return units


def render_empty_state() -> None:
    items = "".join(f"<li>{prompt}</li>" for prompt in EXAMPLE_PROMPTS)
    st.markdown(
        f'<div class="wx-empty"><h4>Try asking</h4><ul>{items}</ul></div>',
        unsafe_allow_html=True,
    )


def render_tool_details(tool_calls: list) -> None:
    """Show which live lookups backed the answer, without dumping raw JSON."""
    if not tool_calls:
        return
    names = ", ".join(call["name"] for call in tool_calls)
    with st.expander(f"🔎 Live data used ({len(tool_calls)}): {names}"):
        for call in tool_calls:
            st.caption(f"**{call['name']}** · `{call['arguments']}`")
            result = call.get("result") or {}
            if isinstance(result, dict) and "error" in result:
                st.warning(result["error"])
            else:
                st.json(result, expanded=False)


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    units = render_sidebar()

    st.markdown('<div class="wx-title">⛅ WeatherGPT</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="wx-subtitle">Ask about current conditions, forecasts, or '
        'severe-weather alerts anywhere in the world. Answers come from live '
        'OpenWeatherMap data.</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        render_empty_state()

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="⛅" if message["role"] == "assistant" else None):
            st.markdown(message["content"])
            render_tool_details(message.get("tool_calls") or [])

    prompt = st.chat_input("Ask about the weather anywhere…")
    if not prompt or not prompt.strip():
        return

    prompt = prompt.strip()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="⛅"):
        with st.spinner("Checking the weather…"):
            try:
                result = llm_service.chat(
                    prompt,
                    history=st.session_state.history,
                    units=units,
                )
            except Exception as exc:
                # llm_service is written not to raise; this keeps a bug there
                # from leaving the app in a broken state.
                result = {
                    "reply": f"Something went wrong handling that question "
                             f"({type(exc).__name__}). Please try again.",
                    "history": st.session_state.history,
                    "tool_calls": [],
                    "error": True,
                }

        st.markdown(result["reply"])
        render_tool_details(result.get("tool_calls") or [])

    st.session_state.history = result["history"]
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["reply"],
        "tool_calls": result.get("tool_calls") or [],
    })


if __name__ == "__main__":
    main()
