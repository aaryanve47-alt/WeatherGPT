# ⛅ WeatherGPT

A conversational weather assistant. Ask about the weather in plain English and
get a real answer, grounded in live OpenWeatherMap data.

WeatherGPT uses an LLM with **tool calling**: the model decides which weather
lookup a question needs, calls it, and phrases the result. It never makes up
temperatures — every number it reports comes back from an API call.

```
You: What should I wear in Pune today?
WeatherGPT: Pune is 27°C right now with light clouds and it feels like 28°C.
            Comfortable for light clothing — no rain expected in the next few hours.
```

---

## Features

- **Current weather** — conditions, temperature, feels-like, humidity, wind
- **Forecasts** — up to 5 days, aggregated into daily highs, lows and rain chance
- **Severe-weather alerts** — active government warnings for a location
- **Natural language** — no commands or syntax to learn
- **Conversational context** — follow-ups like "what about tomorrow?" work
- **Grounded answers** — weather facts come from tool results, never invented
- **Honest failures** — unknown cities, API outages and missing keys are
  explained in plain language instead of crashing

---

## Architecture

```
User
 ↓
Streamlit  (app.py)          presentation only
 ↓
LLM        (llm_service.py)  intent → tool selection → phrasing
 ↓
Tool calling
 ↓
Weather    (weather_service.py)  HTTP, parsing, error normalisation
 ↓
OpenWeatherMap
 ↓
LLM  → natural-language answer
 ↓
User
```

The three layers stay separate on purpose:

| File                 | Responsibility                                              |
| -------------------- | ----------------------------------------------------------- |
| `weather_service.py` | Talks to OpenWeatherMap. Returns plain dicts. Never raises on an expected failure. |
| `llm_service.py`     | Owns the chat loop and tool schemas. Turns tool data into prose. |
| `app.py`             | Renders the conversation. Holds no weather or LLM logic.     |
| `config.py`          | One shared rule for whether a setting is real or still a placeholder. |

`weather_service.py` has no idea an LLM exists, and can be used as a normal
Python module on its own.

---

## Installation

Requires Python 3.9+.

```bash
python -m venv .venv
```

Activate it — Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Then install:

```bash
pip install -r requirements.txt
```

> **On Python 3.13/3.14**, if `pip` appears to hang while "collecting"
> packages, it is backtracking through source distributions that have no wheel
> for your Python version. Force wheels only:
>
> ```bash
> pip install --only-binary=:all: -r requirements.txt
> ```

---

## Environment variables

Copy the example file and fill in your own keys:

```bash
cp .env.example .env
```

### Weather (required)

```env
OPENWEATHER_API_KEY=your_key_here
```

Free tier from <https://home.openweathermap.org/api_keys> — enough for current
weather and forecasts.

### Chat model (required)

WeatherGPT talks to **any OpenAI-compatible API**, so you do not need a paid
OpenAI account. Set `OPENAI_BASE_URL` to point at the provider you want:

| Provider | Free? | `OPENAI_BASE_URL` | Suggested `OPENAI_MODEL` |
| -------- | ----- | ----------------- | ------------------------ |
| **Groq** | ✅ no card | `https://api.groq.com/openai/v1` | `openai/gpt-oss-120b` |
| **Google Gemini** | ✅ no card | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-2.0-flash` |
| **Ollama** (local) | ✅ no key | `http://localhost:11434/v1` | `llama3.1` |
| **OpenAI** | 💳 paid | *(leave unset)* | `gpt-4o-mini` |

Groq is the quickest to set up — a key from
<https://console.groq.com/keys> needs no credit card:

```env
OPENAI_API_KEY=your_groq_key_here
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=openai/gpt-oss-120b
```

For Ollama, install from <https://ollama.com>, run `ollama pull llama3.1`, and
set `OPENAI_API_KEY=ollama` (the value is ignored but the client requires one).

Leave `OPENAI_BASE_URL` unset to use OpenAI itself. Whichever provider you pick,
the model must support **tool calling** — that is how WeatherGPT gets its data.
The sidebar names the detected provider so you can confirm the setup at a glance.

`.env` is listed in `.gitignore` and must never be committed.

> A brand-new OpenWeatherMap key takes **10–60 minutes** to activate. Until it
> does, the API returns 401 and the app will tell you the key was rejected.

---

## Running

```bash
streamlit run app.py
```

The app opens at <http://localhost:8501>. The sidebar shows whether each key was
detected (a ✅/❌ only — key values are never displayed).

---

## Example prompts

```
What's the weather in Delhi right now?
Will it rain in Mumbai tomorrow?
Give me a 3-day forecast for Bangalore.
Are there any weather alerts for Chennai?
What should I wear in Pune today?
What's the weather in Delhi and Mumbai?
```

Follow-ups use conversation context:

```
You: What's the weather in Delhi?
You: What about tomorrow?
```

---

## Tests

The suite runs fully offline — every HTTP call and the chat client are
mocked, so no API keys or network access are needed:

```bash
python -m unittest discover -s tests -v
```

It covers response parsing, daily forecast aggregation across time zones,
every failure mode (timeout, 401, 429, 5xx, malformed JSON, unknown city), tool
dispatch and argument validation, multi-tool turns, and the invariant that
conversation history is never left in a state the OpenAI API would reject.

---

## A note on weather alerts

Current weather, forecasts and geocoding are all on OpenWeatherMap's **free**
tier. Government weather alerts are only served by the **One Call 3.0**
endpoint, which needs a separate subscription (free to start, but it requires a
card on file).

If your key does not have it, WeatherGPT says alerts *could not be checked* for
that location. It deliberately does **not** report "no alerts" — telling
someone there are no warnings when you never actually looked is the one failure
mode a weather app must not have. Everything else keeps working.

---

## Troubleshooting

| Symptom | Cause and fix |
| ------- | ------------- |
| Sidebar shows ❌ for a key | `.env` is missing, or still holds `your_key_here`. Restart the app after editing it. |
| "The OpenWeatherMap API key was rejected" | Key is wrong, or newly created and not yet active (wait up to an hour). |
| "The … API key was rejected" | Check `OPENAI_API_KEY` matches the provider named in `OPENAI_BASE_URL` — an OpenAI key will not work against Groq, and vice versa. |
| "has no API credits remaining" | The account is out of credit. Switch to a free provider (see the table above). |
| "The model … is not available on this API key" | Set `OPENAI_MODEL` in `.env` to a model your account can use. Providers retire models: list Groq's current ones with `curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.groq.com/openai/v1/models`. |
| "Too many requests … Wait a few seconds" | Provider rate limit. This one really does clear on its own — retry shortly. |
| "Could not find location: X" | The city name could not be geocoded. Try adding a country, e.g. `Springfield, US`. |
| `pip` hangs while collecting packages | See the `--only-binary=:all:` note under Installation. |
| Alerts always "could not be checked" | Expected without a One Call 3.0 subscription — see above. |

---

## Project structure

```
weathergpt/
├── app.py                        Streamlit UI
├── llm_service.py                LLM chat loop and tool definitions
├── weather_service.py            OpenWeatherMap client
├── config.py                     Shared "is this actually configured?" check
├── tests/
│   ├── test_weather_service.py   Parsing, aggregation, failure modes
│   ├── test_llm_service.py       Tool dispatch, history integrity
│   └── test_config.py            Placeholder detection
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Future scope

- Amazon Lex V2 + AWS Lambda as an alternative front end, reusing
  `weather_service.py` unchanged
- Hourly forecasts and air-quality data
- Location autocomplete for ambiguous city names
- Response streaming for faster perceived replies
