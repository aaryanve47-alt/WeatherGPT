"""Offline tests for the weather layer.

Every OpenWeatherMap call is mocked with a realistic payload, so these run
without API keys or network access and assert on parsing, daily aggregation
and each failure mode.

Run with:  python -m unittest discover -s tests -v
"""

import os
import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import weather_service  # noqa: E402

FAKE_KEY = "test-key-not-a-real-secret"

GEO_DELHI = [{"name": "Delhi", "lat": 28.6517, "lon": 77.2219, "country": "IN", "state": "Delhi"}]

CURRENT_DELHI = {
    "weather": [{"main": "Clear", "description": "clear sky"}],
    "main": {"temp": 29.4, "feels_like": 30.1, "temp_min": 28.0, "temp_max": 31.0,
             "humidity": 48, "pressure": 1008},
    "wind": {"speed": 3.6, "deg": 270},
    "clouds": {"all": 5},
    "visibility": 6000,
    "dt": 1757260800,
    "timezone": 19800,
}


def _forecast_slot(dt, temp, pop, main="Rain", desc="light rain"):
    return {
        "dt": dt,
        "main": {"temp": temp, "humidity": 70},
        "weather": [{"main": main, "description": desc}],
        "wind": {"speed": 4.1},
        "pop": pop,
    }


IST_OFFSET = 19800  # Mumbai, UTC+5:30


def _ts(local_iso: str, tz_offset: int = IST_OFFSET) -> int:
    """UNIX timestamp for a wall-clock local time at the given UTC offset.

    Derived rather than hard-coded: a literal timestamp silently lands on an
    unintended local day, which is exactly the bug this fixture must not have.
    """
    naive = datetime.fromisoformat(local_iso)
    return int(naive.replace(tzinfo=timezone.utc).timestamp()) - tz_offset


# Three local days, two slots each, so day boundaries are unambiguous.
FORECAST_MUMBAI = {
    "city": {"name": "Mumbai", "timezone": IST_OFFSET},
    "list": [
        _forecast_slot(_ts("2025-09-08T09:00"), 26.0, 0.2),
        _forecast_slot(_ts("2025-09-08T15:00"), 31.5, 0.8),
        _forecast_slot(_ts("2025-09-09T09:00"), 25.0, 0.1),
        _forecast_slot(_ts("2025-09-09T15:00"), 30.0, 0.4),
        _forecast_slot(_ts("2025-09-10T09:00"), 24.5, 0.0),
        _forecast_slot(_ts("2025-09-10T15:00"), 29.0, 0.6),
    ],
}

ALERTS_PAYLOAD = {
    "timezone_offset": 19800,
    "alerts": [{
        "sender_name": "India Meteorological Department",
        "event": "Heavy Rain Warning",
        "start": 1757260800,
        "end": 1757347200,
        "description": "Heavy rainfall expected in coastal areas.",
        "tags": ["Rain"],
    }],
}


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code=200, payload=None, malformed=False):
        self.status_code = status_code
        self._payload = payload
        self._malformed = malformed

    def json(self):
        if self._malformed:
            raise ValueError("No JSON object could be decoded")
        return self._payload


class WeatherServiceTestCase(unittest.TestCase):
    """Shared setup: a fake key, and a clean geocode cache per test."""

    def setUp(self):
        weather_service._geocode_cache.clear()
        self._env = patch.dict(os.environ, {"OPENWEATHER_API_KEY": FAKE_KEY})
        self._env.start()
        self.addCleanup(self._env.stop)

    def route(self, responses):
        """Return a requests.get replacement that dispatches on URL."""
        def fake_get(url, params=None, timeout=None):
            for fragment, response in responses.items():
                if fragment in url:
                    if isinstance(response, Exception):
                        raise response
                    return response
            raise AssertionError(f"Unexpected URL requested: {url}")
        return fake_get


class TestGeocoding(WeatherServiceTestCase):

    def test_resolves_city_to_coordinates(self):
        with patch("requests.get", self.route({"geo/1.0/direct": FakeResponse(200, GEO_DELHI)})):
            result = weather_service.geocode_location("Delhi")
        self.assertNotIn("error", result)
        self.assertEqual(result["lat"], 28.6517)
        self.assertEqual(result["label"], "Delhi, Delhi, IN")

    def test_unknown_city_returns_structured_error(self):
        with patch("requests.get", self.route({"geo/1.0/direct": FakeResponse(200, [])})):
            result = weather_service.geocode_location("Xyzabc")
        self.assertEqual(result["error"], "Could not find location: Xyzabc")

    def test_empty_input_rejected_without_network_call(self):
        with patch("requests.get", side_effect=AssertionError("should not call network")):
            result = weather_service.geocode_location("   ")
        self.assertIn("error", result)

    def test_results_are_cached(self):
        calls = []

        def counting_get(url, params=None, timeout=None):
            calls.append(url)
            return FakeResponse(200, GEO_DELHI)

        with patch("requests.get", counting_get):
            weather_service.geocode_location("Delhi")
            weather_service.geocode_location("delhi")
        self.assertEqual(len(calls), 1)

    def test_cache_cannot_be_mutated_by_caller(self):
        with patch("requests.get", self.route({"geo/1.0/direct": FakeResponse(200, GEO_DELHI)})):
            first = weather_service.geocode_location("Delhi")
            first["lat"] = 0.0
            second = weather_service.geocode_location("Delhi")
        self.assertEqual(second["lat"], 28.6517)


class TestCurrentWeather(WeatherServiceTestCase):

    def test_returns_structured_current_conditions(self):
        with patch("requests.get", self.route({
            "geo/1.0/direct": FakeResponse(200, GEO_DELHI),
            "data/2.5/weather": FakeResponse(200, CURRENT_DELHI),
        })):
            result = weather_service.get_current_weather("Delhi")

        self.assertNotIn("error", result)
        self.assertEqual(result["location"], "Delhi, Delhi, IN")
        self.assertEqual(result["temperature"], 29.4)
        self.assertEqual(result["humidity"], 48)
        self.assertEqual(result["condition"], "Clear")
        self.assertEqual(result["units"]["temperature"], "°C")
        self.assertIn("local", result["observed_at"])

    def test_imperial_units_labelled(self):
        with patch("requests.get", self.route({
            "geo/1.0/direct": FakeResponse(200, GEO_DELHI),
            "data/2.5/weather": FakeResponse(200, CURRENT_DELHI),
        })):
            result = weather_service.get_current_weather("Delhi", units="imperial")
        self.assertEqual(result["units"]["temperature"], "°F")

    def test_missing_fields_do_not_crash(self):
        sparse = {"main": {"temp": 20.0}, "dt": 1757260800, "timezone": 0}
        with patch("requests.get", self.route({
            "geo/1.0/direct": FakeResponse(200, GEO_DELHI),
            "data/2.5/weather": FakeResponse(200, sparse),
        })):
            result = weather_service.get_current_weather("Delhi")
        self.assertEqual(result["temperature"], 20.0)
        self.assertIsNone(result["condition"])
        self.assertIsNone(result["wind_speed"])

    def test_geocode_failure_short_circuits(self):
        with patch("requests.get", self.route({"geo/1.0/direct": FakeResponse(200, [])})):
            result = weather_service.get_current_weather("Xyzabc")
        self.assertIn("Could not find location", result["error"])


class TestForecast(WeatherServiceTestCase):

    def _forecast(self, days=3):
        with patch("requests.get", self.route({
            "geo/1.0/direct": FakeResponse(200, GEO_DELHI),
            "data/2.5/forecast": FakeResponse(200, FORECAST_MUMBAI),
        })):
            return weather_service.get_forecast("Mumbai", days)

    def test_returns_requested_number_of_days(self):
        result = self._forecast(3)
        self.assertNotIn("error", result)
        self.assertEqual(len(result["forecast"]), 3)

    def test_buckets_slots_by_local_calendar_day(self):
        """Slots must group by the location's local date, not by UTC date."""
        dates = [day["date"] for day in self._forecast(3)["forecast"]]
        self.assertEqual(dates, ["2025-09-08", "2025-09-09", "2025-09-10"])

    def test_aggregates_min_and_max_across_slots(self):
        day = self._forecast(3)["forecast"][0]
        self.assertEqual(day["temp_min"], 26.0)
        self.assertEqual(day["temp_max"], 31.5)

    def test_converts_pop_to_percentage(self):
        day = self._forecast(3)["forecast"][0]
        self.assertEqual(day["chance_of_rain_percent"], 80)

    def test_days_are_clamped_to_supported_range(self):
        self.assertEqual(len(self._forecast(99)["forecast"]), 3)  # only 3 days of data
        self.assertEqual(self._forecast(99)["days_requested"], 5)
        self.assertEqual(self._forecast(0)["days_requested"], 1)

    def test_non_numeric_days_falls_back_to_three(self):
        self.assertEqual(self._forecast("abc")["days_requested"], 3)

    def test_days_are_labelled_relative_to_today(self):
        """Guards a real bug: today's entry was reported as 'tomorrow'."""
        self.assertEqual(weather_service._day_label("2026-09-07", "2026-09-07"), "today")
        self.assertEqual(weather_service._day_label("2026-09-08", "2026-09-07"), "tomorrow")
        self.assertEqual(weather_service._day_label("2026-09-10", "2026-09-07"), "Thursday")

    def test_distant_and_malformed_days_fall_back_to_the_date(self):
        self.assertEqual(weather_service._day_label("2026-10-01", "2026-09-07"), "2026-10-01")
        self.assertEqual(weather_service._day_label("not-a-date", "2026-09-07"), "not-a-date")

    def test_forecast_entries_carry_a_day_label(self):
        result = self._forecast(3)
        for day in result["forecast"]:
            self.assertIn("day_label", day)

    def test_today_is_computed_in_the_location_timezone(self):
        """At 23:00 UTC it is already tomorrow in Mumbai; the label must agree."""
        # The fixture covers 2025-09-08..10 local. 23:00 UTC on the 7th is
        # already 04:30 on the 8th in Mumbai, so the 8th must read as "today".
        fixed = datetime(2025, 9, 7, 23, 0, tzinfo=timezone.utc)

        class FrozenDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed if tz else fixed.replace(tzinfo=None)

        with patch.object(weather_service, "datetime", FrozenDatetime):
            labels = weather_service._aggregate_daily(
                FORECAST_MUMBAI["list"], IST_OFFSET)
        by_date = {day["date"]: day["day_label"] for day in labels}
        self.assertEqual(by_date["2025-09-08"], "today")
        self.assertEqual(by_date["2025-09-09"], "tomorrow")

    def test_empty_forecast_list_reports_error(self):
        with patch("requests.get", self.route({
            "geo/1.0/direct": FakeResponse(200, GEO_DELHI),
            "data/2.5/forecast": FakeResponse(200, {"city": {"timezone": 0}, "list": []}),
        })):
            result = weather_service.get_forecast("Mumbai", 3)
        self.assertIn("error", result)

    def test_malformed_slots_are_skipped(self):
        payload = {"city": {"timezone": 0},
                   "list": [{"garbage": True}, _forecast_slot(1757260800, 22.0, 0.1)]}
        with patch("requests.get", self.route({
            "geo/1.0/direct": FakeResponse(200, GEO_DELHI),
            "data/2.5/forecast": FakeResponse(200, payload),
        })):
            result = weather_service.get_forecast("Mumbai", 3)
        self.assertEqual(len(result["forecast"]), 1)


class TestAlerts(WeatherServiceTestCase):

    def test_parses_active_alerts(self):
        with patch("requests.get", self.route({
            "geo/1.0/direct": FakeResponse(200, GEO_DELHI),
            "data/3.0/onecall": FakeResponse(200, ALERTS_PAYLOAD),
        })):
            result = weather_service.get_alerts("Chennai")
        self.assertTrue(result["alerts_available"])
        self.assertEqual(result["alert_count"], 1)
        self.assertEqual(result["alerts"][0]["event"], "Heavy Rain Warning")

    def test_no_alerts_returns_empty_list_and_available_true(self):
        with patch("requests.get", self.route({
            "geo/1.0/direct": FakeResponse(200, GEO_DELHI),
            "data/3.0/onecall": FakeResponse(200, {"timezone_offset": 0}),
        })):
            result = weather_service.get_alerts("Chennai")
        self.assertTrue(result["alerts_available"])
        self.assertEqual(result["alerts"], [])

    def test_unsubscribed_key_reports_unavailable_not_empty(self):
        """A 401 from One Call must never look like 'no alerts'."""
        with patch("requests.get", self.route({
            "geo/1.0/direct": FakeResponse(200, GEO_DELHI),
            "data/3.0/onecall": FakeResponse(401, {"message": "Invalid API key"}),
        })):
            result = weather_service.get_alerts("Chennai")
        self.assertFalse(result["alerts_available"])
        self.assertIn("could not be checked", result["note"])


class TestFailureModes(WeatherServiceTestCase):

    def test_timeout_is_friendly(self):
        with patch("requests.get", self.route({
            "geo/1.0/direct": requests.exceptions.Timeout(),
        })):
            result = weather_service.get_current_weather("Delhi")
        self.assertIn("too long", result["error"])

    def test_connection_error_is_friendly(self):
        with patch("requests.get", self.route({
            "geo/1.0/direct": requests.exceptions.ConnectionError(),
        })):
            result = weather_service.get_current_weather("Delhi")
        self.assertIn("Could not reach", result["error"])

    def test_malformed_json_is_friendly(self):
        with patch("requests.get", self.route({
            "geo/1.0/direct": FakeResponse(200, None, malformed=True),
        })):
            result = weather_service.get_current_weather("Delhi")
        self.assertIn("malformed", result["error"])

    def test_invalid_api_key_is_friendly(self):
        with patch("requests.get", self.route({
            "geo/1.0/direct": FakeResponse(401, {"message": "Invalid API key"}),
        })):
            result = weather_service.get_current_weather("Delhi")
        self.assertIn("rejected", result["error"])

    def test_rate_limit_is_friendly(self):
        with patch("requests.get", self.route({
            "geo/1.0/direct": FakeResponse(429, {}),
        })):
            result = weather_service.get_current_weather("Delhi")
        self.assertIn("rate limit", result["error"])

    def test_server_error_is_friendly(self):
        with patch("requests.get", self.route({
            "geo/1.0/direct": FakeResponse(503, {}),
        })):
            result = weather_service.get_current_weather("Delhi")
        self.assertIn("temporarily unavailable", result["error"])

    def test_missing_api_key_reports_configuration_problem(self):
        with patch.dict(os.environ, {"OPENWEATHER_API_KEY": ""}):
            with patch("requests.get", side_effect=AssertionError("should not call network")):
                result = weather_service.get_current_weather("Delhi")
        self.assertIn("not configured", result["error"])

    def test_placeholder_api_key_treated_as_missing(self):
        with patch.dict(os.environ, {"OPENWEATHER_API_KEY": "your_key_here"}):
            with patch("requests.get", side_effect=AssertionError("should not call network")):
                result = weather_service.get_current_weather("Delhi")
        self.assertIn("not configured", result["error"])

    def test_errors_never_leak_the_api_key(self):
        cases = [
            FakeResponse(401, {}), FakeResponse(429, {}), FakeResponse(500, {}),
            FakeResponse(200, None, malformed=True), requests.exceptions.Timeout(),
        ]
        for response in cases:
            with self.subTest(response=response):
                with patch("requests.get", self.route({"geo/1.0/direct": response})):
                    result = weather_service.get_current_weather("Delhi")
                self.assertNotIn(FAKE_KEY, str(result))


if __name__ == "__main__":
    unittest.main(verbosity=2)
