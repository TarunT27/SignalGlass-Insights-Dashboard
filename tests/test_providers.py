"""Provider normalization, deterministic demo, and fallback contracts."""

from __future__ import annotations

from datetime import date, timedelta
from importlib import import_module

import pandas as pd
import pytest

EXPECTED_PRICE_COLUMNS = ["date", "Open", "High", "Low", "Close", "Volume", "ticker"]


def providers_module():
    return import_module("signalglass.providers")


def test_normalize_price_frame_flattens_yfinance_multiindex_columns() -> None:
    columns = pd.MultiIndex.from_tuples(
        [
            ("Open", "AAPL"),
            ("High", "AAPL"),
            ("Low", "AAPL"),
            ("Close", "AAPL"),
            ("Volume", "AAPL"),
        ],
        names=["Price", "Ticker"],
    )
    raw = pd.DataFrame(
        [[100.0, 102.0, 99.0, 101.0, 1_250_000]],
        index=pd.DatetimeIndex(["2025-01-02"], name="Date"),
        columns=columns,
    )

    result = providers_module().normalize_price_frame(raw, "aapl")

    assert list(result.columns) == EXPECTED_PRICE_COLUMNS
    assert result.loc[0, "ticker"] == "AAPL"
    assert result.loc[0, "Close"] == pytest.approx(101.0)
    assert pd.api.types.is_datetime64_any_dtype(result["date"])


def test_normalize_price_frame_accepts_flat_columns_and_removes_timezone() -> None:
    raw = pd.DataFrame(
        {
            "Date": pd.DatetimeIndex(["2025-01-02 16:00"], tz="America/New_York"),
            "Open": [100.0],
            "High": [102.0],
            "Low": [99.0],
            "Close": [101.0],
            "Adj Close": [100.5],
            "Volume": [1_250_000],
        }
    )

    result = providers_module().normalize_price_frame(raw, "AAPL")

    assert list(result.columns) == EXPECTED_PRICE_COLUMNS
    assert result["date"].dt.tz is None


def test_normalize_price_frame_rejects_missing_required_columns() -> None:
    raw = pd.DataFrame({"Date": ["2025-01-02"], "Close": [101.0]})

    with pytest.raises(ValueError, match="Open|High|Low|Volume|required|schema"):
        providers_module().normalize_price_frame(raw, "AAPL")


def test_demo_bundle_is_deterministic_complete_and_isolated() -> None:
    load_demo_bundle = providers_module().load_demo_bundle

    first = load_demo_bundle("AAPL", days=45)
    second = load_demo_bundle("AAPL", days=45)

    pd.testing.assert_frame_equal(first.prices, second.prices)
    pd.testing.assert_frame_equal(first.news, second.news)
    assert first.ticker == "AAPL"
    assert first.is_demo is True
    assert not first.prices.empty
    assert not first.news.empty
    assert list(first.prices.columns) == EXPECTED_PRICE_COLUMNS

    first.prices.loc[first.prices.index[0], "Close"] = -1
    third = load_demo_bundle("AAPL", days=45)
    assert (third.prices["Close"] > 0).all(), "Cached demo data must not leak caller mutations"


def test_fetch_market_bundle_without_key_uses_demo_without_network(
    monkeypatch: pytest.MonkeyPatch, valid_range
) -> None:
    providers = providers_module()

    def unexpected_network_call(*args, **kwargs):
        raise AssertionError("No-key startup should be deterministic and offline")

    if hasattr(providers, "yf"):
        monkeypatch.setattr(providers.yf, "download", unexpected_network_call)

    bundle = providers.fetch_market_bundle("AAPL", *valid_range, newsapi_key="")

    assert bundle.is_demo is True
    assert bundle.ticker == "AAPL"
    assert not bundle.prices.empty


def test_fetch_market_bundle_falls_back_when_live_provider_fails(
    monkeypatch: pytest.MonkeyPatch, valid_range
) -> None:
    providers = providers_module()

    def provider_failure(*args, **kwargs):
        raise TimeoutError("upstream timed out")

    if not hasattr(providers, "yf"):
        pytest.fail("providers must expose its yfinance adapter for bounded test substitution")
    monkeypatch.setattr(providers.yf, "download", provider_failure)

    bundle = providers.fetch_market_bundle("AAPL", *valid_range, newsapi_key="configured-for-test")

    assert bundle.is_demo is True
    assert not bundle.prices.empty


def _flat_live_price_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "Open": [100.0, 101.0],
            "High": [102.0, 104.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 103.0],
            "Volume": [1_250_000, 1_500_000],
        }
    )


def test_fetch_market_bundle_returns_normalized_live_prices_and_news(
    monkeypatch: pytest.MonkeyPatch, valid_range
) -> None:
    providers = providers_module()
    monkeypatch.setattr(providers.yf, "download", lambda *args, **kwargs: _flat_live_price_frame())

    class StubNewsClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == "test-key"

        def get_everything(self, **kwargs):
            assert kwargs["q"] == "AAPL"
            return {
                "articles": [
                    {
                        "title": "Apple posts record growth",
                        "description": "Strong demand helped results beat estimates.",
                        "url": "https://example.test/apple-growth",
                        "publishedAt": "2025-01-03T13:00:00Z",
                        "source": {"name": "Example Wire"},
                    },
                    {
                        "title": "Undated item",
                        "publishedAt": "not-a-date",
                        "source": {"name": "Example Wire"},
                    },
                ]
            }

    monkeypatch.setattr(providers, "NewsApiClient", StubNewsClient)

    bundle = providers.fetch_market_bundle("AAPL", *valid_range, newsapi_key="test-key")

    assert bundle.is_demo is False
    assert "Yahoo Finance" in bundle.source
    assert "NewsAPI" in bundle.source
    assert bundle.prices["Close"].tolist() == [101.0, 103.0]
    assert bundle.news["source"].tolist() == ["Example Wire"]
    assert bundle.news["sentiment_label"].tolist() == ["Positive"]
    assert bundle.news["sentiment_score"].iloc[0] > 0
    assert "beat estimates" in bundle.news["sentiment_evidence"].iloc[0]


def test_live_utc_headline_uses_new_york_time_for_session_alignment(
    monkeypatch: pytest.MonkeyPatch, valid_range
) -> None:
    providers = providers_module()
    analytics = import_module("signalglass.analytics")
    monkeypatch.setattr(providers.yf, "download", lambda *args, **kwargs: _flat_live_price_frame())

    class StubNewsClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == "test-key"

        def get_everything(self, **kwargs):
            return {
                "articles": [
                    {
                        "title": "Apple outlines product roadmap",
                        "description": "Investors review the announcement.",
                        "url": "https://example.test/apple-roadmap",
                        "publishedAt": "2025-01-03T18:00:00Z",
                        "source": {"name": "Example Wire"},
                    }
                ]
            }

    monkeypatch.setattr(providers, "NewsApiClient", StubNewsClient)

    bundle = providers.fetch_market_bundle("AAPL", *valid_range, newsapi_key="test-key")
    assert bundle.news.loc[0, "publishedAt"] == pd.Timestamp("2025-01-03 13:00:00")
    summary = analytics.prepare_sentiment_summary(bundle.news)
    combined = analytics.merge_market_and_sentiment(bundle.prices, summary)
    friday = combined.loc[combined["date"] == pd.Timestamp("2025-01-03")].iloc[0]

    assert friday["headline_count"] == 1


def test_prefer_live_prices_works_without_newsapi_key_and_labels_synthetic_news(
    monkeypatch: pytest.MonkeyPatch, valid_range
) -> None:
    providers = providers_module()
    monkeypatch.setattr(providers.yf, "download", lambda *args, **kwargs: _flat_live_price_frame())

    bundle = providers.fetch_market_bundle(
        "AAPL",
        *valid_range,
        newsapi_key="",
        prefer_live_prices=True,
    )

    assert bundle.prices["Close"].tolist() == [101.0, 103.0]
    assert not bundle.news.empty
    provenance = f"{bundle.source} {bundle.notice}".lower()
    assert "yahoo" in provenance or "live price" in provenance
    assert "demo" in provenance or "synthetic" in provenance


def test_prefer_live_prices_falls_back_atomically_when_yfinance_fails(
    monkeypatch: pytest.MonkeyPatch, valid_range
) -> None:
    providers = providers_module()

    def provider_failure(*args, **kwargs):
        raise TimeoutError("upstream timed out")

    monkeypatch.setattr(providers.yf, "download", provider_failure)

    bundle = providers.fetch_market_bundle(
        "AAPL",
        *valid_range,
        newsapi_key="",
        prefer_live_prices=True,
    )

    assert bundle.is_demo is True
    assert not bundle.prices.empty
    assert "TimeoutError" in str(bundle.notice)


def test_newsapi_search_stays_inside_the_plan_lookback_window(monkeypatch: pytest.MonkeyPatch) -> None:
    providers = providers_module()
    monkeypatch.setattr(providers.yf, "download", lambda *args, **kwargs: _flat_live_price_frame())
    requested: dict[str, str] = {}

    class StubNewsClient:
        def __init__(self, api_key: str) -> None:
            pass

        def get_everything(self, **kwargs):
            requested.update(kwargs)
            return {"articles": []}

    monkeypatch.setattr(providers, "NewsApiClient", StubNewsClient)
    end = date(2025, 6, 30)

    bundle = providers.fetch_market_bundle("AAPL", date(2024, 6, 30), end, newsapi_key="test-key")

    assert requested["from_param"] == (end - timedelta(days=providers.NEWSAPI_MAX_LOOKBACK_DAYS)).isoformat()
    assert requested["to"] == end.isoformat()
    assert bundle.news_source == "NewsAPI live headlines"


def test_company_name_lookup_failures_are_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    providers = providers_module()
    providers._lookup_company_name.cache_clear()
    calls = {"count": 0}

    class FlakyTicker:
        def __init__(self, symbol: str) -> None:
            pass

        def get_info(self):
            calls["count"] += 1
            if calls["count"] == 1:
                raise TimeoutError("upstream timed out")
            return {"shortName": "SignalGlass Test Corp"}

    monkeypatch.setattr(providers.yf, "Ticker", FlakyTicker)

    assert providers.resolve_company_name("SGTEST") == "SGTEST"
    assert providers.resolve_company_name("SGTEST") == "SignalGlass Test Corp"
    assert providers.resolve_company_name("SGTEST") == "SignalGlass Test Corp"
    assert calls["count"] == 2
    providers._lookup_company_name.cache_clear()
