import datetime
import pandas as pd
import pytest
import data_loader

from data_loader import BTCDataLoader, InsufficientDataError


def make_price_df(start_date="2020-01-01", periods=2200):
    dates = pd.date_range(start_date, periods=periods, freq="D")
    df = pd.DataFrame(index=dates)
    df["Open"] = range(len(dates))
    df["High"] = [x + 2 for x in range(len(dates))]
    df["Low"] = [x for x in range(len(dates))]
    df["Close"] = [10000 + x for x in range(len(dates))]
    return df


def make_fx_df(close_values):
    dates = pd.date_range("2026-01-01", periods=len(close_values), freq="D")
    df = pd.DataFrame(index=dates)
    df["Close"] = close_values
    return df


def test_data_loader_success(cfg, monkeypatch):
    fake_df = make_price_df()

    def fake_download(*args, **kwargs):
        return fake_df.copy()

    monkeypatch.setattr(data_loader.yf, "download", fake_download)

    loader = BTCDataLoader(cfg)
    btc, yesterday_london, available_years = loader.load("BTC-USD", 3)

    assert "Price" in btc.columns
    assert isinstance(yesterday_london, datetime.date)
    assert available_years >= 3.0
    assert not btc.empty


def test_data_loader_success_with_multiindex_columns(cfg, monkeypatch):
    base_df = make_price_df()
    multi_df = pd.DataFrame(
        {
            ("Open", "BTC-USD"): base_df["Open"],
            ("High", "BTC-USD"): base_df["High"],
            ("Low", "BTC-USD"): base_df["Low"],
            ("Close", "BTC-USD"): base_df["Close"],
        },
        index=base_df.index,
    )

    def fake_download(*args, **kwargs):
        return multi_df.copy()

    monkeypatch.setattr(data_loader.yf, "download", fake_download)

    loader = BTCDataLoader(cfg)
    btc, yesterday_london, available_years = loader.load("BTC-USD", 3)

    assert "Open" in btc.columns
    assert "High" in btc.columns
    assert "Low" in btc.columns
    assert "Close" in btc.columns
    assert "Price" in btc.columns
    assert isinstance(yesterday_london, datetime.date)
    assert available_years >= 3.0
    assert not btc.empty


def test_data_loader_not_enough_data(cfg, monkeypatch):
    fake_df = make_price_df(start_date="2025-01-01", periods=100)

    def fake_download(*args, **kwargs):
        return fake_df.copy()

    monkeypatch.setattr(data_loader.yf, "download", fake_download)

    loader = BTCDataLoader(cfg)

    with pytest.raises(InsufficientDataError):
        loader.load("BTC-USD", 3)


def test_data_loader_empty_download(cfg, monkeypatch):
    def fake_download(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(data_loader.yf, "download", fake_download)

    loader = BTCDataLoader(cfg)

    with pytest.raises(ValueError, match="No data was found"):
        loader.load("BTC-USD", 3)


def test_data_loader_missing_required_columns(cfg, monkeypatch):
    dates = pd.date_range("2020-01-01", periods=2200, freq="D")
    fake_df = pd.DataFrame(index=dates)
    fake_df["Open"] = range(len(dates))
    fake_df["High"] = [x + 2 for x in range(len(dates))]
    fake_df["Low"] = [x for x in range(len(dates))]

    def fake_download(*args, **kwargs):
        return fake_df.copy()

    monkeypatch.setattr(data_loader.yf, "download", fake_download)

    loader = BTCDataLoader(cfg)

    with pytest.raises(ValueError, match="missing required columns"):
        loader.load("BTC-USD", 3)


def test_load_usdgbp_rate_success_from_usdgbp(cfg, monkeypatch):
    usdgbp_df = make_fx_df([0.79, 0.80, 0.81])

    def fake_download(ticker, *args, **kwargs):
        if ticker == "USDGBP=X":
            return usdgbp_df.copy()
        return pd.DataFrame()

    monkeypatch.setattr(data_loader.yf, "download", fake_download)

    loader = BTCDataLoader(cfg)
    rate = loader.load_usdgbp_rate()

    assert rate == pytest.approx(0.81)


def test_load_usdgbp_rate_fallback_to_gbpusd(cfg, monkeypatch):
    gbpusd_df = make_fx_df([1.20, 1.25, 1.30])

    def fake_download(ticker, *args, **kwargs):
        if ticker == "USDGBP=X":
            return pd.DataFrame()
        if ticker == "GBPUSD=X":
            return gbpusd_df.copy()
        return pd.DataFrame()

    monkeypatch.setattr(data_loader.yf, "download", fake_download)

    loader = BTCDataLoader(cfg)
    rate = loader.load_usdgbp_rate()

    assert rate == pytest.approx(1 / 1.30)


def test_load_usdgbp_rate_raises_when_both_downloads_fail(cfg, monkeypatch):
    def fake_download(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(data_loader.yf, "download", fake_download)

    loader = BTCDataLoader(cfg)

    with pytest.raises(ValueError, match="Could not load USD/GBP exchange rate"):
        loader.load_usdgbp_rate()


def test_load_usdgbp_rate_raises_when_gbpusd_zero(cfg, monkeypatch):
    gbpusd_df = make_fx_df([0.0])

    def fake_download(ticker, *args, **kwargs):
        if ticker == "USDGBP=X":
            return pd.DataFrame()
        if ticker == "GBPUSD=X":
            return gbpusd_df.copy()
        return pd.DataFrame()

    monkeypatch.setattr(data_loader.yf, "download", fake_download)

    loader = BTCDataLoader(cfg)

    with pytest.raises(ValueError, match="exchange rate is invalid"):
        loader.load_usdgbp_rate()