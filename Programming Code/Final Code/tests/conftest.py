import datetime
import numpy as np
import pandas as pd
import pytest

from config import Config
from features import FeatureEngineer


@pytest.fixture
def cfg():
    return Config()


@pytest.fixture
def sample_price_df():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=1000, freq="D")

    trend = np.linspace(20000, 30000, len(dates))
    noise = rng.normal(0, 200, len(dates))
    close = trend + noise

    df = pd.DataFrame(index=dates)
    df["Open"] = close + rng.normal(0, 50, len(dates))
    df["High"] = close + np.abs(rng.normal(100, 20, len(dates)))
    df["Low"] = close - np.abs(rng.normal(100, 20, len(dates)))
    df["Close"] = close
    df["Price"] = close
    return df


@pytest.fixture
def engineered_dataset(cfg, sample_price_df):
    fe = FeatureEngineer(cfg)

    return fe.build_dataset(
        btc=sample_price_df.copy(),
        yesterday_london=datetime.date(2025, 12, 31),
        asset_display="BTC/USD",
        asset_yahoo="BTC-USD",
        years_requested=3,
        available_years=3.0,
    )