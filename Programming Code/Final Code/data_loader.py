import datetime
import math
from typing import Tuple

import pandas as pd
import yfinance as yf
from zoneinfo import ZoneInfo

from config import Config


class InsufficientDataError(Exception):
    def __init__(self, available_years_floor: int, available_years_exact: float):
        self.available_years_floor = available_years_floor
        self.available_years_exact = available_years_exact
        super().__init__(
            f"Only {available_years_exact:.2f} years of data are available."
        )


class BTCDataLoader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.london = ZoneInfo(cfg.tz_name)

    @staticmethod
    def _download_single_ticker(
        ticker: str,
        **kwargs,
    ) -> pd.DataFrame:
        df = yf.download(
            ticker,
            progress=False,
            ignore_tz=True,
            auto_adjust=True,
            **kwargs,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.dropna(how="all").copy()

        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df.columns.get_level_values(-1):
                df = df.xs(ticker, axis=1, level=-1)
            elif ticker in df.columns.get_level_values(0):
                df = df.xs(ticker, axis=1, level=0)
            else:
                df.columns = [
                    col[0] if isinstance(col, tuple) else col
                    for col in df.columns
                ]

        df.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in df.columns
        ]

        return df

    @staticmethod
    def _get_last_scalar(series_or_value) -> float:
        if isinstance(series_or_value, pd.DataFrame):
            if series_or_value.empty:
                raise ValueError("Expected a scalar value but received an empty DataFrame.")
            return float(series_or_value.iloc[-1, -1])

        if isinstance(series_or_value, pd.Series):
            if series_or_value.empty:
                raise ValueError("Expected a scalar value but received an empty Series.")
            return float(series_or_value.iloc[-1])

        return float(series_or_value)

    def load(self, ticker: str, years_requested: int) -> Tuple[pd.DataFrame, datetime.date, float]:
        now_london = datetime.datetime.now(self.london)
        yesterday_london = (now_london - datetime.timedelta(days=1)).date()
        end_date = yesterday_london + datetime.timedelta(days=1)

        btc_full = self._download_single_ticker(
            ticker,
            start="2010-01-01",
            end=str(end_date),
            interval="1d",
        )

        btc_full = btc_full.dropna().copy()

        if btc_full.empty:
            raise ValueError(f"No data was found for {ticker}.")

        required_columns = ["Open", "High", "Low", "Close"]
        missing = [col for col in required_columns if col not in btc_full.columns]
        if missing:
            raise ValueError(
                f"Downloaded data for {ticker} is missing required columns: {missing}"
            )

        btc_full["Price"] = btc_full["Close"].astype(float)

        first_date = pd.Timestamp(btc_full.index.min()).date()
        days_available = (yesterday_london - first_date).days
        available_years_exact = max(days_available / 365.25, 0.0)
        available_years_floor = max(1, math.floor(available_years_exact))

        if years_requested > available_years_floor:
            raise InsufficientDataError(
                available_years_floor=available_years_floor,
                available_years_exact=available_years_exact,
            )

        start_cutoff = pd.Timestamp(
            yesterday_london - datetime.timedelta(days=int(years_requested * 365.25))
        )
        btc = btc_full.loc[btc_full.index >= start_cutoff].copy()

        if btc.empty:
            raise ValueError(f"Unable to prepare a {years_requested}-year dataset for {ticker}.")

        return btc, yesterday_london, available_years_exact

    def load_usdgbp_rate(self) -> float:
        fx = self._download_single_ticker(
            "USDGBP=X",
            period="10d",
            interval="1d",
        )

        fx = fx.dropna().copy()

        if not fx.empty and "Close" in fx.columns:
            return self._get_last_scalar(fx["Close"])

        gbpusd = self._download_single_ticker(
            "GBPUSD=X",
            period="10d",
            interval="1d",
        )

        gbpusd = gbpusd.dropna().copy()

        if gbpusd.empty or "Close" not in gbpusd.columns:
            raise ValueError("Could not load USD/GBP exchange rate from Yahoo Finance.")

        latest_gbpusd = self._get_last_scalar(gbpusd["Close"])
        if abs(latest_gbpusd) < 1e-12:
            raise ValueError("Loaded GBP/USD exchange rate is invalid.")

        return 1.0 / latest_gbpusd