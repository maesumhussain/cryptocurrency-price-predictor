import numpy as np
import pandas as pd
from config import Config, Dataset


class FeatureEngineer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def build_dataset(
        self,
        btc: pd.DataFrame,
        yesterday_london,
        asset_display: str,
        asset_yahoo: str,
        years_requested: int,
        available_years: float,
    ) -> Dataset:
        # Basic return & lag features
        btc["Return"] = btc["Price"].pct_change()
        btc["LagReturn1"] = btc["Return"].shift(1)
        btc["LagReturn2"] = btc["Return"].shift(2)
        btc["LagReturn3"] = btc["Return"].shift(3)

        # Moving averages
        btc["MA5"] = btc["Price"].rolling(5).mean()
        btc["MA10"] = btc["Price"].rolling(10).mean()
        btc["MA20"] = btc["Price"].rolling(20).mean()

        # Volatility
        btc["Volatility10"] = btc["Return"].rolling(10).std()
        btc["Volatility14"] = btc["Return"].rolling(14).std()
        btc["Volatility30"] = btc["Return"].rolling(30).std()

        # Relative Strength Index
        delta = btc["Price"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        window = 14
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        btc["RSI"] = 100 - (100 / (1 + rs))

        # Exponential Moving Average
        btc["EMA10"] = btc["Price"].ewm(span=10, adjust=False).mean()
        btc["EMA20"] = btc["Price"].ewm(span=20, adjust=False).mean()
        btc["EMA50"] = btc["Price"].ewm(span=50, adjust=False).mean()

        # Moving Average Convergence Divergence
        ema12 = btc["Price"].ewm(span=12, adjust=False).mean()
        ema26 = btc["Price"].ewm(span=26, adjust=False).mean()
        btc["MACD"] = ema12 - ema26
        btc["MACD_signal"] = btc["MACD"].ewm(span=9, adjust=False).mean()

        # Momentum
        btc["Mom7"] = btc["Price"].pct_change(7)
        btc["Mom30"] = btc["Price"].pct_change(30)

        # Bollinger width
        btc["BB_STD20"] = btc["Price"].rolling(20).std()
        btc["BBWidth20"] = (4.0 * btc["BB_STD20"]) / btc["MA20"]

        # Average True Range (14)
        high = btc["High"]
        low = btc["Low"]
        close = btc["Close"]
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        btc["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        btc["ATR14"] = btc["TR"].rolling(14).mean()

        # Non-linear transforms
        btc["ReturnSq"] = btc["Return"] ** 2
        btc["LagReturnSq1"] = btc["ReturnSq"].shift(1)
        btc["RetVolInteraction"] = btc["Return"] * btc["Volatility10"]

        btc["Trend"] = btc["EMA10"] - btc["EMA50"]
        btc["RSI_Change"] = btc["RSI"].diff()
        btc["VolatilityChange"] = btc["Volatility10"].diff()

        # Keep a copy for latest-row prediction BEFORE dropna/target shift
        btc_full = btc.copy()

        # Target: next-day return
        btc["TargetReturn"] = btc["Return"].shift(-1)

        # Drop rows with NaNs
        btc_model = btc.dropna().copy()

        feature_columns = [
            "LagReturn1", "LagReturn2", "LagReturn3",
            "MA5", "MA10", "MA20",
            "Volatility10", "Volatility14", "Volatility30", "RSI",
            "EMA10", "EMA20", "EMA50",
            "MACD", "MACD_signal",
            "Mom7", "Mom30",
            "BBWidth20", "ATR14",
            "ReturnSq", "LagReturnSq1", "RetVolInteraction",
            "Trend", "RSI_Change", "VolatilityChange",
        ]

        X = btc_model[feature_columns].to_numpy(dtype=float)
        y = btc_model["TargetReturn"].to_numpy(dtype=float)
        prices = btc_model["Price"].to_numpy(dtype=float)

        return Dataset(
            asset_display=asset_display,
            asset_yahoo=asset_yahoo,
            years_requested=years_requested,
            available_years=available_years,
            btc_model=btc_model,
            btc_full=btc_full,
            X=X,
            y=y,
            prices=prices,
            feature_columns=feature_columns,
            yesterday_london=yesterday_london,
        )