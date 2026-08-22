import datetime
from types import SimpleNamespace

import pytest

import console_app
from data_loader import InsufficientDataError


def test_console_app_invalid_pair_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported pair"):
        console_app.main(pair_display="FAKE/USD", years_requested=3, make_plots=False)


def test_console_app_handles_insufficient_data(monkeypatch, capsys):
    class FakeLoader:
        def __init__(self, cfg):
            self.cfg = cfg

        def load(self, ticker, years_requested):
            raise InsufficientDataError(
                available_years_floor=2,
                available_years_exact=2.75,
            )

    monkeypatch.setattr(console_app, "BTCDataLoader", FakeLoader)

    result = console_app.main(
        pair_display="BTC/USD",
        years_requested=3,
        make_plots=False,
    )

    captured = capsys.readouterr()

    assert result is None
    assert "Only 2.75 years of data are available for BTC/USD." in captured.out
    assert "Try again with 2 years or fewer." in captured.out


def test_console_app_happy_path_without_plots(monkeypatch, engineered_dataset, capsys):
    calls = {
        "loader": False,
        "features": False,
        "evaluate": False,
        "plot": False,
        "predict": False,
    }

    class FakeLoader:
        def __init__(self, cfg):
            self.cfg = cfg

        def load(self, ticker, years_requested):
            calls["loader"] = True
            return engineered_dataset.btc_full.copy(), datetime.date(2025, 12, 31), 3.5

    class FakeFeatureEngineer:
        def __init__(self, cfg):
            self.cfg = cfg

        def build_dataset(
            self,
            btc,
            yesterday_london,
            asset_display,
            asset_yahoo,
            years_requested,
            available_years,
        ):
            calls["features"] = True
            assert asset_display == "BTC/USD"
            assert asset_yahoo == "BTC-USD"
            assert years_requested == 3
            assert available_years == 3.5
            return engineered_dataset

    class FakeEvaluator:
        def __init__(self, cfg):
            self.cfg = cfg

        def evaluate(self, dataset):
            calls["evaluate"] = True
            assert dataset.asset_display == "BTC/USD"
            return {"fake": "results"}

    class FakePlotter:
        def __init__(self):
            pass

        def plot(self, results, asset_display):
            calls["plot"] = True

    class FakePredictor:
        def __init__(self, cfg):
            self.cfg = cfg

        def predict(self, dataset):
            calls["predict"] = True
            assert dataset.asset_display == "BTC/USD"
            return {"fake": "prediction"}

    monkeypatch.setattr(console_app, "BTCDataLoader", FakeLoader)
    monkeypatch.setattr(console_app, "FeatureEngineer", FakeFeatureEngineer)
    monkeypatch.setattr(console_app, "WalkForwardEvaluator", FakeEvaluator)
    monkeypatch.setattr(console_app, "Plotter", FakePlotter)
    monkeypatch.setattr(console_app, "TomorrowPredictor", FakePredictor)

    console_app.main(
        pair_display="BTC/USD",
        years_requested=3,
        make_plots=False,
    )

    captured = capsys.readouterr()

    assert calls["loader"] is True
    assert calls["features"] is True
    assert calls["evaluate"] is True
    assert calls["plot"] is False
    assert calls["predict"] is True

    assert "Console Crypto Forecasting Application" in captured.out
    assert "Asset: BTC/USD" in captured.out
    assert "Yahoo ticker: BTC-USD" in captured.out


def test_console_app_happy_path_with_plots(monkeypatch, engineered_dataset):
    calls = {
        "plot": False,
    }

    class FakeLoader:
        def __init__(self, cfg):
            self.cfg = cfg

        def load(self, ticker, years_requested):
            return engineered_dataset.btc_full.copy(), datetime.date(2025, 12, 31), 3.5

    class FakeFeatureEngineer:
        def __init__(self, cfg):
            self.cfg = cfg

        def build_dataset(
            self,
            btc,
            yesterday_london,
            asset_display,
            asset_yahoo,
            years_requested,
            available_years,
        ):
            return engineered_dataset

    class FakeEvaluator:
        def __init__(self, cfg):
            self.cfg = cfg

        def evaluate(self, dataset):
            return {"fake": "results"}

    class FakePlotter:
        def __init__(self):
            pass

        def plot(self, results, asset_display):
            calls["plot"] = True
            assert asset_display == "BTC/USD"
            assert results == {"fake": "results"}

    class FakePredictor:
        def __init__(self, cfg):
            self.cfg = cfg

        def predict(self, dataset):
            return {"fake": "prediction"}

    monkeypatch.setattr(console_app, "BTCDataLoader", FakeLoader)
    monkeypatch.setattr(console_app, "FeatureEngineer", FakeFeatureEngineer)
    monkeypatch.setattr(console_app, "WalkForwardEvaluator", FakeEvaluator)
    monkeypatch.setattr(console_app, "Plotter", FakePlotter)
    monkeypatch.setattr(console_app, "TomorrowPredictor", FakePredictor)

    console_app.main(
        pair_display="BTC/USD",
        years_requested=3,
        make_plots=True,
    )

    assert calls["plot"] is True