import numpy as np


def test_build_dataset_returns_expected_fields(engineered_dataset):
    dataset = engineered_dataset

    assert dataset.asset_display == "BTC/USD"
    assert dataset.asset_yahoo == "BTC-USD"
    assert dataset.years_requested == 3
    assert dataset.available_years == 3.0
    assert dataset.yesterday_london.year == 2025


def test_build_dataset_shapes(engineered_dataset):
    dataset = engineered_dataset

    assert dataset.X.shape[0] > 0
    assert dataset.X.shape[1] == len(dataset.feature_columns)
    assert len(dataset.y) == dataset.X.shape[0]
    assert len(dataset.prices) == dataset.X.shape[0]


def test_build_dataset_has_no_nan(engineered_dataset):
    dataset = engineered_dataset

    assert np.isfinite(dataset.X).all()
    assert np.isfinite(dataset.y).all()
    assert np.isfinite(dataset.prices).all()


def test_feature_columns_match_expected_count(engineered_dataset):
    dataset = engineered_dataset
    assert len(dataset.feature_columns) == 25


def test_btc_model_and_btc_full_exist(engineered_dataset):
    dataset = engineered_dataset

    assert dataset.btc_model is not None
    assert dataset.btc_full is not None
    assert not dataset.btc_model.empty
    assert not dataset.btc_full.empty