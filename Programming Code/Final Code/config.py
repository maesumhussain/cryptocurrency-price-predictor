from dataclasses import dataclass
import datetime
from typing import List, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    tz_name: str = "Europe/London"

    # Default selection
    default_pair_display: str = "BTC/USD"
    min_years: int = 3
    max_years: int = 10
    default_years: int = 10

    # Walk-forward evaluation windows
    train_window_days: int = 730
    test_window_days: int = 180

    # Standard tuning
    tscv_splits: int = 5
    ridge_alpha_grid: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 300.0, 1000.0)
    lasso_alpha_grid: Tuple[float, ...] = (0.0001, 0.001, 0.01, 0.03, 0.05)

    # Kernel Ridge tuning
    kernel_ridge_alpha_grid: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    kernel_ridge_gamma_grid: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.5)

    # KNN tuning
    knn_k_grid: Tuple[int, ...] = (5, 7, 9, 11, 15)
    knn_weights_grid: Tuple[str, ...] = ("uniform", "distance")
    knn_p_grid: Tuple[int, ...] = (2,)

    # Lasso solver settings
    lasso_max_iter: int = 5000
    lasso_tol: float = 1e-4

    # Neural network tuning
    nn_hidden_units_grid: Tuple[int, ...] = (8, 16)
    nn_learning_rate_grid: Tuple[float, ...] = (0.0005, 0.001, 0.003)
    nn_l2_grid: Tuple[float, ...] = (0.0001, 0.001)

    # Neural network training settings
    nn_epochs: int = 300
    nn_batch_size: int = 32
    nn_tol: float = 1e-6
    nn_patience: int = 15
    random_seed: int = 42


@dataclass
class Dataset:
    asset_display: str
    asset_yahoo: str
    years_requested: int
    available_years: float
    btc_model: pd.DataFrame
    btc_full: pd.DataFrame
    X: np.ndarray
    y: np.ndarray
    prices: np.ndarray
    feature_columns: List[str]
    yesterday_london: datetime.date


@dataclass(frozen=True)
class BestParams:
    ridge_alpha: float
    lasso_alpha: float
    kernel_ridge_alpha: float
    kernel_ridge_gamma: float
    knn_k: int
    knn_weights: str
    knn_p: int
    nn_hidden_units: int
    nn_learning_rate: float
    nn_l2: float


@dataclass
class EvaluationResults:
    models_metrics: dict
    avg_metrics_for_bar: dict