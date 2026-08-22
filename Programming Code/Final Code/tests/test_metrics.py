import numpy as np
from metrics import Metrics


def test_mean_squared_error():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])

    mse = Metrics.mean_squared_error(y_true, y_pred)
    assert np.isclose(mse, 1 / 3)


def test_r2_score_perfect():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])

    r2 = Metrics.r2_score(y_true, y_pred)
    assert np.isclose(r2, 1.0)


def test_directional_accuracy():
    y_true = np.array([1.0, -1.0, 2.0, -3.0])
    y_pred = np.array([2.0, -4.0, -1.0, -5.0])

    acc = Metrics.directional_accuracy(y_true, y_pred)
    assert np.isclose(acc, 0.75)


def test_directional_accuracy_ignores_zero_targets():
    y_true = np.array([0.0, 1.0, -1.0])
    y_pred = np.array([5.0, 2.0, -3.0])

    acc = Metrics.directional_accuracy(y_true, y_pred)
    assert np.isclose(acc, 1.0)