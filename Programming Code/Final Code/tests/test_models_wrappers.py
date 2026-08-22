import numpy as np
from models_wrappers import RidgeClosedFormRegressor, LassoCDRegressor, KernelRidgeRegressor


def test_ridge_wrapper_fit_predict():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = 3.0 * X[:, 0] + 2.0

    model = RidgeClosedFormRegressor(alpha=0.0)
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == y.shape
    assert np.allclose(preds, y, atol=1e-6)


def test_lasso_wrapper_fit_predict():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 2))
    y = 1.5 * X[:, 0] - 0.8 * X[:, 1]

    model = LassoCDRegressor(alpha=0.001, max_iter=5000, tol=1e-6)
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == y.shape
    assert np.isfinite(preds).all()


def test_kernel_ridge_wrapper_fit_predict():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 2))
    y = np.sin(X[:, 0]) + 0.3 * X[:, 1]

    model = KernelRidgeRegressor(alpha=0.1, gamma=0.5, kernel="rbf")
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == y.shape
    assert np.isfinite(preds).all()