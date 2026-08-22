import numpy as np
from models_math import RegressionMath, NeuralNetworkMath


def test_ridge_closed_simple_linear_case():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = 2.0 * X[:, 0] + 1.0

    coef, intercept = RegressionMath.ridge_closed(X, y, alpha=0.0)
    preds = RegressionMath.compute_predictions(X, coef, intercept)

    assert np.allclose(preds, y, atol=1e-6)


def test_compute_predictions():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    coef = np.array([2.0, 1.0])
    intercept = 0.5

    preds = RegressionMath.compute_predictions(X, coef, intercept)
    expected = np.array([4.5, 10.5])

    assert np.allclose(preds, expected)


def test_soft_threshold():
    assert RegressionMath.soft_threshold(5.0, 2.0) == 3.0
    assert RegressionMath.soft_threshold(-5.0, 2.0) == -3.0
    assert RegressionMath.soft_threshold(1.0, 2.0) == 0.0


def test_lasso_cd_output_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2]

    coef, intercept = RegressionMath.lasso_cd(X, y, alpha=0.001, max_iter=5000, tol=1e-6)
    preds = RegressionMath.compute_predictions(X, coef, intercept)

    assert coef.shape == (3,)
    assert isinstance(intercept, float)
    assert preds.shape == y.shape


def test_rbf_kernel_matrix_is_symmetric_on_same_input():
    X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    K = RegressionMath.compute_kernel_matrix(X, X, kernel="rbf", gamma=0.5)

    assert K.shape == (3, 3)
    assert np.allclose(K, K.T)
    assert np.allclose(np.diag(K), np.ones(3))


def test_kernel_ridge_fit_predict_shapes_and_finiteness():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1]

    dual_coef, intercept = RegressionMath.kernel_ridge_fit(
        X,
        y,
        alpha=0.1,
        gamma=0.5,
        kernel="rbf",
    )
    preds = RegressionMath.kernel_ridge_predict(
        X,
        X,
        dual_coef,
        intercept=intercept,
        gamma=0.5,
        kernel="rbf",
    )

    assert dual_coef.shape == (40,)
    assert isinstance(intercept, float)
    assert preds.shape == y.shape
    assert np.isfinite(preds).all()


def test_relu_and_derivative():
    z = np.array([-2.0, 0.0, 3.0])

    assert np.array_equal(NeuralNetworkMath.relu(z), np.array([0.0, 0.0, 3.0]))
    assert np.array_equal(NeuralNetworkMath.relu_derivative(z), np.array([0.0, 0.0, 1.0]))