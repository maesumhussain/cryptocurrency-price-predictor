from sklearn.base import BaseEstimator, RegressorMixin
from models_math import RegressionMath, NeuralNetworkMath


class KNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_neighbors=5, weights="uniform", p=2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p

    def fit(self, X, y):
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        return RegressionMath.knn_predict(
            self.X_train_,
            self.y_train_,
            X,
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            p=self.p,
        )


class RidgeClosedFormRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, fit_intercept=True, regularize_intercept=False):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.regularize_intercept = regularize_intercept

    def fit(self, X, y):
        self.coef_, self.intercept_ = RegressionMath.ridge_closed(
            X,
            y,
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            regularize_intercept=self.regularize_intercept,
        )
        return self

    def predict(self, X):
        return RegressionMath.compute_predictions(X, self.coef_, self.intercept_)


class LassoCDRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.001, max_iter=5000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        self.coef_, self.intercept_ = RegressionMath.lasso_cd(
            X, y, alpha=self.alpha, max_iter=self.max_iter, tol=self.tol
        )
        return self

    def predict(self, X):
        return RegressionMath.compute_predictions(X, self.coef_, self.intercept_)


class KernelRidgeRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        alpha=1.0,
        gamma=0.1,
        kernel="rbf",
        degree=3,
        coef0=1.0,
        fit_intercept=True,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        self.X_train_ = X
        self.dual_coef_, self.intercept_ = RegressionMath.kernel_ridge_fit(
            X,
            y,
            alpha=self.alpha,
            gamma=self.gamma,
            kernel=self.kernel,
            degree=self.degree,
            coef0=self.coef0,
            fit_intercept=self.fit_intercept,
        )
        return self

    def predict(self, X):
        return RegressionMath.kernel_ridge_predict(
            self.X_train_,
            X,
            self.dual_coef_,
            intercept=self.intercept_,
            gamma=self.gamma,
            kernel=self.kernel,
            degree=self.degree,
            coef0=self.coef0,
        )


class NeuralNetworkRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        hidden_units=16,
        learning_rate=0.001,
        l2=0.0,
        epochs=300,
        batch_size=32,
        tol=1e-6,
        patience=20,
        random_seed=42,
    ):
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs
        self.batch_size = batch_size
        self.tol = tol
        self.patience = patience
        self.random_seed = random_seed

    def fit(self, X, y):
        self.w1_, self.b1_, self.w2_, self.b2_ = NeuralNetworkMath.train_one_hidden_layer_regressor(
            X=X,
            y=y,
            hidden_units=self.hidden_units,
            learning_rate=self.learning_rate,
            l2=self.l2,
            epochs=self.epochs,
            batch_size=self.batch_size,
            tol=self.tol,
            patience=self.patience,
            random_seed=self.random_seed,
        )
        return self

    def predict(self, X):
        return NeuralNetworkMath.predict_one_hidden_layer_regressor(
            X,
            self.w1_,
            self.b1_,
            self.w2_,
            self.b2_,
        )