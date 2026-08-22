from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from config import Config, BestParams
from models_wrappers import (
    RidgeClosedFormRegressor,
    LassoCDRegressor,
    KernelRidgeRegressor,
    NeuralNetworkRegressor,
    KNNRegressor,
)


class HyperparameterTuner:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def tune(self, X, y, verbose: bool = True) -> BestParams:
        if verbose:
            print("Standard hyperparameter tuning (GridSearchCV + TimeSeriesSplit):")
            print("---------------------------------------------------------------")

        tscv = TimeSeriesSplit(n_splits=self.cfg.tscv_splits)

        ridge_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeClosedFormRegressor()),
        ])

        ridge_gs = GridSearchCV(
            ridge_pipe,
            {"model__alpha": list(self.cfg.ridge_alpha_grid)},
            cv=tscv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        ridge_gs.fit(X, y)
        best_ridge_alpha = float(ridge_gs.best_params_["model__alpha"])

        lasso_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LassoCDRegressor(
                max_iter=self.cfg.lasso_max_iter,
                tol=self.cfg.lasso_tol,
            )),
        ])

        lasso_gs = GridSearchCV(
            lasso_pipe,
            {"model__alpha": list(self.cfg.lasso_alpha_grid)},
            cv=tscv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        lasso_gs.fit(X, y)
        best_lasso_alpha = float(lasso_gs.best_params_["model__alpha"])

        kernel_ridge_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", KernelRidgeRegressor(kernel="rbf")),
        ])

        kernel_ridge_gs = GridSearchCV(
            kernel_ridge_pipe,
            {
                "model__alpha": list(self.cfg.kernel_ridge_alpha_grid),
                "model__gamma": list(self.cfg.kernel_ridge_gamma_grid),
            },
            cv=tscv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        kernel_ridge_gs.fit(X, y)
        best_kernel_ridge_alpha = float(kernel_ridge_gs.best_params_["model__alpha"])
        best_kernel_ridge_gamma = float(kernel_ridge_gs.best_params_["model__gamma"])

        knn_pipe = Pipeline([
            ("scaler", RobustScaler()),
            ("model", KNNRegressor()),
        ])

        knn_gs = GridSearchCV(
            knn_pipe,
            {
                "model__n_neighbors": list(self.cfg.knn_k_grid),
                "model__weights": list(self.cfg.knn_weights_grid),
                "model__p": list(self.cfg.knn_p_grid),
            },
            cv=tscv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        knn_gs.fit(X, y)

        best_knn_k = int(knn_gs.best_params_["model__n_neighbors"])
        best_knn_weights = str(knn_gs.best_params_["model__weights"])
        best_knn_p = int(knn_gs.best_params_["model__p"])

        nn_pipe = Pipeline([
            ("scaler", RobustScaler()),
            ("model", NeuralNetworkRegressor(
                epochs=self.cfg.nn_epochs,
                batch_size=self.cfg.nn_batch_size,
                tol=self.cfg.nn_tol,
                patience=self.cfg.nn_patience,
                random_seed=self.cfg.random_seed,
            )),
        ])

        nn_gs = GridSearchCV(
            nn_pipe,
            {
                "model__hidden_units": list(self.cfg.nn_hidden_units_grid),
                "model__learning_rate": list(self.cfg.nn_learning_rate_grid),
                "model__l2": list(self.cfg.nn_l2_grid),
            },
            cv=tscv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        nn_gs.fit(X, y)

        best_nn_hidden_units = int(nn_gs.best_params_["model__hidden_units"])
        best_nn_learning_rate = float(nn_gs.best_params_["model__learning_rate"])
        best_nn_l2 = float(nn_gs.best_params_["model__l2"])

        if verbose:
            print(f"Ridge best alpha: {best_ridge_alpha}")
            print(f"Lasso best alpha: {best_lasso_alpha}")
            print(
                f"KernelRidge best params: alpha={best_kernel_ridge_alpha}, "
                f"gamma={best_kernel_ridge_gamma}"
            )
            print(f"KNN best params: k={best_knn_k}, weights={best_knn_weights}, p={best_knn_p}")
            print(
                f"NeuralNet best params: hidden_units={best_nn_hidden_units}, "
                f"learning_rate={best_nn_learning_rate}, l2={best_nn_l2}"
            )
            print("---------------------------------------------------------------\n")

        return BestParams(
            ridge_alpha=best_ridge_alpha,
            lasso_alpha=best_lasso_alpha,
            kernel_ridge_alpha=best_kernel_ridge_alpha,
            kernel_ridge_gamma=best_kernel_ridge_gamma,
            knn_k=best_knn_k,
            knn_weights=best_knn_weights,
            knn_p=best_knn_p,
            nn_hidden_units=best_nn_hidden_units,
            nn_learning_rate=best_nn_learning_rate,
            nn_l2=best_nn_l2,
        )