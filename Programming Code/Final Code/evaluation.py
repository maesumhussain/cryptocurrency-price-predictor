import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from config import Config, Dataset, EvaluationResults
from metrics import Metrics
from models_math import RegressionMath, NeuralNetworkMath
from tuning import HyperparameterTuner


class WalkForwardEvaluator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tuner = HyperparameterTuner(cfg)

    def evaluate(self, dataset: Dataset) -> EvaluationResults:
        X_full = dataset.X
        y_full = dataset.y
        prices_full = dataset.prices
        n_samples = X_full.shape[0]

        models_metrics = {
            "OLS": {
                "mse_return": [], "r2_return": [], "mse_price": [], "dir_acc": [],
                "y_true_return": [], "y_pred_return": [],
                "true_price": [], "pred_price": [],
            },
            "Ridge": {
                "mse_return": [], "r2_return": [], "mse_price": [], "dir_acc": [],
                "y_true_return": [], "y_pred_return": [],
                "true_price": [], "pred_price": [],
            },
            "Lasso": {
                "mse_return": [], "r2_return": [], "mse_price": [], "dir_acc": [],
                "y_true_return": [], "y_pred_return": [],
                "true_price": [], "pred_price": [],
            },
            "KernelRidge": {
                "mse_return": [], "r2_return": [], "mse_price": [], "dir_acc": [],
                "y_true_return": [], "y_pred_return": [],
                "true_price": [], "pred_price": [],
            },
            "KNN": {
                "mse_return": [], "r2_return": [], "mse_price": [], "dir_acc": [],
                "y_true_return": [], "y_pred_return": [],
                "true_price": [], "pred_price": [],
            },
            "NeuralNet": {
                "mse_return": [], "r2_return": [], "mse_price": [], "dir_acc": [],
                "y_true_return": [], "y_pred_return": [],
                "true_price": [], "pred_price": [],
            },
        }

        train_window = self.cfg.train_window_days
        test_window = self.cfg.test_window_days

        if n_samples < train_window + test_window:
            raise ValueError(
                "Not enough samples for walk-forward evaluation with the current "
                "train_window_days and test_window_days settings."
            )

        train_end = train_window
        split_number = 1

        while train_end + test_window <= n_samples:
            train_start = train_end - train_window

            X_train_raw = X_full[train_start:train_end]
            y_train = y_full[train_start:train_end]

            X_test_raw = X_full[train_end:train_end + test_window]
            y_test = y_full[train_end:train_end + test_window]
            price_test = prices_full[train_end:train_end + test_window]

            best = self.tuner.tune(X_train_raw, y_train, verbose=False)

            print(
                f"Split {split_number}: "
                f"train [{train_start}:{train_end}] "
                f"test [{train_end}:{train_end + test_window}] | "
                f"Ridge alpha={best.ridge_alpha}, "
                f"Lasso alpha={best.lasso_alpha}, "
                f"KernelRidge(alpha={best.kernel_ridge_alpha}, gamma={best.kernel_ridge_gamma}), "
                f"KNN(k={best.knn_k}, weights={best.knn_weights}, p={best.knn_p}), "
                f"NeuralNet(hidden_units={best.nn_hidden_units}, "
                f"lr={best.nn_learning_rate}, l2={best.nn_l2})"
            )

            linear_scaler = StandardScaler()
            X_train_linear = linear_scaler.fit_transform(X_train_raw)
            X_test_linear = linear_scaler.transform(X_test_raw)

            robust_scaler = RobustScaler()
            X_train_robust = robust_scaler.fit_transform(X_train_raw)
            X_test_robust = robust_scaler.transform(X_test_raw)

            true_price_test = price_test * (1.0 + y_test)

            ols_coef, ols_intercept = RegressionMath.ridge_closed(
                X_train_linear, y_train, alpha=0.0
            )
            y_pred_ols = RegressionMath.compute_predictions(
                X_test_linear, ols_coef, ols_intercept
            )
            pred_price_ols = price_test * (1.0 + y_pred_ols)

            ridge_coef, ridge_intercept = RegressionMath.ridge_closed(
                X_train_linear, y_train, alpha=best.ridge_alpha
            )
            y_pred_ridge = RegressionMath.compute_predictions(
                X_test_linear, ridge_coef, ridge_intercept
            )
            pred_price_ridge = price_test * (1.0 + y_pred_ridge)

            lasso_coef, lasso_intercept = RegressionMath.lasso_cd(
                X_train_linear,
                y_train,
                alpha=best.lasso_alpha,
                max_iter=self.cfg.lasso_max_iter,
                tol=self.cfg.lasso_tol,
            )
            y_pred_lasso = RegressionMath.compute_predictions(
                X_test_linear, lasso_coef, lasso_intercept
            )
            pred_price_lasso = price_test * (1.0 + y_pred_lasso)

            kernel_ridge_dual_coef, kernel_ridge_intercept = RegressionMath.kernel_ridge_fit(
                X_train_linear,
                y_train,
                alpha=best.kernel_ridge_alpha,
                gamma=best.kernel_ridge_gamma,
                kernel="rbf",
            )
            y_pred_kernel_ridge = RegressionMath.kernel_ridge_predict(
                X_train_linear,
                X_test_linear,
                kernel_ridge_dual_coef,
                intercept=kernel_ridge_intercept,
                gamma=best.kernel_ridge_gamma,
                kernel="rbf",
            )
            pred_price_kernel_ridge = price_test * (1.0 + y_pred_kernel_ridge)

            y_pred_knn = RegressionMath.knn_predict(
                X_train_robust,
                y_train,
                X_test_robust,
                n_neighbors=best.knn_k,
                weights=best.knn_weights,
                p=best.knn_p,
            )
            pred_price_knn = price_test * (1.0 + y_pred_knn)

            y_mean = float(np.mean(y_train))
            y_std = float(np.std(y_train))
            if y_std < 1e-12:
                y_std = 1.0

            y_train_nn = (y_train - y_mean) / y_std

            nn_w1, nn_b1, nn_w2, nn_b2 = NeuralNetworkMath.train_one_hidden_layer_regressor(
                X=X_train_robust,
                y=y_train_nn,
                hidden_units=best.nn_hidden_units,
                learning_rate=best.nn_learning_rate,
                l2=best.nn_l2,
                epochs=self.cfg.nn_epochs,
                batch_size=self.cfg.nn_batch_size,
                tol=self.cfg.nn_tol,
                patience=self.cfg.nn_patience,
                random_seed=self.cfg.random_seed,
            )

            y_pred_nn_scaled = NeuralNetworkMath.predict_one_hidden_layer_regressor(
                X_test_robust, nn_w1, nn_b1, nn_w2, nn_b2
            )
            y_pred_nn = y_pred_nn_scaled * y_std + y_mean
            pred_price_nn = price_test * (1.0 + y_pred_nn)

            for name, y_pred, p_pred in [
                ("OLS", y_pred_ols, pred_price_ols),
                ("Ridge", y_pred_ridge, pred_price_ridge),
                ("Lasso", y_pred_lasso, pred_price_lasso),
                ("KernelRidge", y_pred_kernel_ridge, pred_price_kernel_ridge),
                ("KNN", y_pred_knn, pred_price_knn),
                ("NeuralNet", y_pred_nn, pred_price_nn),
            ]:
                models_metrics[name]["mse_return"].append(
                    Metrics.mean_squared_error(y_test, y_pred)
                )
                models_metrics[name]["r2_return"].append(
                    Metrics.r2_score(y_test, y_pred)
                )
                models_metrics[name]["mse_price"].append(
                    Metrics.mean_squared_error(true_price_test, p_pred)
                )
                models_metrics[name]["dir_acc"].append(
                    Metrics.directional_accuracy(y_test, y_pred)
                )
                models_metrics[name]["y_true_return"].append(y_test)
                models_metrics[name]["y_pred_return"].append(y_pred)
                models_metrics[name]["true_price"].append(true_price_test)
                models_metrics[name]["pred_price"].append(p_pred)

            train_end += test_window
            split_number += 1

        avg_metrics_for_bar = {
            "model_names": [],
            "mse_return": [],
            "mse_price": [],
            "dir_acc": [],
        }

        for name, metrics in models_metrics.items():
            avg_metrics_for_bar["model_names"].append(name)
            avg_metrics_for_bar["mse_return"].append(float(np.mean(metrics["mse_return"])))
            avg_metrics_for_bar["mse_price"].append(float(np.mean(metrics["mse_price"])))
            avg_metrics_for_bar["dir_acc"].append(float(np.mean(metrics["dir_acc"])))

        return EvaluationResults(
            models_metrics=models_metrics,
            avg_metrics_for_bar=avg_metrics_for_bar,
        )