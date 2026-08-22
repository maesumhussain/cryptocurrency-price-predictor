import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

from config import Config, Dataset
from models_math import RegressionMath, NeuralNetworkMath
from tuning import HyperparameterTuner


class TomorrowPredictor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tuner = HyperparameterTuner(cfg)

    def predict(self, dataset: Dataset) -> dict:
        train_window = self.cfg.train_window_days

        X_recent = dataset.X[-train_window:]
        y_recent = dataset.y[-train_window:]

        print("Retuning hyperparameters on the most recent training window for tomorrow prediction...")
        best = self.tuner.tune(X_recent, y_recent, verbose=True)

        linear_scaler = StandardScaler()
        X_recent_linear = linear_scaler.fit_transform(X_recent)

        robust_scaler = RobustScaler()
        X_recent_robust = robust_scaler.fit_transform(X_recent)

        ols_coef, ols_intercept = RegressionMath.ridge_closed(
            X_recent_linear, y_recent, alpha=0.0
        )

        ridge_coef, ridge_intercept = RegressionMath.ridge_closed(
            X_recent_linear, y_recent, alpha=best.ridge_alpha
        )

        lasso_coef, lasso_intercept = RegressionMath.lasso_cd(
            X_recent_linear,
            y_recent,
            alpha=best.lasso_alpha,
            max_iter=self.cfg.lasso_max_iter,
            tol=self.cfg.lasso_tol,
        )

        kernel_ridge_dual_coef, kernel_ridge_intercept = RegressionMath.kernel_ridge_fit(
            X_recent_linear,
            y_recent,
            alpha=best.kernel_ridge_alpha,
            gamma=best.kernel_ridge_gamma,
            kernel="rbf",
        )

        y_mean = float(np.mean(y_recent))
        y_std = float(np.std(y_recent))
        if y_std < 1e-12:
            y_std = 1.0

        y_recent_nn = (y_recent - y_mean) / y_std

        nn_w1, nn_b1, nn_w2, nn_b2 = NeuralNetworkMath.train_one_hidden_layer_regressor(
            X=X_recent_robust,
            y=y_recent_nn,
            hidden_units=best.nn_hidden_units,
            learning_rate=best.nn_learning_rate,
            l2=best.nn_l2,
            epochs=self.cfg.nn_epochs,
            batch_size=self.cfg.nn_batch_size,
            tol=self.cfg.nn_tol,
            patience=self.cfg.nn_patience,
            random_seed=self.cfg.random_seed,
        )

        latest_row = dataset.btc_model.iloc[-1]

        latest_features = np.array(
            [latest_row[col] for col in dataset.feature_columns],
            dtype=float,
        ).reshape(1, -1)

        latest_linear = linear_scaler.transform(latest_features)
        latest_robust = robust_scaler.transform(latest_features)

        pred_ret_ols = float(
            RegressionMath.compute_predictions(latest_linear, ols_coef, ols_intercept).item()
        )
        pred_ret_ridge = float(
            RegressionMath.compute_predictions(latest_linear, ridge_coef, ridge_intercept).item()
        )
        pred_ret_lasso = float(
            RegressionMath.compute_predictions(latest_linear, lasso_coef, lasso_intercept).item()
        )
        pred_ret_kernel_ridge = float(
            RegressionMath.kernel_ridge_predict(
                X_recent_linear,
                latest_linear,
                kernel_ridge_dual_coef,
                intercept=kernel_ridge_intercept,
                gamma=best.kernel_ridge_gamma,
                kernel="rbf",
            ).item()
        )
        pred_ret_knn = float(
            RegressionMath.knn_predict(
                X_recent_robust,
                y_recent,
                latest_robust,
                n_neighbors=best.knn_k,
                weights=best.knn_weights,
                p=best.knn_p,
            ).item()
        )

        pred_ret_nn_scaled = float(
            NeuralNetworkMath.predict_one_hidden_layer_regressor(
                latest_robust, nn_w1, nn_b1, nn_w2, nn_b2
            ).item()
        )
        pred_ret_nn = pred_ret_nn_scaled * y_std + y_mean

        latest_price = float(latest_row["Price"])
        latest_index = dataset.btc_model.index[-1]
        tomorrow_date = latest_index + datetime.timedelta(days=1)

        predictions = {
            "OLS": {
                "return_decimal": pred_ret_ols,
                "price_usd": latest_price * (1.0 + pred_ret_ols),
                "return_display": f"{pred_ret_ols * 100:.3f}%",
                "params": "No tuning required",
            },
            "Ridge": {
                "return_decimal": pred_ret_ridge,
                "price_usd": latest_price * (1.0 + pred_ret_ridge),
                "return_display": f"{pred_ret_ridge * 100:.3f}%",
                "params": f"alpha={best.ridge_alpha}",
            },
            "Lasso": {
                "return_decimal": pred_ret_lasso,
                "price_usd": latest_price * (1.0 + pred_ret_lasso),
                "return_display": f"{pred_ret_lasso * 100:.3f}%",
                "params": f"alpha={best.lasso_alpha}",
            },
            "KernelRidge": {
                "return_decimal": pred_ret_kernel_ridge,
                "price_usd": latest_price * (1.0 + pred_ret_kernel_ridge),
                "return_display": f"{pred_ret_kernel_ridge * 100:.3f}%",
                "params": (
                    f"alpha={best.kernel_ridge_alpha}, "
                    f"gamma={best.kernel_ridge_gamma}, kernel=rbf"
                ),
            },
            "KNN": {
                "return_decimal": pred_ret_knn,
                "price_usd": latest_price * (1.0 + pred_ret_knn),
                "return_display": f"{pred_ret_knn * 100:.3f}%",
                "params": f"k={best.knn_k}, weights={best.knn_weights}, p={best.knn_p}",
            },
            "NeuralNet": {
                "return_decimal": pred_ret_nn,
                "price_usd": latest_price * (1.0 + pred_ret_nn),
                "return_display": f"{pred_ret_nn * 100:.3f}%",
                "params": (
                    f"hidden_units={best.nn_hidden_units}, "
                    f"learning_rate={best.nn_learning_rate}, "
                    f"l2={best.nn_l2}"
                ),
            },
        }

        print("---------------------------------------------")
        print(f" Latest Close Price and Tomorrow's Prediction ({dataset.asset_display})")
        print("---------------------------------------------")
        print(f"Latest Close Price: {latest_price:.10f}\n")

        print(f"Predicted Return and Price for {tomorrow_date.date()} ({dataset.asset_display})\n")

        for name, values in predictions.items():
            print(f"{name}:")
            print(f"Predicted Return: {values['return_display']}")
            print(f"Predicted Price: {values['price_usd']:.10f}")
            print(f"Params: {values['params']}\n")

        return {
            "asset_display": dataset.asset_display,
            "tomorrow_date": str(tomorrow_date.date()),
            "latest_price_usd": latest_price,
            "models": predictions,
        }