import numpy as np
from typing import Tuple


class RegressionMath:
    @staticmethod
    def ridge_closed(
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        regularize_intercept: bool = False,
    ) -> Tuple[np.ndarray, float]:
        if fit_intercept:
            X_ext = np.c_[np.ones(len(X)), X]
        else:
            X_ext = X

        A = X_ext.T @ X_ext
        b = X_ext.T @ y

        I = np.eye(A.shape[0])
        if fit_intercept and not regularize_intercept:
            I[0, 0] = 0.0

        w = np.linalg.solve(A + alpha * I, b)

        if fit_intercept:
            return w[1:], float(w[0])
        return w, 0.0

    @staticmethod
    def compute_predictions(X: np.ndarray, coef: np.ndarray, intercept: float = 0.0) -> np.ndarray:
        return X @ coef + intercept

    @staticmethod
    def soft_threshold(rho: float, lam: float) -> float:
        if rho < -lam:
            return rho + lam
        if rho > lam:
            return rho - lam
        return 0.0

    @staticmethod
    def lasso_cd(
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.001,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ) -> Tuple[np.ndarray, float]:
        _, n_features = X.shape

        y_mean = y.mean()
        y_centered = y - y_mean

        w = np.zeros(n_features)

        for _ in range(max_iter):
            w_old = w.copy()

            for j in range(n_features):
                residual = y_centered - (X @ w) + X[:, j] * w[j]
                rho_j = float(np.dot(X[:, j], residual))
                z_j = float(np.dot(X[:, j], X[:, j]))
                if z_j == 0.0:
                    continue
                w[j] = RegressionMath.soft_threshold(rho_j, alpha) / z_j

            if np.max(np.abs(w - w_old)) < tol:
                break

        intercept = float(y_mean)
        return w, intercept

    @staticmethod
    def compute_kernel_matrix(
        X1: np.ndarray,
        X2: np.ndarray,
        kernel: str = "rbf",
        gamma: float = 0.1,
        degree: int = 3,
        coef0: float = 1.0,
    ) -> np.ndarray:
        if kernel == "linear":
            return X1 @ X2.T

        if kernel == "poly":
            return (gamma * (X1 @ X2.T) + coef0) ** degree

        if kernel == "rbf":
            x1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            x2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
            sq_dist = np.maximum(x1_sq + x2_sq - 2.0 * (X1 @ X2.T), 0.0)
            return np.exp(-gamma * sq_dist)

        raise ValueError(f"Unsupported kernel: {kernel}")

    @staticmethod
    def kernel_ridge_fit(
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 1.0,
        gamma: float = 0.1,
        kernel: str = "rbf",
        degree: int = 3,
        coef0: float = 1.0,
        fit_intercept: bool = True,
    ) -> Tuple[np.ndarray, float]:
        y = np.asarray(y, dtype=float)
        intercept = float(np.mean(y)) if fit_intercept else 0.0
        y_centered = y - intercept

        K = RegressionMath.compute_kernel_matrix(
            X,
            X,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
        )

        n_samples = K.shape[0]
        dual_coef = np.linalg.solve(K + alpha * np.eye(n_samples), y_centered)
        return dual_coef, intercept

    @staticmethod
    def kernel_ridge_predict(
        X_train: np.ndarray,
        X_test: np.ndarray,
        dual_coef: np.ndarray,
        intercept: float = 0.0,
        gamma: float = 0.1,
        kernel: str = "rbf",
        degree: int = 3,
        coef0: float = 1.0,
    ) -> np.ndarray:
        K_test = RegressionMath.compute_kernel_matrix(
            X_test,
            X_train,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
        )
        return K_test @ dual_coef + intercept

    @staticmethod
    def minkowski_distance_matrix(
        X1: np.ndarray,
        X2: np.ndarray,
        p: int = 2,
    ) -> np.ndarray:
        if p <= 0:
            raise ValueError("p must be a positive integer.")

        X1 = np.asarray(X1, dtype=float)
        X2 = np.asarray(X2, dtype=float)

        diff = np.abs(X1[:, None, :] - X2[None, :, :])
        return np.sum(diff ** p, axis=2) ** (1.0 / p)

    @staticmethod
    def knn_predict(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        n_neighbors: int = 5,
        weights: str = "uniform",
        p: int = 2,
    ) -> np.ndarray:
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        X_test = np.asarray(X_test, dtype=float)

        if X_train.ndim != 2 or X_test.ndim != 2:
            raise ValueError("X_train and X_test must be 2D arrays.")
        if y_train.ndim != 1:
            raise ValueError("y_train must be a 1D array.")
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same number of samples.")
        if len(X_train) == 0:
            raise ValueError("X_train must not be empty.")
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive.")
        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be either 'uniform' or 'distance'.")

        k = min(int(n_neighbors), len(X_train))
        distances = RegressionMath.minkowski_distance_matrix(X_test, X_train, p=p)
        preds = np.empty(X_test.shape[0], dtype=float)

        for i in range(X_test.shape[0]):
            row = distances[i]
            neighbour_idx = np.argpartition(row, k - 1)[:k]
            neighbour_dist = row[neighbour_idx]
            neighbour_y = y_train[neighbour_idx]

            order = np.argsort(neighbour_dist)
            neighbour_dist = neighbour_dist[order]
            neighbour_y = neighbour_y[order]

            if weights == "uniform":
                preds[i] = float(np.mean(neighbour_y))
            else:
                zero_mask = neighbour_dist <= 1e-12
                if np.any(zero_mask):
                    preds[i] = float(np.mean(neighbour_y[zero_mask]))
                else:
                    inv_dist = 1.0 / neighbour_dist
                    preds[i] = float(np.dot(inv_dist, neighbour_y) / np.sum(inv_dist))

        return preds


class NeuralNetworkMath:
    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)

    @staticmethod
    def initialise_parameters(
        n_inputs: int,
        n_hidden: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        w1 = rng.normal(0.0, np.sqrt(2.0 / n_inputs), size=(n_inputs, n_hidden))
        b1 = np.zeros((1, n_hidden))
        w2 = rng.normal(0.0, np.sqrt(2.0 / n_hidden), size=(n_hidden, 1))
        b2 = np.zeros((1, 1))
        return w1, b1, w2, b2

    @staticmethod
    def forward(
        X: np.ndarray,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z1 = X @ w1 + b1
        a1 = NeuralNetworkMath.relu(z1)
        y_hat = a1 @ w2 + b2
        return z1, a1, y_hat

    @staticmethod
    def mse_loss(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        w1: np.ndarray,
        w2: np.ndarray,
        l2: float,
    ) -> float:
        y_true = y_true.reshape(-1, 1)
        mse = np.mean((y_pred - y_true) ** 2)
        reg = l2 * (np.sum(w1 ** 2) + np.sum(w2 ** 2))
        return float(mse + reg)

    @staticmethod
    def train_one_hidden_layer_regressor(
        X: np.ndarray,
        y: np.ndarray,
        hidden_units: int = 16,
        learning_rate: float = 0.001,
        l2: float = 0.0,
        epochs: int = 300,
        batch_size: int = 32,
        tol: float = 1e-6,
        patience: int = 20,
        random_seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_samples, n_inputs = X.shape
        rng = np.random.default_rng(random_seed)

        w1, b1, w2, b2 = NeuralNetworkMath.initialise_parameters(
            n_inputs=n_inputs,
            n_hidden=hidden_units,
            rng=rng,
        )

        y_col = y.reshape(-1, 1)

        best_loss = float("inf")
        best_params = (w1.copy(), b1.copy(), w2.copy(), b2.copy())
        no_improve_count = 0

        for _ in range(epochs):
            indices = rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_col[indices]

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                z1, a1, y_hat = NeuralNetworkMath.forward(X_batch, w1, b1, w2, b2)

                if not np.all(np.isfinite(y_hat)):
                    break

                batch_n = len(X_batch)
                error = y_hat - y_batch
                d_y_hat = (2.0 / batch_n) * error

                d_w2 = a1.T @ d_y_hat + 2.0 * l2 * w2
                d_b2 = np.sum(d_y_hat, axis=0, keepdims=True)

                d_a1 = d_y_hat @ w2.T
                d_z1 = d_a1 * NeuralNetworkMath.relu_derivative(z1)

                d_w1 = X_batch.T @ d_z1 + 2.0 * l2 * w1
                d_b1 = np.sum(d_z1, axis=0, keepdims=True)

                np.clip(d_w1, -1.0, 1.0, out=d_w1)
                np.clip(d_b1, -1.0, 1.0, out=d_b1)
                np.clip(d_w2, -1.0, 1.0, out=d_w2)
                np.clip(d_b2, -1.0, 1.0, out=d_b2)

                w2 -= learning_rate * d_w2
                b2 -= learning_rate * d_b2
                w1 -= learning_rate * d_w1
                b1 -= learning_rate * d_b1

                if (
                    not np.all(np.isfinite(w1))
                    or not np.all(np.isfinite(b1))
                    or not np.all(np.isfinite(w2))
                    or not np.all(np.isfinite(b2))
                ):
                    break

            _, _, full_pred = NeuralNetworkMath.forward(X, w1, b1, w2, b2)

            if not np.all(np.isfinite(full_pred)):
                break

            current_loss = NeuralNetworkMath.mse_loss(y, full_pred, w1, w2, l2)

            if not np.isfinite(current_loss):
                break

            if best_loss - current_loss > tol:
                best_loss = current_loss
                best_params = (w1.copy(), b1.copy(), w2.copy(), b2.copy())
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                break

        return best_params

    @staticmethod
    def predict_one_hidden_layer_regressor(
        X: np.ndarray,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
    ) -> np.ndarray:
        _, _, y_hat = NeuralNetworkMath.forward(X, w1, b1, w2, b2)
        return y_hat.ravel()