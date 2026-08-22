import numpy as np

class Metrics:
    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - ss_res / ss_tot)

    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        sign_true = np.sign(y_true)
        sign_pred = np.sign(y_pred)
        mask = sign_true != 0
        if mask.sum() == 0:
            return float("nan")
        return float((sign_true[mask] == sign_pred[mask]).mean())