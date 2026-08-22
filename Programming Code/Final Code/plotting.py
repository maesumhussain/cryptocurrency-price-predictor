import os
import numpy as np
import matplotlib.pyplot as plt
from config import EvaluationResults


class Plotter:
    def __init__(self):
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _get_percentile_limits(x, y, lower=1, upper=99, padding_ratio=0.08):
        combined = np.concatenate([x, y])
        combined = combined[np.isfinite(combined)]

        if combined.size == 0:
            return -1.0, 1.0

        low = float(np.percentile(combined, lower))
        high = float(np.percentile(combined, upper))

        if abs(high - low) < 1e-12:
            center = low
            pad = max(abs(center) * padding_ratio, 1e-3)
            return center - pad, center + pad

        pad = (high - low) * padding_ratio
        return low - pad, high + pad

    def plot(self, results: EvaluationResults, asset_display: str) -> None:
        model_names = results.avg_metrics_for_bar["model_names"]

        # Bar chart: Return MSE
        plt.figure()
        plt.bar(model_names, results.avg_metrics_for_bar["mse_return"])
        plt.ylabel("Average Test MSE (Return)")
        plt.title(f"{asset_display} - Average Return MSE by Model")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/mse_return.png", dpi=300)
        plt.close()

        # Bar chart: Price MSE
        plt.figure()
        plt.bar(model_names, results.avg_metrics_for_bar["mse_price"])
        plt.ylabel("Average Test MSE (Price)")
        plt.title(f"{asset_display} - Average Price MSE by Model")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/mse_price.png", dpi=300)
        plt.close()

        # Bar chart: Directional Accuracy
        dir_acc_percent = [value * 100 for value in results.avg_metrics_for_bar["dir_acc"]]

        plt.figure()
        plt.bar(model_names, dir_acc_percent)
        plt.ylabel("Average Directional Accuracy (%)")
        plt.ylim(40, 60)
        plt.title(f"{asset_display} - Directional Accuracy by Model")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/directional_accuracy.png", dpi=300)
        plt.close()

        # Scatter: Returns (%)
        for name in results.models_metrics.keys():
            y_true_all = np.concatenate(results.models_metrics[name]["y_true_return"]) * 100
            y_pred_all = np.concatenate(results.models_metrics[name]["y_pred_return"]) * 100

            x_min, x_max = self._get_percentile_limits(y_true_all, y_pred_all)
            y_min, y_max = x_min, x_max

            plt.figure()
            plt.scatter(y_true_all, y_pred_all, s=8, alpha=0.6)

            # Best-fit line for returns
            if len(y_true_all) >= 2 and np.std(y_true_all) > 1e-12:
                slope, intercept = np.polyfit(y_true_all, y_pred_all, 1)
                x_fit = np.linspace(x_min, x_max, 200)
                y_fit = slope * x_fit + intercept
                plt.plot(x_fit, y_fit, linewidth=2)

            plt.xlabel("True Returns (%)")
            plt.ylabel("Predicted Returns (%)")
            plt.title(f"{asset_display} - {name} True vs Predicted Returns")
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)

            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/{name}_returns_scatter.png", dpi=300)
            plt.close()

        # Scatter: Prices
        for name in results.models_metrics.keys():
            true_price_all = np.concatenate(results.models_metrics[name]["true_price"])
            pred_price_all = np.concatenate(results.models_metrics[name]["pred_price"])

            min_val = float(min(true_price_all.min(), pred_price_all.min()))
            max_val = float(max(true_price_all.max(), pred_price_all.max()))

            if abs(max_val - min_val) < 1e-12:
                pad = max(abs(min_val) * 0.05, 1.0)
                min_val -= pad
                max_val += pad

            plt.figure()
            plt.scatter(true_price_all, pred_price_all, s=8, alpha=0.6)

            # Ideal line y = x for prices
            plt.plot([min_val, max_val], [min_val, max_val], linewidth=2)

            plt.xlabel("True Prices (Test)")
            plt.ylabel("Predicted Prices (Test)")
            plt.title(f"{asset_display} - {name} True vs Predicted Prices")
            plt.xlim(min_val, max_val)
            plt.ylim(min_val, max_val)

            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/{name}_prices_scatter.png", dpi=300)
            plt.close()