import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import yfinance as yf
from zoneinfo import ZoneInfo

# 1. DOWNLOAD BTC-USD (LONDON CLOSE) AND FEATURE SETUP

london = ZoneInfo("Europe/London")
now_london = datetime.datetime.now(london)

# Use data up to yesterday's London close (last completed day)
yesterday_london = (now_london - datetime.timedelta(days=1)).date()
end_date = yesterday_london + datetime.timedelta(days=1)

btc = yf.download(
    "BTC-USD",
    start="2015-01-01",
    end=str(end_date),
    interval="1d",
    auto_adjust=True,
    progress=False,
    ignore_tz=True,
)

btc = btc.dropna()
btc["Price"] = btc["Close"]

# Basic return & lag features
btc["Return"] = btc["Price"].pct_change()
btc["LagReturn1"] = btc["Return"].shift(1)
btc["LagReturn2"] = btc["Return"].shift(2)
btc["LagReturn3"] = btc["Return"].shift(3)

# Moving averages
btc["MA5"] = btc["Price"].rolling(5).mean()
btc["MA10"] = btc["Price"].rolling(10).mean()
btc["MA20"] = btc["Price"].rolling(20).mean()

# Volatility
btc["Volatility10"] = btc["Return"].rolling(10).std()

# Relative Strength Index (RSI)
delta = btc["Price"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
window = 14
avg_gain = gain.rolling(window).mean()
avg_loss = loss.rolling(window).mean()
rs = avg_gain / avg_loss
btc["RSI"] = 100 - (100 / (1 + rs))

# Exponential Moving Averages (EMA)
btc["EMA10"] = btc["Price"].ewm(span=10, adjust=False).mean()
btc["EMA20"] = btc["Price"].ewm(span=20, adjust=False).mean()
btc["EMA50"] = btc["Price"].ewm(span=50, adjust=False).mean()

# MACD
ema12 = btc["Price"].ewm(span=12, adjust=False).mean()
ema26 = btc["Price"].ewm(span=26, adjust=False).mean()
btc["MACD"] = ema12 - ema26
btc["MACD_signal"] = btc["MACD"].ewm(span=9, adjust=False).mean()

# Momentum
btc["Mom7"] = btc["Price"].pct_change(7)
btc["Mom30"] = btc["Price"].pct_change(30)

# Bollinger band width (20-day)
btc["BB_STD20"] = btc["Price"].rolling(20).std()
btc["BBWidth20"] = (4.0 * btc["BB_STD20"]) / btc["MA20"]

# ATR (14)
high = btc["High"]
low = btc["Low"]
close = btc["Close"]
prev_close = close.shift(1)

tr1 = high - low
tr2 = (high - prev_close).abs()
tr3 = (low - prev_close).abs()
btc["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
btc["ATR14"] = btc["TR"].rolling(14).mean()

# Non-linear transforms
btc["ReturnSq"] = btc["Return"] ** 2
btc["LagReturnSq1"] = btc["ReturnSq"].shift(1)
btc["RetVolInteraction"] = btc["Return"] * btc["Volatility10"]

# Keep a copy for latest-row prediction BEFORE we drop NaNs / target shift
btc_full = btc.copy()

# Target: next-day return
btc["TargetReturn"] = btc["Return"].shift(-1)

# Drop rows with NaNs in features/target
btc = btc.dropna()

# Dataset matrices
feature_columns = [
    "LagReturn1",
    "LagReturn2",
    "LagReturn3",
    "MA5",
    "MA10",
    "MA20",
    "Volatility10",
    "RSI",
    "EMA10",
    "EMA20",
    "EMA50",
    "MACD",
    "MACD_signal",
    "Mom7",
    "Mom30",
    "BBWidth20",
    "ATR14",
    "ReturnSq",
    "LagReturnSq1",
    "RetVolInteraction",
]

X_full = btc[feature_columns].values
y_full = btc["TargetReturn"].values
prices_full = btc["Price"].values

n_samples, n_features = X_full.shape

print("BTC dataset for modelling (London close)")
print(f"London date used (last complete day): {yesterday_london}")
print(f"Total samples: {n_samples}, features: {n_features}")
print()

# 2. METRIC FUNCTIONS

def mean_squared_error(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


def directional_accuracy(y_true, y_pred):
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)
    mask = sign_true != 0
    if mask.sum() == 0:
        return np.nan
    return float((sign_true[mask] == sign_pred[mask]).mean())

# 3. OLS / RIDGE (CLOSED-FORM) + LASSO (COORDINATE DESCENT)

def ridge_closed(X, y, alpha=1.0, fit_intercept=True, regularize_intercept=False):
    """
    Closed-form ridge (normal equation with L2 penalty).

    J(β) = (y - Xβ)^T (y - Xβ) + λ β^T β
    alpha = 0.0 -> OLS
    alpha > 0.0 -> Ridge
    """
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
    else:
        return w, 0.0


def compute_predictions(X, coef, intercept=0.0):
    return X @ coef + intercept


def soft_threshold(rho, lam):
    if rho < -lam:
        return rho + lam
    elif rho > lam:
        return rho - lam
    else:
        return 0.0


def lasso_cd(X, y, alpha=0.001, max_iter=1000, tol=1e-4):
    """
    Self-implemented Lasso using coordinate descent.

    J(β) = (y - Xβ)^T (y - Xβ) + λ Σ_j |β_j|
    Assumes X is standardised. y is centred. Intercept recovered separately.
    """
    n_samples, n_features = X.shape

    y_mean = y.mean()
    y_centered = y - y_mean

    w = np.zeros(n_features)

    for _ in range(max_iter):
        w_old = w.copy()

        for j in range(n_features):
            residual = y_centered - (X @ w) + X[:, j] * w[j]

            rho_j = np.dot(X[:, j], residual)
            z_j = np.dot(X[:, j], X[:, j])

            if z_j == 0:
                continue

            w[j] = soft_threshold(rho_j, alpha) / z_j

        if np.max(np.abs(w - w_old)) < tol:
            break

    intercept = y_mean
    return w, float(intercept)

# 3B. WRAPPERS SO WE CAN USE STANDARD GRIDSEARCHCV (WHILE KEEPING YOUR IMPLEMENTATIONS)

class RidgeClosedFormRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, fit_intercept=True, regularize_intercept=False):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.regularize_intercept = regularize_intercept

    def fit(self, X, y):
        coef, intercept = ridge_closed(
            X, y,
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            regularize_intercept=self.regularize_intercept,
        )
        self.coef_ = coef
        self.intercept_ = intercept
        return self

    def predict(self, X):
        return compute_predictions(X, self.coef_, self.intercept_)


class LassoCDRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.001, max_iter=5000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        # lasso_cd assumes X is standardised; StandardScaler in Pipeline ensures this
        coef, intercept = lasso_cd(
            X, y,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        self.coef_ = coef
        self.intercept_ = intercept
        return self

    def predict(self, X):
        return compute_predictions(X, self.coef_, self.intercept_)

# 4. WALK-FORWARD SETTINGS

one_year_days = 365
test_window = 90

# Alpha grids
ridge_alpha_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
lasso_alpha_grid = [0.0001, 0.001, 0.01, 0.03, 0.05]

# 5. STANDARD HYPERPARAMETER TUNING (Pipeline + TimeSeriesSplit + GridSearchCV)
#    (Uses YOUR implementations via wrappers above.)

print("Standard hyperparameter tuning (GridSearchCV + TimeSeriesSplit):")
print("---------------------------------------------------------------")

tscv = TimeSeriesSplit(n_splits=5)

ridge_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RidgeClosedFormRegressor()),
])

lasso_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LassoCDRegressor(max_iter=5000, tol=1e-4)),
])

ridge_gs = GridSearchCV(
    ridge_pipe,
    {"model__alpha": ridge_alpha_grid},
    cv=tscv,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
)

lasso_gs = GridSearchCV(
    lasso_pipe,
    {"model__alpha": lasso_alpha_grid},
    cv=tscv,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
)

ridge_gs.fit(X_full, y_full)
lasso_gs.fit(X_full, y_full)

best_ridge_alpha = ridge_gs.best_params_["model__alpha"]
best_lasso_alpha = lasso_gs.best_params_["model__alpha"]

print(f"Ridge best alpha: {best_ridge_alpha}")
print(f"Lasso best alpha: {best_lasso_alpha}")
print("---------------------------------------------------------------\n")

alpha_value = best_ridge_alpha
lasso_alpha = best_lasso_alpha
lasso_max_iter = 5000
lasso_tol = 1e-4

# 6. MAIN WALK-FORWARD EVALUATION WITH TUNED ALPHAS

models_metrics = {
    "OLS":   {"mse_return": [], "r2_return": [], "mse_price": [], "dir_acc": [],
              "y_true_return": [], "y_pred_return": [],
              "true_price": [], "pred_price": []},
    "Ridge": {"mse_return": [], "r2_return": [], "mse_price": [], "dir_acc": [],
              "y_true_return": [], "y_pred_return": [],
              "true_price": [], "pred_price": []},
    "Lasso": {"mse_return": [], "r2_return": [], "mse_price": [], "dir_acc": [],
              "y_true_return": [], "y_pred_return": [],
              "true_price": [], "pred_price": []},
}

train_end = one_year_days

while train_end + test_window <= n_samples:
    X_train_raw = X_full[:train_end]
    y_train = y_full[:train_end]

    X_test_raw = X_full[train_end:train_end + test_window]
    y_test = y_full[train_end:train_end + test_window]
    price_test = prices_full[train_end:train_end + test_window]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # OLS
    ols_coef, ols_intercept = ridge_closed(X_train, y_train, alpha=0.0)
    y_pred_ols = compute_predictions(X_test, ols_coef, ols_intercept)

    true_price_test = price_test * (1.0 + y_test)
    pred_price_ols = price_test * (1.0 + y_pred_ols)

    models_metrics["OLS"]["mse_return"].append(mean_squared_error(y_test, y_pred_ols))
    models_metrics["OLS"]["r2_return"].append(r2_score(y_test, y_pred_ols))
    models_metrics["OLS"]["mse_price"].append(mean_squared_error(true_price_test, pred_price_ols))
    models_metrics["OLS"]["dir_acc"].append(directional_accuracy(y_test, y_pred_ols))
    models_metrics["OLS"]["y_true_return"].append(y_test)
    models_metrics["OLS"]["y_pred_return"].append(y_pred_ols)
    models_metrics["OLS"]["true_price"].append(true_price_test)
    models_metrics["OLS"]["pred_price"].append(pred_price_ols)

    # Ridge
    ridge_coef, ridge_intercept = ridge_closed(X_train, y_train, alpha=alpha_value)
    y_pred_ridge = compute_predictions(X_test, ridge_coef, ridge_intercept)
    pred_price_ridge = price_test * (1.0 + y_pred_ridge)

    models_metrics["Ridge"]["mse_return"].append(mean_squared_error(y_test, y_pred_ridge))
    models_metrics["Ridge"]["r2_return"].append(r2_score(y_test, y_pred_ridge))
    models_metrics["Ridge"]["mse_price"].append(mean_squared_error(true_price_test, pred_price_ridge))
    models_metrics["Ridge"]["dir_acc"].append(directional_accuracy(y_test, y_pred_ridge))
    models_metrics["Ridge"]["y_true_return"].append(y_test)
    models_metrics["Ridge"]["y_pred_return"].append(y_pred_ridge)
    models_metrics["Ridge"]["true_price"].append(true_price_test)
    models_metrics["Ridge"]["pred_price"].append(pred_price_ridge)

    # Lasso
    lasso_coef, lasso_intercept = lasso_cd(
        X_train, y_train, alpha=lasso_alpha, max_iter=lasso_max_iter, tol=lasso_tol
    )
    y_pred_lasso = compute_predictions(X_test, lasso_coef, lasso_intercept)
    pred_price_lasso = price_test * (1.0 + y_pred_lasso)

    models_metrics["Lasso"]["mse_return"].append(mean_squared_error(y_test, y_pred_lasso))
    models_metrics["Lasso"]["r2_return"].append(r2_score(y_test, y_pred_lasso))
    models_metrics["Lasso"]["mse_price"].append(mean_squared_error(true_price_test, pred_price_lasso))
    models_metrics["Lasso"]["dir_acc"].append(directional_accuracy(y_test, y_pred_lasso))
    models_metrics["Lasso"]["y_true_return"].append(y_test)
    models_metrics["Lasso"]["y_pred_return"].append(y_pred_lasso)
    models_metrics["Lasso"]["true_price"].append(true_price_test)
    models_metrics["Lasso"]["pred_price"].append(pred_price_lasso)

    train_end += one_year_days

# 7. SUMMARY OF WALK-FORWARD RESULTS

print("Expanding-window Walk-forward Summary (London close data)")
print("========================================================")
print(f"Base train window (first step):  {one_year_days} days (~1 year)")
print(f"Test window size (each step):    {test_window} days (~3 months)")
print()

avg_metrics_for_bar = {"model_names": [], "mse_return": [], "mse_price": [], "dir_acc": []}

for name, metrics in models_metrics.items():
    n_windows = len(metrics["mse_return"])
    if n_windows == 0:
        print(f"{name}: Not enough data for even one window.\n")
        continue

    avg_mse_return = float(np.mean(metrics["mse_return"]))
    avg_r2_return = float(np.mean(metrics["r2_return"]))
    avg_mse_price = float(np.mean(metrics["mse_price"]))
    avg_dir_acc = float(np.mean(metrics["dir_acc"]))

    avg_metrics_for_bar["model_names"].append(name)
    avg_metrics_for_bar["mse_return"].append(avg_mse_return)
    avg_metrics_for_bar["mse_price"].append(avg_mse_price)
    avg_metrics_for_bar["dir_acc"].append(avg_dir_acc * 100)

    print(f"{name} model:")
    print(f"  Number of windows:           {n_windows}")
    print(f"  Avg Test MSE (return):       {avg_mse_return:.8f}")
    print(f"  Avg Test R^2 (return):       {avg_r2_return:.6f}")
    print(f"  Avg Test MSE (price):        {avg_mse_price:.2f}")
    print(f"  Avg Directional Accuracy:    {avg_dir_acc * 100:.2f}%")
    print()

print("========================================================\n")

# 8. PLOTS – MODEL COMPARISON AND TRUE VS PREDICTED

model_names = avg_metrics_for_bar["model_names"]

plt.figure()
plt.bar(model_names, avg_metrics_for_bar["mse_return"])
plt.ylabel("Average Test MSE (Return)")
plt.title("Average Return MSE by Model (Walk-forward)")
plt.tight_layout()
plt.show()

plt.figure()
plt.bar(model_names, avg_metrics_for_bar["mse_price"])
plt.ylabel("Average Test MSE (Price)")
plt.title("Average Price MSE by Model (Walk-forward)")
plt.tight_layout()
plt.show()

plt.figure()
plt.bar(model_names, avg_metrics_for_bar["dir_acc"])
plt.ylabel("Average Directional Accuracy (%)")
plt.ylim(40, 60)
plt.title("Directional Accuracy by Model (Walk-forward)")
plt.tight_layout()
plt.show()

for name in models_metrics.keys():
    y_true_all = np.concatenate(models_metrics[name]["y_true_return"])
    y_pred_all = np.concatenate(models_metrics[name]["y_pred_return"])

    min_val = min(y_true_all.min(), y_pred_all.min())
    max_val = max(y_true_all.max(), y_pred_all.max())

    plt.figure()
    plt.scatter(y_true_all, y_pred_all, s=8, alpha=0.6)
    plt.plot([min_val, max_val], [min_val, max_val], linewidth=2)
    plt.xlabel("True Returns (Test)")
    plt.ylabel("Predicted Returns (Test)")
    plt.title(f"{name} – True vs Predicted Returns (Walk-forward)")
    plt.tight_layout()
    plt.show()

for name in models_metrics.keys():
    true_price_all = np.concatenate(models_metrics[name]["true_price"])
    pred_price_all = np.concatenate(models_metrics[name]["pred_price"])

    min_val = min(true_price_all.min(), pred_price_all.min())
    max_val = max(true_price_all.max(), pred_price_all.max())

    plt.figure()
    plt.scatter(true_price_all, pred_price_all, s=8, alpha=0.6)
    plt.plot([min_val, max_val], [min_val, max_val], linewidth=2)
    plt.xlabel("True Prices (Test)")
    plt.ylabel("Predicted Prices (Test)")
    plt.title(f"{name} – True vs Predicted Prices (Walk-forward)")
    plt.tight_layout()
    plt.show()

# 9. FIT FINAL MODELS ON ALL DATA & PREDICT TOMORROW

scaler_final = StandardScaler()
X_full_scaled = scaler_final.fit_transform(X_full)

ols_coef_full, ols_intercept_full = ridge_closed(X_full_scaled, y_full, alpha=0.0)
ridge_coef_full, ridge_intercept_full = ridge_closed(X_full_scaled, y_full, alpha=alpha_value)
lasso_coef_full, lasso_intercept_full = lasso_cd(
    X_full_scaled, y_full,
    alpha=lasso_alpha,
    max_iter=lasso_max_iter,
    tol=lasso_tol
)

latest_row = btc_full.iloc[-1]

latest_features = np.array([
    latest_row["LagReturn1"],
    latest_row["LagReturn2"],
    latest_row["LagReturn3"],
    latest_row["MA5"],
    latest_row["MA10"],
    latest_row["MA20"],
    latest_row["Volatility10"],
    latest_row["RSI"],
    latest_row["EMA10"],
    latest_row["EMA20"],
    latest_row["EMA50"],
    latest_row["MACD"],
    latest_row["MACD_signal"],
    latest_row["Mom7"],
    latest_row["Mom30"],
    latest_row["BBWidth20"],
    latest_row["ATR14"],
    latest_row["ReturnSq"],
    latest_row["LagReturnSq1"],
    latest_row["RetVolInteraction"],
]).reshape(1, -1)

if np.isnan(latest_features).any():
    print("Cannot predict tomorrow's price: latest feature row contains NaN values.")
else:
    latest_features_scaled = scaler_final.transform(latest_features)

    pred_ret_ols = compute_predictions(latest_features_scaled, ols_coef_full, ols_intercept_full).item()
    pred_ret_ridge = compute_predictions(latest_features_scaled, ridge_coef_full, ridge_intercept_full).item()
    pred_ret_lasso = compute_predictions(latest_features_scaled, lasso_coef_full, lasso_intercept_full).item()

    latest_date = btc_full.index[-1]
    price_obj = latest_row["Price"]
    latest_price = float(price_obj.iloc[0]) if isinstance(price_obj, pd.Series) else float(price_obj)

    tomorrow_date = latest_date + datetime.timedelta(days=1)

    pred_price_ols = latest_price * (1.0 + pred_ret_ols)
    pred_price_ridge = latest_price * (1.0 + pred_ret_ridge)
    pred_price_lasso = latest_price * (1.0 + pred_ret_lasso)

    print("---------------------------------------------")
    print(" Latest BTC Close Price and Tomorrow's Prediction (London close)")
    print("---------------------------------------------")
    print(f"Latest London Date (last completed day): {latest_date.date()}")
    print(f"Latest BTC Close Price: ${latest_price:,.2f}")
    print()
    print("Predicted Return and Price for", tomorrow_date.date())
    print()
    print("  OLS:")
    print(f"    Predicted Return: {pred_ret_ols * 100:.3f}%")
    print(f"    Predicted Price:  ${pred_price_ols:,.2f}")
    print()
    print(f"  Ridge (alpha={alpha_value}):")
    print(f"    Predicted Return: {pred_ret_ridge * 100:.3f}%")
    print(f"    Predicted Price:  ${pred_price_ridge:,.2f}")
    print()
    print(f"  Lasso (alpha={lasso_alpha}):")
    print(f"    Predicted Return: {pred_ret_lasso * 100:.3f}%")
    print(f"    Predicted Price:  ${pred_price_lasso:,.2f}")
    print("---------------------------------------------\n")