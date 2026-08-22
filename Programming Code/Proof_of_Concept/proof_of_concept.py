import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge as SkRidge

random_number_generator = np.random.default_rng(23)
n_samples = 240
n_features = 3

# Data Set Up for Proof of Concept
X = random_number_generator.normal(size=(n_samples, n_features))
true_coefficents = np.array([4.6, -5.1 , 2.7])
true_bias = 0.75
noise = random_number_generator.normal(scale=1.0, size=n_samples)
y = X @ true_coefficents + true_bias + noise

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2307
)

# Self Implementation of Ridge Closed Regression
def ridge_closed(X, y, alpha=1.0, fit_intercept=True, regularize_intercept=False):
    if fit_intercept:
        X_extended = np.c_[np.ones(len(X)), X]
    else:
        X_extended = X
    XT_X_matrix = X_extended.T @ X_extended
    XT_y_vector = X_extended.T @ y
    regularization_matrix = np.eye(X_extended.shape[1])
    if fit_intercept and not regularize_intercept:
        regularization_matrix[0, 0] = 0.0
    regularization_system_matrix = XT_X_matrix + alpha * regularization_matrix
    parameter_vector = np.linalg.solve(regularization_system_matrix, XT_y_vector)
    if fit_intercept:
        return parameter_vector[1:], float(parameter_vector[0])
    else:
        return parameter_vector, 0.0


def compute_predictions(X, coefficient, intercept=0.0):
    return X @ coefficient + intercept

def mean_squared_error(y_true, y_prediction):
    return float(np.mean((y_true - y_prediction) ** 2))

def r2_score(y_true, y_prediction):
    residual_sum_of_squares = np.sum((y_true - y_prediction) ** 2)
    total_variance = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - (residual_sum_of_squares / total_variance))

alpha_value = 1.0

# Set up for OLS and Ridge Closed 
ols_coefficient, ols_intercept = ridge_closed(X_train, y_train, alpha=0.0)
ridge_coefficient, ridge_intercept   = ridge_closed(X_train, y_train, alpha=alpha_value)

y_predicted_ols_test = compute_predictions(X_test, ols_coefficient, ols_intercept)
y_predicted_ridge_test  = compute_predictions(X_test, ridge_coefficient, ridge_intercept)

# Set up for Ridge Model from SkLearn Library
sk_ridge_model = SkRidge(alpha=alpha_value, fit_intercept=True)
sk_ridge_model.fit(X_train, y_train)
sk_ridge_coefficient = sk_ridge_model.coef_
sk_ridge_intercept = float(sk_ridge_model.intercept_)
y_predicted_sk_ridge_test = sk_ridge_model.predict(X_test)

# Results
print("True Coefficients:", true_coefficents, " True Bias:", true_bias)

print("\nCoefficients")
print("OLS (alpha=0):", ols_coefficient, "  b =", round(ols_intercept, 3))
print(f"Closed Ridge (alpha={alpha_value}):", ridge_coefficient, "  b =", round(ridge_intercept, 3))
print(f"sklearn Ridge (alpha={alpha_value}):", sk_ridge_coefficient, "  b =", round(sk_ridge_intercept, 3))

print("\nTest Performance")
print("OLS     -> MSE:", round(mean_squared_error(y_test, y_predicted_ols_test), 4), " | R²:", round(r2_score(y_test, y_predicted_ols_test), 4))
print("NumPy   -> MSE:", round(mean_squared_error(y_test, y_predicted_ridge_test),  4), " | R²:", round(r2_score(y_test, y_predicted_ridge_test),  4))
print("sklearn -> MSE:", round(mean_squared_error(y_test, y_predicted_sk_ridge_test), 4), " | R²:", round(r2_score(y_test, y_predicted_sk_ridge_test), 4))

coefficient_difference = np.linalg.norm(ridge_coefficient - sk_ridge_coefficient)
print (f"Coefficient Difference between Self Implementation of Closed Ridge and Sklearn Ridge: ", round(coefficient_difference, 10))
bias_difference = abs(ridge_intercept - sk_ridge_intercept)
print (f"Bias Difference between Self Implementation of Closed Ridge and Sklearn Ridge: ", round(bias_difference , 10))

# Graph 1: Alpha vs Coefficient Value for Closed Ridge
alphas = np.logspace(-1, 5, 40)
coefficient_path = []

for a in alphas:
    c, _ = ridge_closed(X_train, y_train, alpha=a)
    coefficient_path.append(c)

coefficient_path = np.array(coefficient_path)

plt.figure()
for j in range(n_features):
    plt.plot(alphas, coefficient_path[:, j], label=f"w{j+1}")

plt.xscale('log')
plt.xlabel("Alpha (log scale)")
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.ylabel("Coefficient Value")
plt.title("Ridge Coefficient Paths (Closed Ridge Self Implementation)")
plt.legend()
plt.show()

# Graph 2: True vs Predicted Graph for OLS
plt.figure()
plt.scatter(y_test, y_predicted_ols_test, alpha=1)
minimum = min(np.min(y_test), np.min(y_predicted_ols_test))
maximum = max(np.max(y_test), np.max(y_predicted_ols_test))
plt.plot([minimum, maximum], [minimum, maximum])
plt.xlabel("True y (test)")
plt.ylabel("Predicted y (test)")
plt.title(f"OLS (alpha=0): True vs Predicted")
plt.show()

# Graph 3: True vs Predicted Graph for Closed Ridge
plt.figure()
plt.scatter(y_test, y_predicted_ridge_test, alpha=1)
minimum = min(np.min(y_test), np.min(y_predicted_ridge_test))
maximum = max(np.max(y_test), np.max(y_predicted_ridge_test))
plt.plot([minimum, maximum], [minimum, maximum])
plt.xlabel("True y (test)")
plt.ylabel("Predicted y (test)")
plt.title(f"Closed Ridge (alpha={alpha_value}): True vs Predicted")
plt.show()

# Graph 4: True vs Predicted Graph for Sklearn Ridge
plt.figure()
plt.scatter(y_test, y_predicted_sk_ridge_test, alpha=1)
minimum = min(np.min(y_test), np.min(y_predicted_sk_ridge_test))
maximum = max(np.max(y_test), np.max(y_predicted_sk_ridge_test))
plt.plot([minimum, maximum], [minimum, maximum])
plt.xlabel("True y (test)")
plt.ylabel("Predicted y (test)")
plt.title(f"Sklearn Ridge (alpha={alpha_value}): True vs Predicted")
plt.show()