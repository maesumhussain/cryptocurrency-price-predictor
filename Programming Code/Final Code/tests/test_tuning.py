from tuning import HyperparameterTuner


class FakeGridSearchCV:
    call_count = 0

    def __init__(self, estimator, param_grid, cv, scoring, n_jobs):
        self.estimator = estimator
        self.param_grid = param_grid

    def fit(self, X, y):
        FakeGridSearchCV.call_count += 1

        if FakeGridSearchCV.call_count == 1:
            self.best_params_ = {"model__alpha": 10.0}
        elif FakeGridSearchCV.call_count == 2:
            self.best_params_ = {"model__alpha": 0.01}
        elif FakeGridSearchCV.call_count == 3:
            self.best_params_ = {"model__alpha": 0.1, "model__gamma": 0.5}
        elif FakeGridSearchCV.call_count == 4:
            self.best_params_ = {
                "model__n_neighbors": 7,
                "model__weights": "distance",
                "model__p": 2,
            }
        else:
            self.best_params_ = {
                "model__hidden_units": 16,
                "model__learning_rate": 0.001,
                "model__l2": 0.0001,
            }
        return self


def test_tuner_returns_best_params(cfg, monkeypatch):
    import tuning

    FakeGridSearchCV.call_count = 0
    monkeypatch.setattr(tuning, "GridSearchCV", FakeGridSearchCV)

    tuner = HyperparameterTuner(cfg)

    X = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    best = tuner.tune(X, y, verbose=False)

    assert best.ridge_alpha == 10.0
    assert best.lasso_alpha == 0.01
    assert best.kernel_ridge_alpha == 0.1
    assert best.kernel_ridge_gamma == 0.5
    assert best.knn_k == 7
    assert best.knn_weights == "distance"
    assert best.knn_p == 2
    assert best.nn_hidden_units == 16
    assert best.nn_learning_rate == 0.001
    assert best.nn_l2 == 0.0001