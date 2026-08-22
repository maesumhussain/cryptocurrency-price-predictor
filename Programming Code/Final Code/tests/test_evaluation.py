from types import SimpleNamespace

from evaluation import WalkForwardEvaluator


def test_evaluator_runs_and_returns_evaluation_results(cfg, engineered_dataset, monkeypatch):
    evaluator = WalkForwardEvaluator(cfg)

    fake_best = SimpleNamespace(
        ridge_alpha=1.0,
        lasso_alpha=0.001,
        kernel_ridge_alpha=0.1,
        kernel_ridge_gamma=0.5,
        knn_k=5,
        knn_weights="uniform",
        knn_p=2,
        nn_hidden_units=8,
        nn_learning_rate=0.001,
        nn_l2=0.0001,
    )

    monkeypatch.setattr(evaluator.tuner, "tune", lambda X, y, verbose=False: fake_best)

    results = evaluator.evaluate(engineered_dataset)

    assert hasattr(results, "models_metrics")
    assert hasattr(results, "avg_metrics_for_bar")

    assert "model_names" in results.avg_metrics_for_bar
    assert len(results.avg_metrics_for_bar["model_names"]) == 6

    for model_name in ["OLS", "Ridge", "Lasso", "KernelRidge", "KNN", "NeuralNet"]:
        assert model_name in results.models_metrics