from types import SimpleNamespace

from predictor import TomorrowPredictor


def test_predictor_runs_and_returns_expected_structure(cfg, engineered_dataset, monkeypatch):
    predictor = TomorrowPredictor(cfg)

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

    monkeypatch.setattr(predictor.tuner, "tune", lambda X, y, verbose=True: fake_best)

    result = predictor.predict(engineered_dataset)

    assert "asset_display" in result
    assert "tomorrow_date" in result
    assert "latest_price_usd" in result
    assert "models" in result

    assert result["asset_display"] == "BTC/USD"
    assert isinstance(result["models"], dict)

    for model_name in ["OLS", "Ridge", "Lasso", "KernelRidge", "KNN", "NeuralNet"]:
        assert model_name in result["models"]
        assert "return_decimal" in result["models"][model_name]
        assert "price_usd" in result["models"][model_name]
        assert "return_display" in result["models"][model_name]
        assert "params" in result["models"][model_name]