from config import Config
from crypto_pairs import CRYPTO_PAIRS
from data_loader import BTCDataLoader, InsufficientDataError
from features import FeatureEngineer
from evaluation import WalkForwardEvaluator
from plotting import Plotter
from predictor import TomorrowPredictor


def main(
    pair_display: str = "BTC/USD",
    years_requested: int | None = None,
    make_plots: bool = True,
):
    cfg = Config()

    if years_requested is None:
        years_requested = cfg.default_years

    if pair_display not in CRYPTO_PAIRS:
        raise ValueError(
            f"Unsupported pair '{pair_display}'. "
            f"Choose from: {', '.join(CRYPTO_PAIRS.keys())}"
        )

    asset_yahoo = CRYPTO_PAIRS[pair_display]

    loader = BTCDataLoader(cfg)
    fe = FeatureEngineer(cfg)
    evaluator = WalkForwardEvaluator(cfg)
    plotter = Plotter()
    predictor = TomorrowPredictor(cfg)

    try:
        btc, yesterday_london, available_years = loader.load(asset_yahoo, years_requested)
    except InsufficientDataError as e:
        print(
            f"Only {e.available_years_exact:.2f} years of data are available for "
            f"{pair_display}."
        )
        print(f"Try again with {e.available_years_floor} years or fewer.")
        return

    dataset = fe.build_dataset(
        btc=btc,
        yesterday_london=yesterday_london,
        asset_display=pair_display,
        asset_yahoo=asset_yahoo,
        years_requested=years_requested,
        available_years=available_years,
    )

    n_samples, n_features = dataset.X.shape

    print("--------------------------------------------------")
    print("Console Crypto Forecasting Application")
    print("--------------------------------------------------")
    print(f"Asset: {dataset.asset_display}")
    print(f"Yahoo ticker: {dataset.asset_yahoo}")
    print(f"Years requested: {dataset.years_requested}")
    print(f"Years available: {dataset.available_years:.2f}")
    print(f"London date used (last complete day): {dataset.yesterday_london}")
    print(f"Total samples: {n_samples}")
    print(f"Total features: {n_features}")
    print("--------------------------------------------------")
    print()

    results = evaluator.evaluate(dataset)

    if make_plots:
        plotter.plot(results, dataset.asset_display)
        print("Plots saved to the 'plots' folder.")
        print()

    predictor.predict(dataset)


if __name__ == "__main__":
    main(pair_display="BTC/USD", years_requested=10, make_plots=True)