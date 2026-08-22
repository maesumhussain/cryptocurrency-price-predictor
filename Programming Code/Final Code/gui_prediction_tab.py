from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider, QPushButton,
    QFrame, QMessageBox, QApplication, QGridLayout
)
from PyQt5.QtCore import Qt

from crypto_pairs import CRYPTO_PAIRS
from data_loader import InsufficientDataError
from gui_helpers import create_prediction_card, create_info_card


def build_prediction_tab(app, tab_widget):
    app.current_prediction_usd = None
    app.current_fx_rate = None
    app.current_currency = "USD"

    main_layout = QVBoxLayout()
    main_layout.setSpacing(14)
    main_layout.setContentsMargins(0, 0, 0, 0)

    controls_frame = QFrame()
    controls_frame.setObjectName("sectionFrame")

    controls_layout = QVBoxLayout()
    controls_layout.setContentsMargins(16, 14, 16, 14)
    controls_layout.setSpacing(14)

    top_controls = QHBoxLayout()
    top_controls.setSpacing(16)

    pair_col = QVBoxLayout()
    pair_col.setSpacing(8)

    pair_label = QLabel("Cryptocurrency Pair")
    pair_label.setObjectName("sectionTitleSmall")

    app.pair_dropdown = QComboBox()
    app.pair_dropdown.addItems(list(CRYPTO_PAIRS.keys()))
    app.pair_dropdown.setCurrentText(app.cfg.default_pair_display)
    app.pair_dropdown.currentTextChanged.connect(
        lambda value: app.selected_pair_header_label.setText(f"Selected Pair: {value}")
    )

    pair_col.addWidget(pair_label)
    pair_col.addWidget(app.pair_dropdown)

    years_col = QVBoxLayout()
    years_col.setSpacing(8)

    years_title = QLabel("Historical Data Window")
    years_title.setObjectName("sectionTitleSmall")

    years_slider_row = QHBoxLayout()
    years_slider_row.setSpacing(12)

    app.years_slider = QSlider(Qt.Horizontal)
    app.years_slider.setMinimum(app.cfg.min_years)
    app.years_slider.setMaximum(app.cfg.max_years)
    app.years_slider.setValue(app.cfg.default_years)
    app.years_slider.setTickInterval(1)
    app.years_slider.setSingleStep(1)
    app.years_slider.setTickPosition(QSlider.NoTicks)

    app.years_value_label = QLabel(f"{app.cfg.default_years} years")
    app.years_value_label.setObjectName("sliderValueLabel")
    app.years_value_label.setAlignment(Qt.AlignCenter)
    app.years_value_label.setFixedWidth(95)

    app.years_slider.valueChanged.connect(
        lambda value: app.years_value_label.setText(f"{value} years")
    )

    years_slider_row.addWidget(app.years_slider)
    years_slider_row.addWidget(app.years_value_label)

    years_col.addWidget(years_title)
    years_col.addLayout(years_slider_row)

    currency_col = QVBoxLayout()
    currency_col.setSpacing(8)

    currency_title = QLabel("Display Currency")
    currency_title.setObjectName("sectionTitleSmall")

    app.currency_dropdown = QComboBox()
    app.currency_dropdown.addItems(["USD", "GBP"])
    app.currency_dropdown.setCurrentText("USD")
    app.currency_dropdown.currentTextChanged.connect(
        lambda value: update_prediction_currency_display(app, value)
    )

    currency_col.addWidget(currency_title)
    currency_col.addWidget(app.currency_dropdown)

    top_controls.addLayout(pair_col, 3)
    top_controls.addLayout(years_col, 4)
    top_controls.addLayout(currency_col, 2)

    controls_layout.addLayout(top_controls)

    lower_controls = QHBoxLayout()
    lower_controls.setSpacing(14)

    app.exchange_rate_label = QLabel("Exchange Rate: --")
    app.exchange_rate_label.setObjectName("exchangeRateLabel")

    app.predict_button = QPushButton("Predict Tomorrow's Price and Change")
    app.predict_button.clicked.connect(app.run_prediction)
    app.predict_button.setObjectName("primaryButton")

    app.status_label = QLabel("Ready")
    app.status_label.setObjectName("statusLabel")
    app.status_label.setAlignment(Qt.AlignCenter)
    app.status_label.setFixedWidth(190)

    lower_controls.addWidget(app.exchange_rate_label, 2)
    lower_controls.addWidget(app.predict_button, 4)
    lower_controls.addWidget(app.status_label, 1)

    controls_layout.addLayout(lower_controls)
    controls_frame.setLayout(controls_layout)

    main_layout.addWidget(controls_frame)

    summary_row = QHBoxLayout()
    summary_row.setSpacing(12)

    app.latest_price_card = create_info_card("Latest Close Price", "--")
    summary_row.addWidget(app.latest_price_card, 2)

    main_layout.addLayout(summary_row)

    predictions_wrapper = QFrame()
    predictions_wrapper.setObjectName("sectionFrame")

    app.prediction_grid = QGridLayout()
    app.prediction_grid.setContentsMargins(14, 14, 14, 14)
    app.prediction_grid.setHorizontalSpacing(14)
    app.prediction_grid.setVerticalSpacing(14)
    app.prediction_grid.setColumnStretch(0, 1)
    app.prediction_grid.setColumnStretch(1, 1)
    app.prediction_grid.setColumnStretch(2, 1)
    app.prediction_grid.setRowStretch(0, 1)
    app.prediction_grid.setRowStretch(1, 1)

    app.model_cards = {}
    models = ["OLS", "Ridge", "Lasso", "KernelRidge", "KNN", "NeuralNet"]
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for model, (row, col) in zip(models, positions):
        card = create_prediction_card(model)
        app.model_cards[model] = card
        app.prediction_grid.addWidget(card["frame"], row, col)

    predictions_wrapper.setLayout(app.prediction_grid)
    main_layout.addWidget(predictions_wrapper, 1)

    tab_widget.setLayout(main_layout)


def format_params(params_text):
    if not params_text:
        return "Params: --"
    return f"Params: {params_text}"


def format_price(value_usd, currency, fx_rate):
    if value_usd is None:
        return "--"

    display_value = float(value_usd)
    symbol = "$"

    if currency == "GBP":
        if fx_rate is None:
            return "--"
        display_value = display_value * fx_rate
        symbol = "£"

    abs_value = abs(display_value)

    if abs_value >= 1000:
        decimals = 2
    elif abs_value >= 1:
        decimals = 2
    elif abs_value >= 0.01:
        decimals = 4
    elif abs_value >= 0.0001:
        decimals = 6
    else:
        decimals = 10

    return f"{symbol}{display_value:,.{decimals}f}"


def update_prediction_currency_display(app, currency):
    app.current_currency = currency

    if app.current_fx_rate is not None:
        app.exchange_rate_label.setText(
            f"Exchange Rate: 1 USD = {app.current_fx_rate:.4f} GBP"
        )
    else:
        app.exchange_rate_label.setText("Exchange Rate: --")

    if not app.current_prediction_usd:
        return

    latest_price_usd = app.current_prediction_usd.get("latest_price_usd")
    app.latest_price_card.title_label.setText("Latest Close Price")
    app.latest_price_card.value_label.setText(
        format_price(latest_price_usd, currency, app.current_fx_rate)
    )

    model_values = app.current_prediction_usd.get("models", {})
    for model, values in model_values.items():
        if model in app.model_cards:
            app.model_cards[model]["return"].setText(values.get("return_display", "--"))
            app.model_cards[model]["price"].setText(
                format_price(values.get("price_usd"), currency, app.current_fx_rate)
            )
            app.model_cards[model]["params"].setText(
                format_params(values.get("params"))
            )


def run_prediction_workflow(app):
    try:
        app.status_label.setText("Running...")
        QApplication.processEvents()

        selected_display = app.pair_dropdown.currentText()
        selected_yahoo = CRYPTO_PAIRS[selected_display]
        years_requested = int(app.years_slider.value())

        app.selected_pair_header_label.setText(f"Selected Pair: {selected_display}")

        try:
            btc, yesterday, available_years = app.loader.load(selected_yahoo, years_requested)
        except InsufficientDataError as e:
            app.years_slider.setValue(e.available_years_floor)
            app.years_value_label.setText(f"{e.available_years_floor} years")
            app.status_label.setText("Needs Adjustment")
            QMessageBox.warning(
                app,
                "Not Enough Historical Data",
                (
                    f"Only {e.available_years_exact:.2f} years of data are available for "
                    f"{selected_display}.\n\n"
                    f"The slider has been adjusted to {e.available_years_floor} years.\n"
                    f"Please run the model again."
                ),
            )
            return False

        app.dataset = app.fe.build_dataset(
            btc=btc,
            yesterday_london=yesterday,
            asset_display=selected_display,
            asset_yahoo=selected_yahoo,
            years_requested=years_requested,
            available_years=available_years,
        )

        app.results = app.evaluator.evaluate(app.dataset)
        app.plotter.plot(app.results, app.dataset.asset_display)

        prediction_result = app.predictor.predict(app.dataset)

        app.current_fx_rate = app.loader.load_usdgbp_rate()
        app.current_prediction_usd = prediction_result

        for model, values in prediction_result["models"].items():
            if model in app.model_cards:
                return_text = values.get("return_display", "--")

                if return_text.startswith("-"):
                    app.model_cards[model]["return"].setStyleSheet(
                        "color:#f87171; font-size:20px; font-weight:700;"
                    )
                else:
                    app.model_cards[model]["return"].setStyleSheet(
                        "color:#22c55e; font-size:20px; font-weight:700;"
                    )

        update_prediction_currency_display(app, app.currency_dropdown.currentText())

        app.status_label.setText("Complete")

        QMessageBox.information(
            app,
            "Success",
            f"Prediction completed successfully for {selected_display}."
        )
        return True

    except Exception as e:
        app.status_label.setText("Error")
        QMessageBox.critical(app, "Error", f"Something went wrong:\n{str(e)}")
        return False