import os

from PyQt5.QtWidgets import (
    QFrame, QLabel, QVBoxLayout, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap


def format_money(value: float, symbol: str = "$") -> str:
    abs_value = abs(float(value))

    if abs_value >= 1000:
        decimals = 2
    elif abs_value >= 1:
        decimals = 2
    elif abs_value >= 0.1:
        decimals = 4
    elif abs_value >= 0.01:
        decimals = 4
    elif abs_value >= 0.001:
        decimals = 6
    else:
        decimals = 8

    return f"{symbol}{value:,.{decimals}f}"


def format_price_by_currency(value_usd, currency, fx_rate):
    if value_usd is None:
        return "--"

    if currency == "GBP":
        if fx_rate is None:
            return "--"
        return format_money(value_usd * fx_rate, "£")

    return format_money(value_usd, "$")


def build_header(app):
    frame = QFrame()
    frame.setObjectName("headerFrame")

    layout = QVBoxLayout()
    layout.setContentsMargins(18, 12, 18, 12)
    layout.setSpacing(2)

    title = QLabel("Crypto Prediction Dashboard")
    title.setObjectName("titleLabel")

    app.selected_pair_header_label = QLabel(f"Selected Pair: {app.cfg.default_pair_display}")
    app.selected_pair_header_label.setObjectName("selectedPairHeaderLabel")

    subtitle = QLabel(
        "Forecast tomorrow's crypto price and review historical and model-specific performance"
    )
    subtitle.setObjectName("subtitleLabel")

    layout.addWidget(title)
    layout.addWidget(app.selected_pair_header_label)
    layout.addWidget(subtitle)
    frame.setLayout(layout)
    return frame


def create_info_card(title_text, value_text):
    frame = QFrame()
    frame.setObjectName("infoCard")
    frame.setMinimumHeight(88)
    frame.setMaximumHeight(104)

    layout = QVBoxLayout()
    layout.setContentsMargins(14, 10, 14, 10)
    layout.setSpacing(4)

    title = QLabel(title_text)
    title.setObjectName("cardTitle")

    value = QLabel(value_text)
    value.setObjectName("bigValue")

    frame.value_label = value
    frame.title_label = title

    layout.addWidget(title)
    layout.addWidget(value)
    frame.setLayout(layout)
    return frame


def create_prediction_card(model_name):
    frame = QFrame()
    frame.setObjectName("predictionCard")
    frame.setMinimumHeight(145)
    frame.setMaximumHeight(170)
    frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    layout = QVBoxLayout()
    layout.setContentsMargins(14, 10, 14, 10)
    layout.setSpacing(4)

    title = QLabel(model_name)
    title.setObjectName("modelTitle")

    return_label_title = QLabel("Predicted Return")
    return_label_title.setObjectName("smallLabel")

    return_value = QLabel("--")
    return_value.setObjectName("returnValue")

    price_label_title = QLabel("Predicted Price")
    price_label_title.setObjectName("smallLabel")

    price_value = QLabel("--")
    price_value.setObjectName("priceValue")

    params_label = QLabel("--")
    params_label.setObjectName("paramsLabel")
    params_label.setWordWrap(True)

    layout.addWidget(title)
    layout.addWidget(return_label_title)
    layout.addWidget(return_value)
    layout.addWidget(price_label_title)
    layout.addWidget(price_value)
    layout.addWidget(params_label)

    frame.setLayout(layout)

    return {
        "frame": frame,
        "return": return_value,
        "price": price_value,
        "params": params_label,
    }


def create_comparison_plot_card(title_text):
    frame = QFrame()
    frame.setObjectName("plotCard")

    layout = QVBoxLayout()
    layout.setContentsMargins(14, 14, 14, 14)
    layout.setSpacing(10)

    title = QLabel(title_text)
    title.setObjectName("cardTitle")

    image_label = QLabel("Plot will appear here after loading historical performance.")
    image_label.setObjectName("plotImageLabel")
    image_label.setAlignment(Qt.AlignCenter)
    image_label.setMinimumSize(520, 340)
    image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    layout.addWidget(title)
    layout.addWidget(image_label, alignment=Qt.AlignCenter)

    frame.setLayout(layout)

    return {
        "frame": frame,
        "image": image_label,
    }


def create_model_plot_card(title_text):
    frame = QFrame()
    frame.setObjectName("plotCard")

    layout = QVBoxLayout()
    layout.setContentsMargins(14, 14, 14, 14)
    layout.setSpacing(10)

    title = QLabel(title_text)
    title.setObjectName("cardTitle")

    image_label = QLabel("Run the prediction first to generate and display this plot.")
    image_label.setObjectName("plotImageLabel")
    image_label.setAlignment(Qt.AlignCenter)
    image_label.setMinimumSize(700, 470)
    image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    layout.addWidget(title)
    layout.addWidget(image_label)

    frame.setLayout(layout)

    return {
        "frame": frame,
        "image": image_label,
    }


def set_plot_image(label, image_path):
    if not os.path.exists(image_path):
        label.setText(f"Plot not found:\n{image_path}")
        label.setPixmap(QPixmap())
        return

    pixmap = QPixmap(image_path)
    if pixmap.isNull():
        label.setText(f"Could not load image:\n{image_path}")
        label.setPixmap(QPixmap())
        return

    scaled = pixmap.scaled(
        label.width() - 20,
        label.height() - 20,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation
    )
    label.setPixmap(scaled)
    label.setText("")