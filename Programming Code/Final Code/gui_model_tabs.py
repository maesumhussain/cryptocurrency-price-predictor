import os

from PyQt5.QtWidgets import (
    QVBoxLayout, QLabel, QFrame, QGridLayout, QScrollArea
)

from gui_helpers import create_model_plot_card, set_plot_image


def build_model_tab(app, tab_widget, model_key, heading, description):
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)

    container = QFrame()
    layout = QVBoxLayout(container)
    layout.setSpacing(16)
    layout.setContentsMargins(0, 0, 0, 0)

    title = QLabel(heading)
    title.setObjectName("modelPageTitle")

    desc = QLabel(description)
    desc.setObjectName("modelPageDescription")
    desc.setWordWrap(True)

    layout.addWidget(title)
    layout.addWidget(desc)

    plots_wrapper = QFrame()
    plots_wrapper.setObjectName("sectionFrame")

    plots_grid = QGridLayout()
    plots_grid.setContentsMargins(14, 14, 14, 14)
    plots_grid.setSpacing(16)

    price_card = create_model_plot_card(f"{model_key} - True vs Predicted Prices")
    returns_card = create_model_plot_card(f"{model_key} - True vs Predicted Returns")

    plots_grid.addWidget(price_card["frame"], 0, 0)
    plots_grid.addWidget(returns_card["frame"], 0, 1)
    plots_grid.setColumnStretch(0, 1)
    plots_grid.setColumnStretch(1, 1)

    plots_wrapper.setLayout(plots_grid)
    layout.addWidget(plots_wrapper)

    scroll.setWidget(container)

    outer_layout = QVBoxLayout()
    outer_layout.addWidget(scroll)
    tab_widget.setLayout(outer_layout)

    app.model_plot_tabs[model_key] = {
        "price_label": price_card["image"],
        "returns_label": returns_card["image"],
    }


def load_model_tab_images(app):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plot_folder = os.path.join(base_dir, "plots")

    model_file_names = {
        "OLS": ("OLS_prices_scatter.png", "OLS_returns_scatter.png"),
        "Ridge": ("Ridge_prices_scatter.png", "Ridge_returns_scatter.png"),
        "Lasso": ("Lasso_prices_scatter.png", "Lasso_returns_scatter.png"),
        "KernelRidge": ("KernelRidge_prices_scatter.png", "KernelRidge_returns_scatter.png"),
        "KNN": ("KNN_prices_scatter.png", "KNN_returns_scatter.png"),
        "NeuralNet": ("NeuralNet_prices_scatter.png", "NeuralNet_returns_scatter.png"),
    }

    for model_key, labels in app.model_plot_tabs.items():
        price_file, return_file = model_file_names[model_key]
        price_path = os.path.join(plot_folder, price_file)
        return_path = os.path.join(plot_folder, return_file)

        set_plot_image(labels["price_label"], price_path)
        set_plot_image(labels["returns_label"], return_path)