import os

from PyQt5.QtWidgets import (
    QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QPushButton, QFrame, QGridLayout, QHeaderView,
    QScrollArea, QAbstractItemView, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from gui_helpers import create_comparison_plot_card, set_plot_image


def build_performance_tab(app, tab_widget):
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)

    container = QFrame()
    layout = QVBoxLayout(container)
    layout.setSpacing(14)
    layout.setContentsMargins(0, 0, 0, 0)

    heading = QLabel("Historical Model Performance Comparison")
    heading.setObjectName("sectionTitle")
    layout.addWidget(heading)

    app.table = QTableWidget()
    app.table.setObjectName("performanceTable")
    app.table.setColumnCount(6)
    app.table.setHorizontalHeaderLabels(
        [
            "Model",
            "Return MSE",
            "Price MSE",
            "Directional Accuracy (%)",
            "Combined Score",
            "Rank",
        ]
    )
    app.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    app.table.verticalHeader().setVisible(False)
    app.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    app.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    app.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    app.table.setSelectionMode(QAbstractItemView.NoSelection)
    app.table.setFocusPolicy(Qt.NoFocus)
    layout.addWidget(app.table)

    app.refresh_button = QPushButton("Load Historical Performance")
    app.refresh_button.clicked.connect(app.load_performance)
    app.refresh_button.setObjectName("secondaryButton")
    layout.addWidget(app.refresh_button)

    plots_heading = QLabel("Comparison Plots")
    plots_heading.setObjectName("sectionTitle")
    layout.addWidget(plots_heading)

    app.comparison_plots_frame = QFrame()
    app.comparison_plots_frame.setObjectName("sectionFrame")

    app.comparison_plots_grid = QGridLayout()
    app.comparison_plots_grid.setContentsMargins(14, 14, 14, 14)
    app.comparison_plots_grid.setSpacing(14)

    app.return_mse_plot_card = create_comparison_plot_card("Return MSE Comparison")
    app.price_mse_plot_card = create_comparison_plot_card("Price MSE Comparison")
    app.dir_acc_plot_card = create_comparison_plot_card("Directional Accuracy Comparison")

    app.comparison_plots_grid.addWidget(app.return_mse_plot_card["frame"], 0, 0)
    app.comparison_plots_grid.addWidget(app.price_mse_plot_card["frame"], 0, 1)
    app.comparison_plots_grid.addWidget(app.dir_acc_plot_card["frame"], 0, 2)

    app.comparison_plots_grid.setColumnStretch(0, 1)
    app.comparison_plots_grid.setColumnStretch(1, 1)
    app.comparison_plots_grid.setColumnStretch(2, 1)

    app.comparison_plots_frame.setLayout(app.comparison_plots_grid)
    layout.addWidget(app.comparison_plots_frame)

    scroll.setWidget(container)

    outer_layout = QVBoxLayout()
    outer_layout.addWidget(scroll)
    tab_widget.setLayout(outer_layout)


def highlight_best_rank(table):
    highlight = QColor("#14532d")

    for row in range(table.rowCount()):
        rank_item = table.item(row, 5)
        if rank_item and rank_item.text() == "1":
            for col in range(table.columnCount()):
                item = table.item(row, col)
                if item:
                    item.setBackground(highlight)


def load_comparison_plots(app):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plot_folder = os.path.join(base_dir, "plots")

    return_mse_path = os.path.join(plot_folder, "mse_return.png")
    price_mse_path = os.path.join(plot_folder, "mse_price.png")
    dir_acc_path = os.path.join(plot_folder, "directional_accuracy.png")

    set_plot_image(app.return_mse_plot_card["image"], return_mse_path)
    set_plot_image(app.price_mse_plot_card["image"], price_mse_path)
    set_plot_image(app.dir_acc_plot_card["image"], dir_acc_path)


def safe_min_max_normalize_desc(values):
    """
    For metrics where lower is better, e.g. MSE.
    Best value gets 1.0, worst gets 0.0.
    """
    min_val = min(values)
    max_val = max(values)

    if abs(max_val - min_val) < 1e-12:
        return [1.0 for _ in values]

    return [(max_val - v) / (max_val - min_val) for v in values]


def safe_min_max_normalize_asc(values):
    """
    For metrics where higher is better, e.g. directional accuracy.
    Best value gets 1.0, worst gets 0.0.
    """
    min_val = min(values)
    max_val = max(values)

    if abs(max_val - min_val) < 1e-12:
        return [1.0 for _ in values]

    return [(v - min_val) / (max_val - min_val) for v in values]


def build_ranked_rows(models, mse_return, mse_price, dir_acc):
    return_scores = safe_min_max_normalize_desc(mse_return)
    price_scores = safe_min_max_normalize_desc(mse_price)
    dir_scores = safe_min_max_normalize_asc(dir_acc)

    rows = []
    for i, model in enumerate(models):
        combined_score = (return_scores[i] + price_scores[i] + dir_scores[i]) / 3.0
        rows.append({
            "model": model,
            "return_mse": mse_return[i],
            "price_mse": mse_price[i],
            "dir_acc": dir_acc[i],
            "combined_score": combined_score,
        })

    rows.sort(key=lambda row: row["combined_score"], reverse=True)

    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return rows


def load_performance_into_tab(app):
    if app.results is None:
        QMessageBox.warning(app, "Error", "Please run the prediction first.")
        return

    models = app.results.avg_metrics_for_bar["model_names"]
    mse_return = app.results.avg_metrics_for_bar["mse_return"]
    mse_price = app.results.avg_metrics_for_bar["mse_price"]
    dir_acc = app.results.avg_metrics_for_bar["dir_acc"]

    ranked_rows = build_ranked_rows(models, mse_return, mse_price, dir_acc)

    app.table.setRowCount(len(ranked_rows))

    for i, row in enumerate(ranked_rows):
        app.table.setItem(i, 0, QTableWidgetItem(row["model"]))
        app.table.setItem(i, 1, QTableWidgetItem(f"{row['return_mse']:.6f}"))
        app.table.setItem(i, 2, QTableWidgetItem(f"{row['price_mse']:.2f}"))
        app.table.setItem(i, 3, QTableWidgetItem(f"{row['dir_acc'] * 100:.2f}%"))
        app.table.setItem(i, 4, QTableWidgetItem(f"{row['combined_score']:.4f}"))
        app.table.setItem(i, 5, QTableWidgetItem(str(row["rank"])))

    app.table.resizeRowsToContents()
    header_height = app.table.horizontalHeader().height()
    row_heights = sum(app.table.rowHeight(i) for i in range(app.table.rowCount()))
    frame_height = 6
    total_height = header_height + row_heights + frame_height
    app.table.setFixedHeight(total_height + 4)

    highlight_best_rank(app.table)
    load_comparison_plots(app)