import sys

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTabWidget

from config import Config
from data_loader import BTCDataLoader
from features import FeatureEngineer
from evaluation import WalkForwardEvaluator
from predictor import TomorrowPredictor
from plotting import Plotter

from gui_styles import get_stylesheet
from gui_prediction_tab import build_prediction_tab, run_prediction_workflow
from gui_performance_tab import build_performance_tab, load_performance_into_tab
from gui_model_tabs import build_model_tab, load_model_tab_images
from gui_helpers import build_header


class BTCApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Crypto Prediction Dashboard")
        self.setGeometry(90, 50, 1820, 1080)

        self.cfg = Config()

        self.loader = BTCDataLoader(self.cfg)
        self.fe = FeatureEngineer(self.cfg)
        self.evaluator = WalkForwardEvaluator(self.cfg)
        self.predictor = TomorrowPredictor(self.cfg)
        self.plotter = Plotter()

        self.dataset = None
        self.results = None

        self.model_plot_tabs = {}

        self.init_ui()
        self.setStyleSheet(get_stylesheet())

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(14)

        header = build_header(self)
        main_layout.addWidget(header)

        self.tabs = QTabWidget()

        self.tab_predict = QWidget()
        self.tab_performance_comparison = QWidget()
        self.tab_ols = QWidget()
        self.tab_ridge = QWidget()
        self.tab_lasso = QWidget()
        self.tab_kernel_ridge = QWidget()
        self.tab_knn = QWidget()
        self.tab_nn = QWidget()

        self.tabs.addTab(self.tab_predict, "Tomorrow Prediction")
        self.tabs.addTab(self.tab_performance_comparison, "Historical Performance Comparison")
        self.tabs.addTab(self.tab_ols, "OLS")
        self.tabs.addTab(self.tab_ridge, "Ridge")
        self.tabs.addTab(self.tab_lasso, "Lasso")
        self.tabs.addTab(self.tab_kernel_ridge, "Kernel Ridge")
        self.tabs.addTab(self.tab_knn, "KNN")
        self.tabs.addTab(self.tab_nn, "NeuralNet")

        build_prediction_tab(self, self.tab_predict)
        build_performance_tab(self, self.tab_performance_comparison)

        build_model_tab(
            self,
            self.tab_ols,
            model_key="OLS",
            heading="Ordinary Least Squares",
            description="Ordinary Least Squares is a type of regression with no regularization."
        )
        build_model_tab(
            self,
            self.tab_ridge,
            model_key="Ridge",
            heading="Ridge Regression",
            description="Ridge Regression is a linear regression model that uses L2 regularization."
        )
        build_model_tab(
            self,
            self.tab_lasso,
            model_key="Lasso",
            heading="Lasso Regression",
            description="Lasso Regression is a linear regression model that uses L1 regularization."
        )
        build_model_tab(
            self,
            self.tab_kernel_ridge,
            model_key="KernelRidge",
            heading="Kernel Ridge Regression",
            description="Kernel Ridge Regression extends ridge regression by using a kernel function to model non-linear relationships. It uses an RBF kernel, so predictions are based on the similarity between observations rather than only a straight-line fit in the original feature space."
        )
        build_model_tab(
            self,
            self.tab_knn,
            model_key="KNN",
            heading="K-Nearest Neighbours",
            description="K-Nearest Neighbours predicts values using the closest historical observations."
        )
        build_model_tab(
            self,
            self.tab_nn,
            model_key="NeuralNet",
            heading="Feedforward Neural Network",
            description="A feedforward neural network is a simple neural network where information moves in one direction from input to output. This model uses a small network with one hidden layer to learn patterns in the data."
        )

        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def run_prediction(self):
        success = run_prediction_workflow(self)
        if success:
            load_model_tab_images(self)

    def load_performance(self):
        load_performance_into_tab(self)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BTCApp()
    window.show()
    sys.exit(app.exec_())