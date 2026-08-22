def get_stylesheet():
    return """
        QWidget {
            background-color: #0f172a;
            color: #e5e7eb;
            font-family: Segoe UI, Arial, sans-serif;
            font-size: 14px;
        }

        QFrame#headerFrame {
            background-color: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 #1d4ed8, stop:1 #0f766e
            );
            border-radius: 16px;
        }

        QLabel#titleLabel {
            font-size: 24px;
            font-weight: 700;
            color: white;
            background: transparent;
        }

        QLabel#selectedPairHeaderLabel {
            font-size: 16px;
            font-weight: 600;
            color: #dbeafe;
            background: transparent;
        }

        QLabel#subtitleLabel {
            font-size: 13px;
            color: #dbeafe;
            background: transparent;
        }

        QTabWidget::pane {
            border: none;
            margin-top: 8px;
        }

        QTabBar::tab {
            background: #1e293b;
            color: #cbd5e1;
            padding: 10px 10px;
            margin-right: 4px;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
            min-width: 105px;
            font-size: 12px;
            font-weight: 600;
        }

        QTabBar::tab:selected {
            background: #2563eb;
            color: white;
        }

        QPushButton {
            border: none;
            border-radius: 10px;
            padding: 10px 14px;
            font-weight: 600;
        }

        QPushButton#primaryButton {
            background-color: #2563eb;
            color: white;
        }

        QPushButton#primaryButton:hover {
            background-color: #1d4ed8;
        }

        QPushButton#secondaryButton {
            background-color: #0f766e;
            color: white;
        }

        QPushButton#secondaryButton:hover {
            background-color: #0d9488;
        }

        QComboBox {
            background-color: #111827;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 8px 10px;
            color: #f8fafc;
            min-height: 20px;
        }

        QComboBox QAbstractItemView {
            background-color: #111827;
            color: #f8fafc;
            selection-background-color: #2563eb;
        }

        QSlider::groove:horizontal {
            border: none;
            height: 6px;
            background: #334155;
            border-radius: 3px;
        }

        QSlider::handle:horizontal {
            background: #2563eb;
            border: none;
            width: 16px;
            margin: -5px 0;
            border-radius: 8px;
        }

        QLabel#sliderValueLabel {
            font-size: 14px;
            font-weight: 700;
            color: #93c5fd;
        }

        QLabel#exchangeRateLabel {
            font-size: 14px;
            font-weight: 600;
            color: #cbd5e1;
            padding-left: 6px;
        }

        QLabel#statusLabel {
            background-color: #1e293b;
            border-radius: 10px;
            padding: 10px;
            font-weight: 700;
            color: #93c5fd;
        }

        QFrame#infoCard, QFrame#predictionCard, QFrame#sectionFrame, QFrame#plotCard {
            background-color: #111827;
            border: 1px solid #1f2937;
            border-radius: 16px;
        }

        QLabel#cardTitle {
            color: #93c5fd;
            font-size: 13px;
            font-weight: 600;
        }

        QLabel#bigValue {
            color: white;
            font-size: 24px;
            font-weight: 700;
        }

        QLabel#modelTitle {
            color: #f8fafc;
            font-size: 18px;
            font-weight: 700;
        }

        QLabel#smallLabel {
            color: #94a3b8;
            font-size: 11px;
            font-weight: 600;
        }

        QLabel#returnValue {
            color: #22c55e;
            font-size: 18px;
            font-weight: 700;
        }

        QLabel#priceValue {
            color: #fbbf24;
            font-size: 18px;
            font-weight: 700;
        }

        QLabel#paramsLabel {
            color: #94a3b8;
            font-size: 10px;
            font-weight: 500;
            padding-top: 2px;
        }

        QLabel#sectionTitle {
            font-size: 18px;
            font-weight: 700;
            color: #e2e8f0;
            padding: 4px 2px;
        }

        QLabel#sectionTitleSmall {
            font-size: 15px;
            font-weight: 700;
            color: #e2e8f0;
            padding: 0;
        }

        QLabel#subSectionLabel {
            font-size: 13px;
            color: #94a3b8;
            padding-bottom: 6px;
        }

        QLabel#modelPageTitle {
            font-size: 30px;
            font-weight: 700;
            color: #f8fafc;
            padding-top: 4px;
            padding-bottom: 6px;
        }

        QLabel#modelPageDescription {
            font-size: 17px;
            color: #cbd5e1;
            line-height: 1.4;
            padding-bottom: 10px;
        }

        QLabel#plotImageLabel {
            background-color: #0b1220;
            border: 1px solid #253046;
            border-radius: 12px;
            color: #94a3b8;
            padding: 10px;
        }

        QTableWidget#performanceTable {
            background-color: #111827;
            border: 1px solid #1f2937;
            border-radius: 14px;
            gridline-color: #334155;
            color: #e5e7eb;
        }

        QHeaderView::section {
            background-color: #1e293b;
            color: #f8fafc;
            padding: 10px;
            border: none;
            font-weight: 700;
        }

        QTableWidget::item {
            padding: 8px;
        }
    """