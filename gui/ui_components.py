from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    def __init__(self, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#1e1e1e')
        super().__init__(self.fig)
        self.setStyleSheet("background-color: #1e1e1e; border-radius: 8px;")


class GlassCard(QtWidgets.QFrame):
    """现代玻璃效果卡片组件"""
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setFrameStyle(QtWidgets.QFrame.NoFrame)
        self.setStyleSheet(
            """
            GlassCard {
                background-color: rgba(45, 45, 48, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
            }
            """
        )
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(20, 16, 20, 20)
        self.layout.setSpacing(12)
        if title:
            title_label = QtWidgets.QLabel(title)
            title_label.setStyleSheet(
                """
                color: #ffffff;
                font-size: 16px;
                font-weight: 600;
                margin-bottom: 8px;
                """
            )
            self.layout.addWidget(title_label)

    def addWidget(self, widget: QtWidgets.QWidget):
        self.layout.addWidget(widget)

    def addLayout(self, layout: QtWidgets.QLayout):
        self.layout.addLayout(layout)


class ModernButton(QtWidgets.QPushButton):
    """现代风格按钮"""
    def __init__(self, text, button_type: str = "primary", icon: QtGui.QIcon | None = None, parent=None):
        super().__init__(text, parent)
        if icon:
            self.setIcon(icon)
        styles = {
            "primary": """
                QPushButton { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #4A9EFF, stop:1 #2E7CE6); color: white; border: none; border-radius: 8px; padding: 12px 24px; font-size: 14px; font-weight: 600; min-width: 120px; }
                QPushButton:hover { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #5BADFF, stop:1 #3F8DF7); }
                QPushButton:pressed { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #396FD5, stop:1 #1D6BD5); }
                QPushButton:disabled { background: #555555; color: #999999; }
            """,
            "success": """
                QPushButton { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #52C41A, stop:1 #389E0D); color: white; border: none; border-radius: 8px; padding: 12px 24px; font-size: 14px; font-weight: 600; min-width: 120px; }
                QPushButton:hover { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #73D13D, stop:1 #52C41A); }
            """,
            "warning": """
                QPushButton { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #FA8C16, stop:1 #D46B08); color: white; border: none; border-radius: 8px; padding: 12px 24px; font-size: 14px; font-weight: 600; min-width: 120px; }
                QPushButton:hover { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #FFA940, stop:1 #FA8C16); }
            """,
            "ghost": """
                QPushButton { background: transparent; color: #ffffff; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px; padding: 12px 24px; font-size: 14px; font-weight: 500; min-width: 120px; }
                QPushButton:hover { background: rgba(255, 255, 255, 0.1); border-color: rgba(255, 255, 255, 0.5); }
            """,
        }
        self.setStyleSheet(styles.get(button_type, styles["primary"]))


class AnimatedProgressBar(QtWidgets.QProgressBar):
    """动画进度条"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(8)
        self.setTextVisible(False)
        self.setStyleSheet(
            """
            QProgressBar { border: none; border-radius: 4px; background-color: rgba(255,255,255,0.1); }
            QProgressBar::chunk { border-radius: 4px; background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #4A9EFF, stop:0.5 #2E7CE6, stop:1 #1890FF); }
            """
        )


class KPICard(QtWidgets.QFrame):
    """KPI指标卡片"""
    def __init__(self, title, value: str = "0", icon: QtGui.QIcon | None = None, color: str = "#4A9EFF", parent=None):
        super().__init__(parent)
        self.setFixedHeight(120)
        self.setStyleSheet(
            f"""
            KPICard {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 rgba(45,45,48,0.9), stop:1 rgba(30,30,30,0.9));
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                border-left: 4px solid {color};
            }}
            """
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(8)
        title_layout = QtWidgets.QHBoxLayout()
        if icon:
            icon_label = QtWidgets.QLabel()
            icon_label.setPixmap(icon.pixmap(24, 24))
            title_layout.addWidget(icon_label)
        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: 600; margin: 0;")
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        self.value_label = QtWidgets.QLabel(value)
        self.value_label.setStyleSheet("color: #ffffff; font-size: 28px; font-weight: 700; margin: 0;")
        self.value_label.setAlignment(Qt.AlignCenter)
        layout.addLayout(title_layout)
        layout.addWidget(self.value_label, 1, Qt.AlignCenter)

    def update_value(self, value):
        self.value_label.setText(str(value))


class ModernInput(QtWidgets.QWidget):
    """现代风格输入框包装器"""
    def __init__(self, label: str, widget: QtWidgets.QWidget, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        label_widget = QtWidgets.QLabel(label)
        label_widget.setStyleSheet("color: #ffffff; font-size: 13px; font-weight: 500; margin-bottom: 4px;")
        widget.setStyleSheet(
            """
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 6px;
                color: #ffffff;
                padding: 10px 12px;
                font-size: 14px;
                min-height: 16px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #4A9EFF; background-color: rgba(74, 158, 255, 0.1); }
            QComboBox::drop-down { border: none; width: 30px; }
            QComboBox::down-arrow { image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iOCIgdmlld0JveD0iMCAwIDEyIDgiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0xIDFMNiA2TDExIDEiIHN0cm9rZT0iI2ZmZmZmZiIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KPC9zdmc+); width: 12px; height: 8px; }
            """
        )
        layout.addWidget(label_widget)
        layout.addWidget(widget)
        self.widget = widget

    def value(self):
        if hasattr(self.widget, 'value'):
            return self.widget.value()
        if hasattr(self.widget, 'text'):
            return self.widget.text()
        if hasattr(self.widget, 'currentText'):
            return self.widget.currentText()
        return None
