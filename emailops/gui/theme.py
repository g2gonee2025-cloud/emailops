"""
Theme Manager for EmailOps GUI
"""

from PyQt6.QtGui import QColor


class Theme:
    """
    A class to hold the theme colors.
    """
    def __init__(self):
        self.primary = "#1E88E5"
        self.primary_dark = "#1565C0"
        self.primary_light = "#42A5F5"
        self.accent = "#00BCD4"
        self.background = "#121212"
        self.surface = "#1E1E1E"
        self.surface_elevated = "#242424"
        self.text_primary = "#E0E0E0"
        self.text_secondary = "#B0B0B0"
        self.text_disabled = "#606060"
        self.success = "#4CAF50"
        self.warning = "#FF9800"
        self.error = "#F44336"
        self.info = "#2196F3"
        self.card_bg = "#1E1E1E"
        self.card_hover = "#2A2A2A"
        self.border = "#333333"
        self.shadow = "rgba(0, 0, 0, 0.5)"

class ThemeManager:
    """
    A class to manage the application's theme.
    """
    def __init__(self):
        self.theme = Theme()

    def get_color(self, name: str) -> QColor:
        """
        Get a color from the theme by name.
        """
        return QColor(getattr(self.theme, name, "#000000"))

    def get_stylesheet(self, widget_name: str) -> str:
        """
        Get a stylesheet for a specific widget.
        """
        if widget_name == "AnimatedButton":
            return f"""
                QPushButton {{
                    background-color: {self.theme.primary};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 10px 24px;
                    font-weight: 600;
                    font-size: 14px;
                    text-transform: uppercase;
                }}
                QPushButton:hover {{
                    background-color: {self.theme.primary_light};
                }}
                QPushButton:pressed {{
                    background-color: {self.theme.primary_dark};
                }}
                QPushButton:disabled {{
                    background-color: {self.theme.surface_elevated};
                    color: {self.theme.text_disabled};
                }}
            """
        elif widget_name == "MaterialCard":
            return f"""
                #MaterialCard {{
                    background-color: {self.theme.card_bg};
                    border: 1px solid {self.theme.border};
                    border-radius: 8px;
                    padding: 16px;
                }}
            """
        return ""
