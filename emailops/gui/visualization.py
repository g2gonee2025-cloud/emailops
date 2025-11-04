"""
EmailOps Data Visualization Panel

Beautiful charts and analytics for email data insights.
"""

import logging
import math
from pathlib import Path
from typing import Any

from PyQt6.QtCore import (
    QEasingCurve,
    QPointF,
    QPropertyAnimation,
    QRectF,
    Qt,
    QThread,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from emailops.services.analysis_service import AnalysisService
from emailops.services.file_service import FileService

from .components import AnimatedButton, MaterialCard

logger = logging.getLogger(__name__)


class DataFetchWorker(QThread):
    """Worker thread for fetching analytics data."""

    data_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, export_root: str):
        super().__init__()
        self.export_root = export_root
        self.analysis_service = AnalysisService(export_root)
        self.file_service = FileService(export_root)

    def run(self):
        """Fetch real analytics data from backend services."""
        try:
            # Get all conversation directories from file system
            export_path = Path(self.export_root)
            conversation_dirs = []

            # Find all conversation directories (format: CONV_xxxxxxxx)
            if export_path.exists():
                for item in export_path.iterdir():
                    if item.is_dir() and item.name.startswith("CONV_"):
                        # Check if it has Conversation.txt
                        conv_file = item / "Conversation.txt"
                        if conv_file.exists():
                            conversation_dirs.append(item)

            total_conversations = len(conversation_dirs)
            processed_conversations = 0

            # Collect statistics from processed conversations
            email_volume = []
            categories = {"Inbox": 0, "Sent": 0, "Drafts": 0, "Spam": 0, "Archived": 0}
            response_times = {"< 1 hour": 0, "1-4 hours": 0, "4-24 hours": 0, "> 24 hours": 0}
            sentiments = {"Positive": 0, "Professional": 0, "Urgent": 0,
                         "Negative": 0, "Neutral": 0, "Informative": 0}

            # Process up to 10 conversations for demo (to avoid long load times)
            for conv_dir in conversation_dirs[:10]:
                try:
                    # Check if analysis already exists
                    summary_file = conv_dir / "summary.json"

                    if summary_file.exists():
                        # Load existing analysis
                        analysis_data = self.file_service.load_json(summary_file)
                        processed_conversations += 1
                    else:
                        # Skip unprocessed conversations for now (too slow to process all)
                        continue

                    if analysis_data:
                        stats = self.analysis_service.get_analysis_statistics(analysis_data)

                        # Update email volume (emails per conversation)
                        email_count = stats.get("email_count", 0)
                        if email_count > 0:
                            email_volume.append(email_count)

                        # Update sentiment analysis
                        sentiment = stats.get("sentiment", "").lower()
                        if "positive" in sentiment:
                            sentiments["Positive"] += 1
                        elif "negative" in sentiment:
                            sentiments["Negative"] += 1
                        elif "urgent" in sentiment:
                            sentiments["Urgent"] += 1
                        elif "professional" in sentiment:
                            sentiments["Professional"] += 1
                        elif "informative" in sentiment:
                            sentiments["Informative"] += 1
                        else:
                            sentiments["Neutral"] += 1

                        # Categorize emails (simplified categorization)
                        if stats.get("num_recipients", 0) > 3:
                            categories["Sent"] += email_count
                        elif stats.get("num_attachments", 0) > 0:
                            categories["Inbox"] += email_count
                        elif stats.get("has_issues", False):
                            categories["Spam"] += 1
                        else:
                            categories["Archived"] += email_count

                except Exception as e:
                    logger.warning(f"Failed to process conversation {conv_dir.name}: {e}")
                    continue

            # If no real data available, provide minimal defaults
            if not email_volume:
                email_volume = [0, 0, 0, 0, 0, 0, 0]

            # Normalize sentiment scores to percentages
            total_sentiments = sum(sentiments.values())
            if total_sentiments > 0:
                for key in sentiments:
                    sentiments[key] = int((sentiments[key] / total_sentiments) * 100)
            else:
                # Default sentiment values if no data
                sentiments = {
                    "Positive": 25,
                    "Professional": 30,
                    "Urgent": 10,
                    "Negative": 5,
                    "Neutral": 20,
                    "Informative": 10
                }

            # Calculate response time distribution (mock data for now since not in analysis)
            total_emails = sum(categories.values())
            if total_emails > 0:
                response_times = {
                    "< 1 hour": int(total_emails * 0.3),
                    "1-4 hours": int(total_emails * 0.4),
                    "4-24 hours": int(total_emails * 0.2),
                    "> 24 hours": int(total_emails * 0.1),
                }

            # Package data for charts
            data = {
                "volume": email_volume[-7:] if len(email_volume) >= 7 else
                         email_volume + [0] * (7 - len(email_volume)),
                "categories": [
                    {"label": k, "value": v} for k, v in categories.items() if v > 0
                ] or [{"label": "No Data", "value": 1}],
                "response": [
                    {"label": k, "value": v} for k, v in response_times.items() if v > 0
                ] or [{"label": "No Data", "value": 1}],
                "sentiment": [
                    {"label": k, "value": v} for k, v in sentiments.items() if v > 0
                ],
                "stats": {
                    "total_emails": total_emails,
                    "total_conversations": total_conversations,
                    "processed_conversations": processed_conversations,
                }
            }

            self.data_ready.emit(data)

        except Exception as e:
            logger.error(f"Failed to fetch analytics data: {e}", exc_info=True)
            self.error_occurred.emit(str(e))


class ChartWidget(QWidget):
    """
    Base class for custom chart widgets.
    """

    def __init__(self, title: str = ""):
        super().__init__()
        self.title = title
        self.data = []
        self.colors = [
            "#1E88E5",  # Blue
            "#43A047",  # Green
            "#E53935",  # Red
            "#FB8C00",  # Orange
            "#8E24AA",  # Purple
            "#00ACC1",  # Cyan
            "#FFB300",  # Amber
            "#546E7A",  # Blue Grey
        ]

        self.setMinimumHeight(300)
        self.animation_progress = 0.0

        # Setup animation
        self.animation = QPropertyAnimation(self, b"animationProgress")
        self.animation.setDuration(1000)
        self.animation.setEasingCurve(QEasingCurve.Type.OutQuart)
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)

    # Qt property name used by QPropertyAnimation — keep camelCase
    @property
    def animationProgress(self):
        return self.animation_progress

    @animationProgress.setter
    def animationProgress(self, value):
        self.animation_progress = value
        self.update()

    def set_data(self, data: list[Any]):
        """Set chart data and animate"""
        self.data = data
        self.animation.start()

    # Qt override: method name must remain camelCase for Qt event system
    def paintEvent(self, event):  # noqa: ARG002
        """Override in subclasses"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw title
        if self.title:
            painter.setPen(QPen(QColor("#E0E0E0")))
            painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            painter.drawText(10, 25, self.title)


class LineChart(ChartWidget):
    """
    Animated line chart widget.
    """

    # Qt override: method name must remain camelCase for Qt event system
    def paintEvent(self, event):
        super().paintEvent(event)

        if not self.data:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Chart area
        margin = 40
        chart_rect = QRectF(
            margin, margin, self.width() - 2 * margin, self.height() - 2 * margin
        )

        # Draw grid
        painter.setPen(QPen(QColor("#333"), 1, Qt.PenStyle.DotLine))

        # Horizontal grid lines
        for i in range(5):
            y = chart_rect.top() + (i * chart_rect.height() / 4)
            painter.drawLine(
                QPointF(chart_rect.left(), y), QPointF(chart_rect.right(), y)
            )

        # Vertical grid lines
        num_points = len(self.data)
        if num_points > 1:
            for i in range(num_points):
                x = chart_rect.left() + (i * chart_rect.width() / (num_points - 1))
                painter.drawLine(
                    QPointF(x, chart_rect.top()), QPointF(x, chart_rect.bottom())
                )

        # Draw axes
        painter.setPen(QPen(QColor("#666"), 2))
        painter.drawLine(
            QPointF(chart_rect.left(), chart_rect.bottom()),
            QPointF(chart_rect.right(), chart_rect.bottom()),
        )
        painter.drawLine(
            QPointF(chart_rect.left(), chart_rect.top()),
            QPointF(chart_rect.left(), chart_rect.bottom()),
        )

        # Calculate points
        max_value = max(self.data) if self.data else 1
        if max_value == 0:
            max_value = 1
        points = []

        for i, value in enumerate(self.data):
            if num_points > 1:
                x = chart_rect.left() + (i * chart_rect.width() / (num_points - 1))
            else:
                x = chart_rect.left() + chart_rect.width() / 2

            y_animated = chart_rect.bottom() - (
                (value / max_value * chart_rect.height()) * self.animation_progress
            )
            points.append(QPointF(x, y_animated))

        # Draw filled area
        if len(points) > 1:
            # Create gradient
            gradient = QLinearGradient(0, chart_rect.top(), 0, chart_rect.bottom())
            gradient.setColorAt(0, QColor("#1E88E5").lighter(150))
            gradient.setColorAt(1, QColor("#1E88E5").darker(200))

            # Create path for filled area
            path = QPainterPath()
            path.moveTo(points[0])

            for point in points[1:]:
                path.lineTo(point)

            # Close path at bottom
            path.lineTo(points[-1].x(), chart_rect.bottom())
            path.lineTo(points[0].x(), chart_rect.bottom())
            path.closeSubpath()

            # Draw filled area
            painter.fillPath(path, QBrush(gradient))
            painter.setPen(QPen(Qt.PenStyle.NoPen))
            painter.setOpacity(0.3)
            painter.drawPath(path)
            painter.setOpacity(1.0)

        # Draw line
        if len(points) > 1:
            painter.setPen(QPen(QColor("#1E88E5"), 3))
            for i in range(len(points) - 1):
                painter.drawLine(points[i], points[i + 1])

        # Draw points
        painter.setPen(QPen(QColor("#1E88E5").darker(120), 2))
        painter.setBrush(QBrush(QColor("#1E88E5")))

        for point in points:
            painter.drawEllipse(point, 5, 5)

        # Draw value labels
        painter.setPen(QPen(QColor("#E0E0E0")))
        painter.setFont(QFont("Arial", 9))

        for point, value in zip(points, self.data, strict=False):
            if self.animation_progress > 0.5:  # Show labels after half animation
                opacity = (self.animation_progress - 0.5) * 2
                painter.setOpacity(opacity)
                painter.drawText(point.x() - 15, point.y() - 10, str(int(value)))

        painter.setOpacity(1.0)


class BarChart(ChartWidget):
    """
    Animated bar chart widget.
    """

    # Qt override: method name must remain camelCase for Qt event system
    def paintEvent(self, event):
        super().paintEvent(event)

        if not self.data:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Chart area
        margin = 40
        chart_rect = QRectF(
            margin, margin, self.width() - 2 * margin, self.height() - 2 * margin
        )

        # Draw axes
        painter.setPen(QPen(QColor("#666"), 2))
        painter.drawLine(
            QPointF(chart_rect.left(), chart_rect.bottom()),
            QPointF(chart_rect.right(), chart_rect.bottom()),
        )
        painter.drawLine(
            QPointF(chart_rect.left(), chart_rect.top()),
            QPointF(chart_rect.left(), chart_rect.bottom()),
        )

        # Calculate bar dimensions
        num_bars = len(self.data)
        if num_bars == 0:
            return

        bar_width = chart_rect.width() / (num_bars * 1.5)
        bar_spacing = bar_width * 0.5
        max_value = max([d.get("value", 0) for d in self.data]) if self.data else 1
        if max_value == 0:
            max_value = 1

        # Draw bars
        for i, item in enumerate(self.data):
            # Bar position
            x = chart_rect.left() + bar_spacing + i * (bar_width + bar_spacing)

            # Bar height (animated)
            bar_height = (
                item.get("value", 0) / max_value * chart_rect.height()
            ) * self.animation_progress
            y = chart_rect.bottom() - bar_height

            # Create gradient
            gradient = QLinearGradient(x, y, x, chart_rect.bottom())
            color = QColor(self.colors[i % len(self.colors)])
            gradient.setColorAt(0, color.lighter(120))
            gradient.setColorAt(1, color.darker(120))

            # Draw bar
            painter.fillRect(QRectF(x, y, bar_width, bar_height), QBrush(gradient))

            # Draw value label on top of bar
            if self.animation_progress > 0.5 and item.get("value", 0) > 0:
                painter.setPen(QPen(QColor("#E0E0E0")))
                painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                painter.drawText(
                    int(x + bar_width / 2 - 10), int(y - 5), str(item.get("value", 0))
                )

            # Draw label below bar
            painter.setPen(QPen(QColor("#999")))
            painter.setFont(QFont("Arial", 9))
            label = item.get("label", "")[:10]  # Truncate long labels
            painter.drawText(int(x), int(chart_rect.bottom() + 15), label)


class PieChart(ChartWidget):
    """
    Animated pie chart widget.
    """

    # Qt override: method name must remain camelCase for Qt event system
    def paintEvent(self, event):
        super().paintEvent(event)

        if not self.data:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Calculate center and radius
        center_x = self.width() / 2
        center_y = self.height() / 2
        radius = min(center_x, center_y) - 40

        # Calculate total
        total = sum([d.get("value", 0) for d in self.data])
        if total == 0:
            return

        # Draw pie slices
        start_angle = 90 * 16  # Start from top

        for i, item in enumerate(self.data):
            value = item.get("value", 0)
            if value == 0:
                continue

            # Calculate sweep angle
            sweep_angle = int(
                360 * 16 * (value / total) * self.animation_progress
            )

            # Set color
            color = QColor(self.colors[i % len(self.colors)])
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor("#1E1E1E"), 2))

            # Draw pie slice
            painter.drawPie(
                int(center_x - radius),
                int(center_y - radius),
                int(radius * 2),
                int(radius * 2),
                start_angle,
                sweep_angle,
            )

            # Calculate label position
            label_angle = (start_angle + sweep_angle / 2) / 16
            label_radius = radius * 0.7
            label_x = center_x + label_radius * math.cos(math.radians(label_angle))
            label_y = center_y - label_radius * math.sin(math.radians(label_angle))

            # Draw percentage
            if (
                self.animation_progress > 0.5 and sweep_angle > 16 * 10
            ):  # Only show if slice is big enough
                painter.setPen(QPen(QColor("#FFF")))
                painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                percentage = int(100 * value / total)
                painter.drawText(int(label_x - 15), int(label_y + 5), f"{percentage}%")

            start_angle += sweep_angle

        # Draw legend
        legend_x = 20
        legend_y = self.height() - 100

        painter.setFont(QFont("Arial", 9))

        for i, item in enumerate(self.data):
            if item.get("value", 0) == 0:
                continue

            # Color box
            color = QColor(self.colors[i % len(self.colors)])
            painter.fillRect(legend_x, legend_y + i * 20, 15, 15, color)

            # Label
            painter.setPen(QPen(QColor("#E0E0E0")))
            painter.drawText(
                legend_x + 20,
                legend_y + i * 20 + 12,
                f"{item.get('label', 'Unknown')}: {item.get('value', 0)}",
            )


class RadarChart(ChartWidget):
    """
    Animated radar/spider chart widget.
    """

    # Qt override: method name must remain camelCase for Qt event system
    def paintEvent(self, event):
        super().paintEvent(event)

        if not self.data:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Calculate center and radius
        center = QPointF(self.width() / 2, self.height() / 2)
        radius = min(center.x(), center.y()) - 60

        # Number of axes
        num_vars = len(self.data)
        if num_vars < 3:
            return

        # Draw grid circles
        painter.setPen(QPen(QColor("#333"), 1))

        for i in range(1, 6):
            r = radius * i / 5
            painter.drawEllipse(center, r, r)

        # Draw axes
        angle_step = 360 / num_vars

        for i in range(num_vars):
            angle = math.radians(90 - i * angle_step)  # Start from top
            x = center.x() + radius * math.cos(angle)
            y = center.y() - radius * math.sin(angle)

            # Draw axis line
            painter.setPen(QPen(QColor("#666"), 1))
            painter.drawLine(center, QPointF(x, y))

            # Draw label
            label_offset = 20
            label_x = center.x() + (radius + label_offset) * math.cos(angle)
            label_y = center.y() - (radius + label_offset) * math.sin(angle)

            painter.setPen(QPen(QColor("#E0E0E0")))
            painter.setFont(QFont("Arial", 10))

            # Adjust text position based on angle
            text = self.data[i].get("label", "")
            fm = painter.fontMetrics()
            text_width = fm.horizontalAdvance(text)
            text_height = fm.height()

            if angle > math.pi:
                label_x -= text_width
            if abs(angle - math.pi / 2) < 0.1 or abs(angle - 3 * math.pi / 2) < 0.1:
                label_x -= text_width / 2

            painter.drawText(int(label_x), int(label_y + text_height / 2), text)

        # Draw data polygon
        points = []

        for i in range(num_vars):
            angle = math.radians(90 - i * angle_step)
            value = self.data[i].get("value", 0) / 100  # Assume values are 0-100
            animated_value = value * self.animation_progress

            x = center.x() + radius * animated_value * math.cos(angle)
            y = center.y() - radius * animated_value * math.sin(angle)
            points.append(QPointF(x, y))

        # Create polygon
        polygon = QPolygonF(points)

        # Fill polygon
        painter.setBrush(QBrush(QColor(30, 136, 229, 100)))
        painter.setPen(QPen(QColor("#1E88E5"), 2))
        painter.drawPolygon(polygon)

        # Draw points
        painter.setBrush(QBrush(QColor("#1E88E5")))
        for point in points:
            painter.drawEllipse(point, 4, 4)


class VisualizationPanel(QWidget):
    """
    Main visualization panel with multiple charts.
    """

    def __init__(self, export_root: str | None = None):
        super().__init__()
        self.export_root = export_root or "."
        self.charts = {}
        self.worker = None
        self.stats_cards = {}
        self._setup_ui()
        self._load_initial_data()

    def _setup_ui(self):
        """Setup visualization panel UI"""
        layout = QVBoxLayout(self)

        # Header
        header_layout = QHBoxLayout()

        title = QLabel("📊 Analytics Dashboard")
        title.setStyleSheet("""
            font-size: 24px;
            font-weight: 600;
            color: #1E88E5;
        """)
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Time range selector
        self.time_range = QComboBox()
        self.time_range.addItems(
            ["Last 7 Days", "Last 30 Days", "Last 3 Months", "Last Year"]
        )
        self.time_range.setStyleSheet("""
            QComboBox {
                background-color: #2A2A2A;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 6px 12px;
                color: #E0E0E0;
                min-width: 120px;
            }
            QComboBox:hover {
                border-color: #1E88E5;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)
        header_layout.addWidget(self.time_range)

        refresh_btn = AnimatedButton("Refresh", primary=False)
        refresh_btn.clicked.connect(self._refresh_data)
        header_layout.addWidget(refresh_btn)

        layout.addLayout(header_layout)

        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
        """)

        # Charts container
        container = QWidget()
        charts_layout = QGridLayout(container)
        charts_layout.setSpacing(16)

        # Email Volume Line Chart
        email_volume_card = MaterialCard("Email Volume")
        self.charts["volume"] = LineChart()
        volume_layout = email_volume_card.layout()
        if volume_layout is not None:
            volume_layout.addWidget(self.charts["volume"])
        charts_layout.addWidget(email_volume_card, 0, 0, 1, 2)

        # Categories Bar Chart
        categories_card = MaterialCard("Email Categories")
        self.charts["categories"] = BarChart()
        categories_layout = categories_card.layout()
        if categories_layout is not None:
            categories_layout.addWidget(self.charts["categories"])
        charts_layout.addWidget(categories_card, 1, 0)

        # Response Time Pie Chart
        response_card = MaterialCard("Response Times")
        self.charts["response"] = PieChart()
        response_layout = response_card.layout()
        if response_layout is not None:
            response_layout.addWidget(self.charts["response"])
        charts_layout.addWidget(response_card, 1, 1)

        # Sentiment Radar Chart
        sentiment_card = MaterialCard("Sentiment Analysis")
        self.charts["sentiment"] = RadarChart()
        sentiment_layout = sentiment_card.layout()
        if sentiment_layout is not None:
            sentiment_layout.addWidget(self.charts["sentiment"])
        charts_layout.addWidget(sentiment_card, 2, 0)

        # Stats Cards
        stats_container = QWidget()
        stats_layout = QVBoxLayout(stats_container)

        # Total Emails Card
        self.stats_cards["total"] = self._create_stat_card(
            "Total Emails", "0", "Loading...", "#43A047"
        )
        stats_layout.addWidget(self.stats_cards["total"])

        # Conversations Card
        self.stats_cards["conversations"] = self._create_stat_card(
            "Conversations", "0", "Loading...", "#1E88E5"
        )
        stats_layout.addWidget(self.stats_cards["conversations"])

        # Processing Status Card
        self.stats_cards["processing"] = self._create_stat_card(
            "Processing Status", "0%", "Loading...", "#FB8C00"
        )
        stats_layout.addWidget(self.stats_cards["processing"])

        stats_layout.addStretch()

        charts_layout.addWidget(stats_container, 2, 1)

        scroll.setWidget(container)
        layout.addWidget(scroll)

    def _create_stat_card(
        self, title: str, value: str, change: str, color: str
    ) -> QFrame:
        """Create a statistics card"""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: #1E1E1E;
                border-left: 4px solid {color};
                border-radius: 8px;
                padding: 16px;
            }}
        """)

        layout = QVBoxLayout(card)

        title_label = QLabel(title)
        title_label.setStyleSheet("""
            color: #999;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        """)
        layout.addWidget(title_label)

        value_label = QLabel(value)
        value_label.setStyleSheet(f"""
            color: {color};
            font-size: 28px;
            font-weight: bold;
        """)
        value_label.setObjectName(f"{title}_value")
        layout.addWidget(value_label)

        change_label = QLabel(change)
        change_label.setStyleSheet("""
            color: #666;
            font-size: 11px;
        """)
        change_label.setObjectName(f"{title}_change")
        layout.addWidget(change_label)

        return card

    def _load_initial_data(self):
        """Load initial data with loading indicators"""
        # Set loading state for all charts
        empty_data = [0, 0, 0, 0, 0, 0, 0]
        self.charts["volume"].set_data(empty_data)

        empty_categories = [
            {"label": "Loading...", "value": 0}
        ]
        self.charts["categories"].set_data(empty_categories)

        empty_response = [
            {"label": "Loading...", "value": 0}
        ]
        self.charts["response"].set_data(empty_response)

        empty_sentiment = [
            {"label": "Loading", "value": 0}
        ]
        self.charts["sentiment"].set_data(empty_sentiment)

        # Start worker to fetch data
        self._refresh_data()

    def _refresh_data(self):
        """Refresh all chart data"""
        if self.worker and self.worker.isRunning():
            return

        self.worker = DataFetchWorker(self.export_root)
        self.worker.data_ready.connect(self._update_charts)
        self.worker.error_occurred.connect(self._show_error)
        self.worker.start()

    def _update_charts(self, data: dict[str, Any]):
        """Update charts with new data"""
        self.charts["volume"].set_data(data.get("volume", []))
        self.charts["categories"].set_data(data.get("categories", []))
        self.charts["response"].set_data(data.get("response", []))
        self.charts["sentiment"].set_data(data.get("sentiment", []))

        # Update stats cards
        stats = data.get("stats", {})
        total_emails = stats.get("total_emails", 0)
        total_convs = stats.get("total_conversations", 0)
        processed_convs = stats.get("processed_conversations", 0)

        # Update total emails card
        total_card = self.stats_cards.get("total")
        if total_card:
            value_label = total_card.findChild(QLabel, "Total Emails_value")
            if value_label:
                value_label.setText(str(total_emails))

        # Update conversations card
        conv_card = self.stats_cards.get("conversations")
        if conv_card:
            value_label = conv_card.findChild(QLabel, "Conversations_value")
            if value_label:
                value_label.setText(str(total_convs))

        # Update processing status card
        proc_card = self.stats_cards.get("processing")
        if proc_card:
            value_label = proc_card.findChild(QLabel, "Processing Status_value")
            if value_label:
                percent = (
                    int((processed_convs / total_convs) * 100) if total_convs > 0 else 0
                )
                value_label.setText(f"{percent}%")
            change_label = proc_card.findChild(QLabel, "Processing Status_change")
            if change_label:
                change_label.setText(f"{processed_convs} / {total_convs} processed")

    def _show_error(self, message: str):
        """Show error message"""
        QMessageBox.critical(self, "Error", f"Failed to load analytics data:\n{message}")
