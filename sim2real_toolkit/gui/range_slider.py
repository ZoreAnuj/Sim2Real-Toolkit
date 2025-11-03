"""Custom range slider with 3 handles (min, preview, max)"""

from PySide6.QtWidgets import QWidget, QStyleOptionSlider, QStyle
from PySide6.QtCore import Qt, Signal, QRect, QPoint
from PySide6.QtGui import QPainter, QColor, QPen


class TripleRangeSlider(QWidget):
    """Slider with 3 handles: min, preview (center), max"""
    
    valueChanged = Signal()
    
    def __init__(self, minimum=0, maximum=100, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(40)
        
        self._minimum = minimum
        self._maximum = maximum
        
        # Three values: min, preview, max
        self._min_value = minimum
        self._preview_value = (minimum + maximum) // 2
        self._max_value = maximum
        
        # Which handle is being dragged
        self._active_handle = None  # 'min', 'preview', 'max', or None
        
        self.setMouseTracking(True)
    
    def minimum(self):
        return self._minimum
    
    def maximum(self):
        return self._maximum
    
    def setMinimum(self, value):
        self._minimum = value
        self.update()
    
    def setMaximum(self, value):
        self._maximum = value
        self.update()
    
    def minValue(self):
        return self._min_value
    
    def previewValue(self):
        return self._preview_value
    
    def maxValue(self):
        return self._max_value
    
    def setMinValue(self, value):
        self._min_value = max(self._minimum, min(value, self._preview_value))
        self.update()
    
    def setPreviewValue(self, value):
        self._preview_value = max(self._min_value, min(value, self._max_value))
        self.update()
    
    def setMaxValue(self, value):
        self._max_value = max(self._preview_value, min(value, self._maximum))
        self.update()
    
    def setValues(self, min_val, preview_val, max_val):
        """Set all three values at once"""
        self._min_value = max(self._minimum, min(min_val, self._maximum))
        self._max_value = max(self._minimum, min(max_val, self._maximum))
        self._preview_value = max(self._min_value, min(preview_val, self._max_value))
        self.update()
    
    def _value_to_pos(self, value):
        """Convert value to x position"""
        margin = 20
        width = self.width() - 2 * margin
        ratio = (value - self._minimum) / (self._maximum - self._minimum) if self._maximum > self._minimum else 0
        return int(margin + ratio * width)
    
    def _pos_to_value(self, x):
        """Convert x position to value"""
        margin = 20
        width = self.width() - 2 * margin
        ratio = max(0, min(1, (x - margin) / width))
        return int(self._minimum + ratio * (self._maximum - self._minimum))
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        margin = 20
        track_y = self.height() // 2
        track_width = self.width() - 2 * margin
        
        # Draw track background
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(200, 200, 200))
        painter.drawRoundedRect(margin, track_y - 2, track_width, 4, 2, 2)
        
        # Draw active range (min to max)
        min_x = self._value_to_pos(self._min_value)
        max_x = self._value_to_pos(self._max_value)
        painter.setBrush(QColor(100, 150, 255, 100))
        painter.drawRect(min_x, track_y - 4, max_x - min_x, 8)
        
        # Draw preview line
        preview_x = self._value_to_pos(self._preview_value)
        painter.setPen(QPen(QColor(0, 120, 215), 2))
        painter.drawLine(preview_x, track_y - 8, preview_x, track_y + 8)
        
        # Draw handles
        handle_radius = 6
        
        # Min handle (blue)
        painter.setPen(QPen(QColor(0, 100, 200), 2))
        painter.setBrush(QColor(100, 150, 255))
        painter.drawEllipse(QPoint(min_x, track_y), handle_radius, handle_radius)
        
        # Max handle (blue)
        painter.setPen(QPen(QColor(0, 100, 200), 2))
        painter.setBrush(QColor(100, 150, 255))
        painter.drawEllipse(QPoint(max_x, track_y), handle_radius, handle_radius)
        
        # Preview handle (green, larger)
        painter.setPen(QPen(QColor(0, 150, 0), 2))
        painter.setBrush(QColor(76, 175, 80))
        painter.drawEllipse(QPoint(preview_x, track_y), handle_radius + 2, handle_radius + 2)
    
    def _get_handle_at_pos(self, x, y):
        """Get which handle is at position (x, y)"""
        track_y = self.height() // 2
        handle_radius = 10  # Click tolerance
        
        # Check preview first (it's on top)
        preview_x = self._value_to_pos(self._preview_value)
        if abs(x - preview_x) <= handle_radius and abs(y - track_y) <= handle_radius:
            return 'preview'
        
        # Check min
        min_x = self._value_to_pos(self._min_value)
        if abs(x - min_x) <= handle_radius and abs(y - track_y) <= handle_radius:
            return 'min'
        
        # Check max
        max_x = self._value_to_pos(self._max_value)
        if abs(x - max_x) <= handle_radius and abs(y - track_y) <= handle_radius:
            return 'max'
        
        return None
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._active_handle = self._get_handle_at_pos(event.position().x(), event.position().y())
            if self._active_handle:
                event.accept()
    
    def mouseMoveEvent(self, event):
        if self._active_handle:
            value = self._pos_to_value(event.position().x())
            
            if self._active_handle == 'min':
                self.setMinValue(value)
            elif self._active_handle == 'preview':
                self.setPreviewValue(value)
            elif self._active_handle == 'max':
                self.setMaxValue(value)
            
            self.valueChanged.emit()
            event.accept()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._active_handle = None
            event.accept()

