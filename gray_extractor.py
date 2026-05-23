"""
Gray Extractor v4.0 — Multi-Channel + Illumination Correction + Depth Calibration
=============================================================================
New in v4.0:
  - 6-channel extraction (Gray, Redness, Greenness, Blueness, CIE L*, CIE a*)
  - Polynomial illumination gradient correction
  - Pixel-to-depth calibration via 2 known drill holes
  - All v3.0 features retained

Math:
  Redness  = R/(R+G+B)          — hematite / oxidation
  Greenness = G/(R+G+B)         — chlorite
  Blueness  = B/(R+G+B)         — carbonate
  CIE L*   = luminance (perceptually uniform)
  CIE a*   = red-green axis (most geologically useful)

  Illumination correction:
    trend(y) = polyfit(y, gray(y), degree=2)
    gray_corrected(y) = gray(y) - trend(y) + global_mean

  Depth calibration:
    depth(y) = top_depth + (y - top_y) / (bottom_y - top_y) * (bottom_depth - top_depth)
"""
import sys
import os
import csv
import numpy as np
import cv2
from PyQt6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# ═══════════════════════════════════════════════
#  Multi-channel extraction engine
# ═══════════════════════════════════════════════

class MultiChannelExtractor:
    """Extract 6-channel profiles from rectangular ROIs."""

    CHANNELS = ["gray", "redness", "greenness", "blueness", "L_star", "a_star"]

    @staticmethod
    def inner_roi(red_mask: np.ndarray, outer_rect: tuple,
                  min_frac: float = 0.15, max_scan: int = 20):
        """Scan inward from each edge to find where the red border ends.

        To avoid false positives from red rock markings (spray paint),
        each edge is scanned only in a narrow central band:
        - top/bottom: scan only columns [w/4 .. 3w/4]
        - left/right: scan only rows [h/4 .. 3h/4]

        Max scan distance limits how far we search before falling back
        to a conservative 3px trim (handles photos with internal red marks).

        Returns: (ix, iy, iw, ih) inner rectangle.
        """
        x, y, w, h = outer_rect
        roi = red_mask[y:y + h, x:x + w]
        col_a, col_b = w // 4, 3 * w // 4
        row_a, row_b = h // 4, 3 * h // 4

        def _scan_edge(direction, limit):
            n = limit
            for i in range(limit):
                if direction == 'top':
                    strip = roi[i, col_a:col_b]
                elif direction == 'bottom':
                    strip = roi[h - 1 - i, col_a:col_b]
                elif direction == 'left':
                    strip = roi[row_a:row_b, i]
                else:  # 'right'
                    strip = roi[row_a:row_b, w - 1 - i]
                if np.count_nonzero(strip) / max(len(strip), 1) < min_frac:
                    n = i
                    break
            return n

        top_i = _scan_edge('top', max_scan)
        bot_i = _scan_edge('bottom', max_scan)
        left_i = _scan_edge('left', max_scan)
        right_i = _scan_edge('right', max_scan)

        # If scanner hit the limit (red everywhere), fall back to 3px
        if top_i >= max_scan:
            top_i = 3
        if bot_i >= max_scan:
            bot_i = 3
        if left_i >= max_scan:
            left_i = 3
        if right_i >= max_scan:
            right_i = 3

        ix, iy = x + left_i, y + top_i
        iw = (w - right_i) - left_i
        ih = (h - bot_i) - top_i
        if iw < 10 or ih < 20:
            return outer_rect
        return (ix, iy, iw, ih)

    @staticmethod
    def extract_single(image_bgr: np.ndarray, roi: tuple):
        """Extract all 6 channels from one ROI.
        roi must be the INNER rectangle (after border trimming)."""
        x, y, w, h = roi
        roi_img = image_bgr[y:y + h, x:x + w].astype(np.float32)

        B = roi_img[:, :, 0]
        G = roi_img[:, :, 1]
        R = roi_img[:, :, 2]
        total = R + G + B + 1e-6

        profiles = {}
        profiles["gray"] = np.mean(0.299 * R + 0.587 * G + 0.114 * B, axis=1)
        profiles["redness"] = np.mean(R / total, axis=1)
        profiles["greenness"] = np.mean(G / total, axis=1)
        profiles["blueness"] = np.mean(B / total, axis=1)

        roi_uint8 = image_bgr[y:y + h, x:x + w]
        lab = cv2.cvtColor(roi_uint8, cv2.COLOR_BGR2Lab).astype(np.float32)
        profiles["L_star"] = np.mean(lab[:, :, 0], axis=1)
        profiles["a_star"] = np.mean(lab[:, :, 1], axis=1)

        return profiles

    @staticmethod
    def group_boxes(boxes: list):
        """Return list of groups, each group = list of box indices that overlap >=50%.
        Boxes within a group are parallel (same strata), groups are sequential."""
        if not boxes:
            return []
        groups = []
        used = set()
        for i, (_, y_i, _, h_i) in enumerate(boxes):
            if i in used:
                continue
            group = [i]
            used.add(i)
            top_i, bottom_i = y_i, y_i + h_i
            for j, (_, y_j, _, h_j) in enumerate(boxes):
                if j in used:
                    continue
                top_j, bottom_j = y_j, y_j + h_j
                overlap = min(bottom_i, bottom_j) - max(top_i, top_j)
                span_i = bottom_i - top_i
                span_j = bottom_j - top_j
                min_span = min(span_i, span_j)
                if min_span > 0 and overlap / min_span >= 0.5:
                    group.append(j)
                    used.add(j)
                    top_i = min(top_i, top_j)
                    bottom_i = max(bottom_i, bottom_j)
            groups.append(group)
        return groups

    @staticmethod
    def extract_all(image_bgr: np.ndarray, boxes: list, red_mask: np.ndarray = None):
        """Extract all channels from all boxes, handling overlapping boxes.
        If red_mask is provided, inner_roi() is used to trim the red border
        precisely (scanning inward from each edge).

        Boxes that overlap vertically (>=50% overlap) are averaged — they
        are parallel views of the same strata, not sequential segments.
        Non-overlapping boxes are concatenated in stratigraphic order.
        """
        if not boxes:
            return {ch: np.array([]) for ch in MultiChannelExtractor.CHANNELS}, []

        # Step 0 — Compute inner ROIs from the red mask (precise border trim)
        if red_mask is not None:
            inner_boxes = [MultiChannelExtractor.inner_roi(red_mask, b) for b in boxes]
        else:
            inner_boxes = boxes

        # Step 1 — Extract raw profiles from each inner ROI
        raw_profiles = []
        for inner_rect in inner_boxes:
            raw_profiles.append(
                MultiChannelExtractor.extract_single(image_bgr, inner_rect))

        # Step 2 — Group overlapping boxes
        groups = []  # list of lists: [[box_idx, ...], ...]
        used = set()

        for i, (_, y_i, _, h_i) in enumerate(boxes):
            if i in used:
                continue
            group = [i]
            used.add(i)
            top_i, bottom_i = y_i, y_i + h_i

            for j, (_, y_j, _, h_j) in enumerate(boxes):
                if j in used:
                    continue
                top_j, bottom_j = y_j, y_j + h_j
                overlap = min(bottom_i, bottom_j) - max(top_i, top_j)
                span_i = bottom_i - top_i
                span_j = bottom_j - top_j
                min_span = min(span_i, span_j)
                if min_span > 0 and overlap / min_span >= 0.5:
                    group.append(j)
                    used.add(j)
                    # Expand the reference span for subsequent comparisons
                    top_i = min(top_i, top_j)
                    bottom_i = max(bottom_i, bottom_j)

            groups.append(group)

        # Step 3 — Build merged profiles
        all_profiles = {ch: [] for ch in MultiChannelExtractor.CHANNELS}
        segments = []
        offset = 0

        for group in groups:
            if len(group) == 1:
                # Single box — use directly
                idx = group[0]
                prof = raw_profiles[idx]
            else:
                # Multiple overlapping boxes — average them
                # First, pad all to the same length (max height in group)
                max_h = max(len(raw_profiles[i]["gray"]) for i in group)
                prof = {}
                for ch in MultiChannelExtractor.CHANNELS:
                    padded = np.full((len(group), max_h), np.nan)
                    for k, idx in enumerate(group):
                        data = raw_profiles[idx][ch]
                        padded[k, :len(data)] = data
                    prof[ch] = np.nanmean(padded, axis=0)

            n = len(prof["gray"])
            for ch in MultiChannelExtractor.CHANNELS:
                all_profiles[ch].append(prof[ch])
            # Segment metadata: use the first box's index as label
            segments.append((offset, offset + n, group[0]))
            offset += n

        return {ch: np.concatenate(all_profiles[ch])
                for ch in MultiChannelExtractor.CHANNELS}, segments


class IlluminationCorrector:
    """Remove low-frequency illumination gradient via polynomial detrending."""

    @staticmethod
    def correct(profile: np.ndarray, degree: int = 2) -> tuple:
        """Fit a polynomial trend and subtract it.
        Returns (corrected_profile, trend_line).
        """
        n = len(profile)
        if n < 10:
            return profile.copy(), np.full(n, np.mean(profile))
        x = np.arange(n, dtype=np.float64)
        coeffs = np.polyfit(x, profile, degree)
        trend = np.polyval(coeffs, x)
        global_mean = np.mean(profile)
        corrected = profile - trend + global_mean
        return corrected.astype(np.float32), trend.astype(np.float32)


class DepthCalibrator:
    """Pixel-to-depth conversion using 2 known drill hole positions."""

    def __init__(self):
        self.top_y = None       # pixel y at top reference
        self.bottom_y = None    # pixel y at bottom reference
        self.top_depth_m = None
        self.bottom_depth_m = None
        self.is_calibrated = False

    def set_points(self, top_y, bottom_y, top_depth_m, bottom_depth_m):
        self.top_y = top_y
        self.bottom_y = bottom_y
        self.top_depth_m = top_depth_m
        self.bottom_depth_m = bottom_depth_m
        self.is_calibrated = True

    def pixel_to_depth(self, y_pixel_array):
        """Convert pixel y coordinates to depth in meters."""
        if not self.is_calibrated:
            return np.asarray(y_pixel_array, dtype=np.float32)
        y = np.asarray(y_pixel_array, dtype=np.float64)
        fraction = (y - self.top_y) / (self.bottom_y - self.top_y)
        return (self.top_depth_m + fraction * (self.bottom_depth_m - self.top_depth_m)).astype(np.float32)

    def clear(self):
        self.__init__()


# ═══════════════════════════════════════════════
#  Red box detector (unchanged from v3)
# ═══════════════════════════════════════════════

class RedBoxDetector:
    @staticmethod
    def detect_all(image_bgr: np.ndarray):
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (160, 80, 80), (180, 255, 255))
        raw_mask = cv2.bitwise_or(mask1, mask2)  # pre-morphology, for inner_roi
        kernel = np.ones((7, 7), np.uint8)
        clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            aspect = h / max(w, 1)
            if h >= 150 and area >= 5000 and aspect >= 2.0:
                boxes.append((x, y, w, h))
        if not boxes:
            return [], raw_mask
        boxes.sort(key=lambda b: b[1] + b[3], reverse=True)
        return boxes, raw_mask


# ═══════════════════════════════════════════════
#  Colors & styling
# ═══════════════════════════════════════════════

BOX_COLORS = [
    '#2D7DD2', '#E8734A', '#1A9C6E', '#F0B429',
    '#8E44AD', '#E74C3C', '#27AE60', '#2980B9',
    '#D35400', '#16A085', '#C0392B', '#7F8C8D',
]

CHANNEL_COLORS = {
    "gray":       '#333333',
    "redness":    '#E74C3C',
    "greenness":  '#27AE60',
    "blueness":   '#2980B9',
    "L_star":     '#555555',
    "a_star":     '#E8734A',
}

CHANNEL_LABELS = {
    "gray":       "Gray (0–255)",
    "redness":    "Redness R/(R+G+B)",
    "greenness":  "Greenness G/(R+G+B)",
    "blueness":   "Blueness B/(R+G+B)",
    "L_star":     "CIE L* (0–100)",
    "a_star":     "CIE a* (red–green)",
}


# ═══════════════════════════════════════════════
#  Multi-channel plot widget
# ═══════════════════════════════════════════════

class MultiChannelPlotWidget(FigureCanvasQTAgg):
    """2×3 grid plot showing all 6 channels with illumination correction."""

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(8, 10), dpi=100)
        self.figure.set_facecolor('#fafafa')
        super().__init__(self.figure)
        self.setParent(parent)
        self.axes = {}
        self._create_subplots()

    def _create_subplots(self):
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        for (row, col), ch in zip(positions, MultiChannelExtractor.CHANNELS):
            self.axes[ch] = self.figure.add_subplot(2, 3, row * 3 + col + 1)

    def plot(self, profiles: dict, segments: list, depth_m: np.ndarray | None = None,
             illum_corrected: bool = False, trend_lines: dict | None = None):
        """Plot all channels. depth_m overrides y-axis labels when calibrated."""
        for i, ch in enumerate(MultiChannelExtractor.CHANNELS):
            ax = self.axes[ch]
            ax.clear()
            data = profiles.get(ch)
            if data is None or len(data) == 0:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=10, color='gray')
                ax.set_title(f'{ch}', fontsize=9, fontweight='bold')
                continue

            n = len(data)
            color = CHANNEL_COLORS[ch]
            y_vals = np.arange(n)

            # Per-box background bands
            for seg_start, seg_end, box_idx in segments:
                col = BOX_COLORS[box_idx % len(BOX_COLORS)]
                ax.axhspan(seg_start, seg_end, alpha=0.06, color=col)

            # Main signal
            ax.fill_betweenx(y_vals, data, data.min(), alpha=0.15, color=color)
            ax.plot(data, y_vals, color=color, linewidth=0.5)

            # Illumination trend (if corrected)
            if illum_corrected and trend_lines and ch in trend_lines:
                ax.plot(trend_lines[ch], y_vals, '--', color='#999999',
                        linewidth=0.8, alpha=0.7)

            # Mean line
            mean_val = float(np.mean(data))
            ax.axvline(mean_val, color=color, linestyle=':', linewidth=0.8, alpha=0.6)

            ax.invert_yaxis()
            ax.set_xlabel(CHANNEL_LABELS[ch], fontsize=7)
            ax.set_title(ch, fontsize=9, fontweight='bold', color=color)
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.tick_params(labelsize=7)

        self.figure.tight_layout(pad=2.0)
        self.draw()


# ═══════════════════════════════════════════════
#  Box list widget
# ═══════════════════════════════════════════════

class BoxListWidget(QtWidgets.QWidget):
    order_changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QtWidgets.QLabel("Detected Boxes (bottom→top)")
        header.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px;")
        layout.addWidget(header)
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.setStyleSheet("""
            QListWidget { border: 1px solid #ccc; border-radius: 4px; font-size: 12px; }
            QListWidget::item { padding: 4px; }
            QListWidget::item:alternate { background: #f5f5f5; }
        """)
        layout.addWidget(self.list_widget, 1)
        btn_layout = QtWidgets.QHBoxLayout()
        self.up_btn = QtWidgets.QPushButton("↑ Up")
        self.down_btn = QtWidgets.QPushButton("↓ Down")
        self.up_btn.clicked.connect(self._move_up)
        self.down_btn.clicked.connect(self._move_down)
        self.up_btn.setStyleSheet("padding: 4px 12px;")
        self.down_btn.setStyleSheet("padding: 4px 12px;")
        btn_layout.addWidget(self.up_btn)
        btn_layout.addWidget(self.down_btn)
        layout.addLayout(btn_layout)
        self._boxes = []

    def set_boxes(self, boxes, groups=None):
        self._boxes = boxes
        self.list_widget.clear()
        if groups is None:
            groups = [list(range(len(boxes)))]
        for g_idx, group in enumerate(groups):
            color = BOX_COLORS[g_idx % len(BOX_COLORS)]
            for pos, box_idx in enumerate(group):
                x, y, w, h = boxes[box_idx]
                lbl = f"G{g_idx+1}#{pos+1}" if len(group) > 1 else f"Box {box_idx+1}"
                item = QtWidgets.QListWidgetItem(
                    f"{lbl}  @ ({x},{y})  {w}×{h} px  [{h} rows]")
                item.setForeground(QtGui.QColor(color))
                self.list_widget.addItem(item)

    def get_boxes(self):
        return self._boxes

    def _move_up(self):
        row = self.list_widget.currentRow()
        if row <= 0:
            return
        self._boxes[row], self._boxes[row-1] = self._boxes[row-1], self._boxes[row]
        self.set_boxes(self._boxes)
        self.list_widget.setCurrentRow(row - 1)
        self.order_changed.emit()

    def _move_down(self):
        row = self.list_widget.currentRow()
        if row < 0 or row >= len(self._boxes) - 1:
            return
        self._boxes[row], self._boxes[row+1] = self._boxes[row+1], self._boxes[row]
        self.set_boxes(self._boxes)
        self.list_widget.setCurrentRow(row + 1)
        self.order_changed.emit()


# ═══════════════════════════════════════════════
#  Depth calibration dialog
# ═══════════════════════════════════════════════

class DepthCalibrationDialog(QtWidgets.QDialog):
    """Dialog for entering depth calibration via 2 known drill holes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Depth Calibration")
        self.setMinimumWidth(350)
        layout = QtWidgets.QFormLayout(self)

        layout.addRow(QtWidgets.QLabel(
            "Enter pixel Y-coordinates and depths for TWO known drill holes.\n"
            "Hover over the image to see pixel coordinates.\n\n"
            "Top hole (shallower):"))
        self.top_y = QtWidgets.QDoubleSpinBox()
        self.top_y.setDecimals(1)
        self.top_y.setRange(0, 10000)
        layout.addRow("Top Y (pixel):", self.top_y)

        self.top_depth = QtWidgets.QDoubleSpinBox()
        self.top_depth.setDecimals(3)
        self.top_depth.setRange(0, 1000)
        self.top_depth.setSuffix(" m")
        layout.addRow("Top depth (m):", self.top_depth)

        layout.addRow(QtWidgets.QLabel("\nBottom hole (deeper):"))
        self.bottom_y = QtWidgets.QDoubleSpinBox()
        self.bottom_y.setDecimals(1)
        self.bottom_y.setRange(0, 10000)
        layout.addRow("Bottom Y (pixel):", self.bottom_y)

        self.bottom_depth = QtWidgets.QDoubleSpinBox()
        self.bottom_depth.setDecimals(3)
        self.bottom_depth.setRange(0, 1000)
        self.bottom_depth.setSuffix(" m")
        layout.addRow("Bottom depth (m):", self.bottom_depth)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_values(self):
        return (self.top_y.value(), self.top_depth.value(),
                self.bottom_y.value(), self.bottom_depth.value())

    def set_values(self, top_y, top_d, bot_y, bot_d):
        self.top_y.setValue(top_y)
        self.top_depth.setValue(top_d)
        self.bottom_y.setValue(bot_y)
        self.bottom_depth.setValue(bot_d)


# ═══════════════════════════════════════════════
#  Main window
# ═══════════════════════════════════════════════

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.image_bgr = None
        self.boxes = []
        self.profiles = {}
        self.segments = []
        self.current_path = ""
        self.calibrator = DepthCalibrator()
        self.illum_corrected = False
        self.trend_lines = {}
        self.last_calib_y_positions = None

        self.setWindowTitle("Gray Extractor v4.0 — Multi-Channel + Illum + Depth")
        self.setMinimumSize(1300, 800)
        self.resize(1600, 950)

        self._setup_ui()
        self._setup_toolbar()
        self._setup_statusbar()
        self.setAcceptDrops(True)
        self._apply_style()

    # ── UI ────────────────────────────────────

    def _setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        # Left panel
        left_panel = QtWidgets.QVBoxLayout()
        self.image_label = QtWidgets.QLabel("Drag & drop an image here\nor click File → Open")
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaaaaa; border-radius: 8px;
                background: #f0f0f0; color: #888888;
                font-size: 15px; min-width: 500px; min-height: 400px;
            }
        """)
        self.image_label.setMinimumSize(500, 400)
        self.image_label.setMouseTracking(True)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)
        left_panel.addWidget(self.scroll_area, 1)

        self.info_label = QtWidgets.QLabel("")
        self.info_label.setStyleSheet("color: #666; font-size: 12px; padding: 4px;")
        left_panel.addWidget(self.info_label)

        # Right panel
        right_panel = QtWidgets.QVBoxLayout()
        self.plot_widget = MultiChannelPlotWidget()
        self.box_list = BoxListWidget()
        self.box_list.order_changed.connect(self._on_order_changed)
        right_panel.addWidget(self.plot_widget, 4)
        right_panel.addWidget(self.box_list, 1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        left_container = QtWidgets.QWidget()
        left_container.setLayout(left_panel)
        right_container = QtWidgets.QWidget()
        right_container.setLayout(right_panel)
        splitter.addWidget(left_container)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter)

        # Enable mouse tracking on the image label
        self.image_label.setMouseTracking(True)

    def _setup_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QtCore.QSize(20, 20))

        act_open = QtGui.QAction("Open Image", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._open_image)
        tb.addAction(act_open)

        tb.addSeparator()

        self.act_calib = QtGui.QAction("Calibrate Depth", self)
        self.act_calib.setShortcut("Ctrl+D")
        self.act_calib.triggered.connect(self._open_depth_calib)
        self.act_calib.setEnabled(False)
        tb.addAction(self.act_calib)

        self.act_illum = QtGui.QAction("Illum Correct", self)
        self.act_illum.setShortcut("Ctrl+I")
        self.act_illum.setCheckable(True)
        self.act_illum.toggled.connect(self._toggle_illum)
        self.act_illum.setEnabled(False)
        tb.addAction(self.act_illum)

        tb.addSeparator()

        self.act_export = QtGui.QAction("Export CSV", self)
        self.act_export.setShortcut("Ctrl+S")
        self.act_export.triggered.connect(self._export_csv)
        self.act_export.setEnabled(False)
        tb.addAction(self.act_export)

        tb.addSeparator()

        act_refresh = QtGui.QAction("Re-detect", self)
        act_refresh.setShortcut("F5")
        act_refresh.triggered.connect(self._detect_and_extract)
        tb.addAction(act_refresh)

    def _setup_statusbar(self):
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready — Open an image to start")

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background: #fafafa; }
            QToolBar { background: #fff; border-bottom: 1px solid #ddd; padding: 4px; spacing: 6px; }
            QStatusBar { background: #fff; border-top: 1px solid #ddd; color: #444; font-size: 12px; }
        """)

    # ── Image loading ─────────────────────────

    def _open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*)")
        if path:
            self._load_image(path)

    def _load_image(self, path: str):
        # Try OpenCV first, fall back to PIL (handles RGBA, CMYK, etc.)
        self.image_bgr = cv2.imread(path)
        if self.image_bgr is None:
            try:
                from PIL import Image
                pil = Image.open(path)
                if pil.mode in ('RGBA', 'LA', 'P'):
                    pil = pil.convert('RGB')
                self.image_bgr = cv2.cvtColor(
                    np.array(pil), cv2.COLOR_RGB2BGR)
            except Exception:
                pass
        if self.image_bgr is None:
            QtWidgets.QMessageBox.critical(self, "Error",
                f"Cannot open image:\n{path}\n\nCheck file format.")
            return
        self.current_path = path
        fname = os.path.basename(path)
        h, w = self.image_bgr.shape[:2]
        self.status_bar.showMessage(f"Loaded: {fname}  ({w} × {h})")
        self._detect_and_extract()

    # ── Detection + extraction ───────────────

    def _detect_and_extract(self):
        if self.image_bgr is None:
            return

        self.boxes, mask = RedBoxDetector.detect_all(self.image_bgr)
        self.red_mask = mask  # keep for inner_roi border trimming
        self.act_calib.setEnabled(bool(self.boxes))
        self.act_illum.setEnabled(bool(self.boxes))
        self.act_export.setEnabled(bool(self.boxes))
        self.illum_corrected = False
        self.act_illum.setChecked(False)
        self.trend_lines = {}

        display = self.image_bgr.copy()

        if not self.boxes:
            cv2.putText(display, "X  No red boxes found!",
                        (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            self.profiles = {ch: np.array([]) for ch in MultiChannelExtractor.CHANNELS}
            self.segments = []
            self.box_list.set_boxes([])
            self.info_label.setText("No red boxes detected.")
            self.status_bar.showMessage("WARNING: No red boxes found")
        else:
            # ── Compute groups: overlapping boxes share a group ──
            groups = MultiChannelExtractor.group_boxes(self.boxes)

            # Draw box borders — group color, label like "G1#1"
            for g_idx, group in enumerate(groups):
                color_bgr = self._hex_to_bgr(BOX_COLORS[g_idx % len(BOX_COLORS)])
                for pos, box_idx in enumerate(group):
                    x, y, w, h = self.boxes[box_idx]
                    cv2.rectangle(display, (x, y), (x + w, y + h), color_bgr, 3)
                    lbl = f"G{g_idx+1}#{pos+1}" if len(group) > 1 else f"#{box_idx+1}"
                    (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)
                    cv2.rectangle(display, (x, y - lh - 10), (x + lw + 6, y), color_bgr, -1)
                    cv2.putText(display, lbl, (x + 3, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Draw arrows between groups (not within groups)
            for g_idx in range(len(groups) - 1):
                # Last box of current group
                last_box = groups[g_idx][-1]
                x1, y1, w1, h1 = self.boxes[last_box]
                # First box of next group
                first_box = groups[g_idx + 1][0]
                x2, y2, w2, h2 = self.boxes[first_box]
                cv2.arrowedLine(display, (x1 + w1 // 2, y1 + h1),
                                (x2 + w2 // 2, y2),
                                (80, 80, 80), 2, tipLength=0.04)

            # Draw depth calibration markers if set
            if self.calibrator.is_calibrated:
                ty, by = self.calibrator.top_y, self.calibrator.bottom_y
                for py, lbl in [(ty, "Top"), (by, "Bottom")]:
                    if 0 <= py <= display.shape[0]:
                        cv2.line(display, (0, int(py)), (display.shape[1], int(py)),
                                 (255, 165, 0), 2)
                        cv2.putText(display, f"{lbl} {py:.0f}px",
                                    (10, int(py) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 2)

            # Extract (with border-trim info)
            outer_heights = sum(b[3] for b in self.boxes)
            self.profiles, self.segments = MultiChannelExtractor.extract_all(
                self.image_bgr, self.boxes, self.red_mask)
            inner_heights = sum(s[1] - s[0] for s in self.segments) if self.segments else 0
            trimmed_px = outer_heights - inner_heights
            # Update box list with group-aware display
            self.box_list.set_boxes(self.boxes, groups)

            n = len(self.profiles.get("gray", []))
            info = f"Boxes: {len(self.boxes)}  →  {len(groups)} group(s)  →  Profile: {n} rows\n"
            info += f"Border trimmed: {trimmed_px} px total ({trimmed_px//len(self.boxes)//2} px/edge avg)\n"
            info += "─" * 40 + "\n"
            if self.calibrator.is_calibrated:
                td = self.calibrator.top_depth_m
                bd = self.calibrator.bottom_depth_m
                info += f"Depth: {td:.2f} – {bd:.2f} m  ({abs(bd-td):.2f} m span)\n"
            for g_idx, group in enumerate(groups):
                grp_label = f"Group {g_idx+1}"
                if len(group) == 1:
                    x, y, w, h = self.boxes[group[0]]
                    info += f"  {grp_label}: 1 box @ ({x},{y}) {w}×{h} px\n"
                else:
                    info += f"  {grp_label}: {len(group)} overlapping boxes → averaged\n"
                    for pos, box_idx in enumerate(group):
                        x, y, w, h = self.boxes[box_idx]
                        info += f"    #{box_idx+1}: ({x},{y}) {w}×{h} px\n"
            info += "\n— Groups = parallel strata (averaged)\n— Arrow = sequential strata (concatenated)"
            self.info_label.setText(info)
            self.status_bar.showMessage(
                f"{len(groups)} group(s)  |  "
                f"{n} rows  |  6 channels")

        self._refresh_plot()
        self._show_image(display)

    def _refresh_plot(self):
        """Re-plot with current illumination correction state."""
        profiles_to_plot = self.profiles.copy()
        trend = {}
        if self.illum_corrected and self.profiles:
            for ch in MultiChannelExtractor.CHANNELS:
                if ch in self.profiles and len(self.profiles[ch]) > 0:
                    profiles_to_plot[ch], trend[ch] = IlluminationCorrector.correct(
                        self.profiles[ch])
            self.trend_lines = trend
        self.plot_widget.plot(profiles_to_plot, self.segments,
                              illum_corrected=self.illum_corrected,
                              trend_lines=self.trend_lines)

    # ── Depth calibration ────────────────────

    def _open_depth_calib(self):
        dlg = DepthCalibrationDialog(self)
        if self.last_calib_y_positions:
            dlg.set_values(*self.last_calib_y_positions)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            ty, td, by, bd = dlg.get_values()
            self.last_calib_y_positions = (ty, td, by, bd)
            self.calibrator.set_points(ty, by, td, bd)
            self._detect_and_extract()
            self.status_bar.showMessage(
                f"Depth calibrated: {td:.2f} – {bd:.2f} m  "
                f"({abs(bd-td):.2f} m over {abs(by-ty):.0f} px)")

    # ── Illumination correction toggle ────────

    def _toggle_illum(self, checked):
        self.illum_corrected = checked
        self._refresh_plot()
        if checked:
            self.status_bar.showMessage(
                "Illumination correction ON — 2nd-order polynomial trend removed")
        else:
            self.status_bar.showMessage("Illumination correction OFF — raw data")

    # ── Box reorder ───────────────────────────

    def _on_order_changed(self):
        self.boxes = self.box_list.get_boxes()
        if self.image_bgr is not None and self.boxes:
            self.profiles, self.segments = MultiChannelExtractor.extract_all(
                self.image_bgr, self.boxes, getattr(self, 'red_mask', None))
            self._refresh_plot()
            self._redraw_image_only()
            self.status_bar.showMessage(f"Order updated — {len(self.boxes)} box(es)")

    # ── Export ────────────────────────────────

    def _export_csv(self):
        profiles_src = self.profiles
        if self.illum_corrected:
            profiles_src = {}
            for ch in MultiChannelExtractor.CHANNELS:
                if ch in self.profiles and len(self.profiles[ch]) > 0:
                    corrected, _ = IlluminationCorrector.correct(self.profiles[ch])
                    profiles_src[ch] = corrected

        gray_data = profiles_src.get("gray", np.array([]))
        if len(gray_data) == 0:
            return

        default_name = "gray_profile.csv"
        if self.current_path:
            base = os.path.splitext(os.path.basename(self.current_path))[0]
            default_name = f"{base}_gray_profile.csv"

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Profile", default_name,
            "CSV Files (*.csv);;All Files (*)")
        if not path:
            return

        # Build box_id and depth columns
        box_id = np.full(len(gray_data), -1, dtype=int)
        for seg_start, seg_end, box_idx in self.segments:
            box_id[seg_start:seg_end] = box_idx + 1

        row_indices = np.arange(len(gray_data), dtype=np.float64)
        depth_m = self.calibrator.pixel_to_depth(row_indices)

        headers = ["row_px", "depth_m", "box_id"] + MultiChannelExtractor.CHANNELS
        if not self.calibrator.is_calibrated:
            headers.remove("depth_m")

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for i in range(len(gray_data)):
                row = [i]
                if self.calibrator.is_calibrated:
                    row.append(round(float(depth_m[i]), 4))
                row.append(int(box_id[i]))
                for ch in MultiChannelExtractor.CHANNELS:
                    row.append(round(float(profiles_src[ch][i]), 4))
                writer.writerow(row)

        extra = ""
        if self.illum_corrected:
            extra = " (illum-corrected)"
        self.status_bar.showMessage(
            f"Exported: {os.path.basename(path)}  "
            f"({len(gray_data)} rows, 6 channels{extra})")

    # ── Drag & drop ───────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                self._load_image(path)
                break

    # ── Display helpers ───────────────────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.image_bgr is not None:
            self._redraw_image_only()

    def _redraw_image_only(self):
        if self.image_bgr is None:
            return
        display = self.image_bgr.copy()
        if self.boxes:
            groups = MultiChannelExtractor.group_boxes(self.boxes)
            for g_idx, group in enumerate(groups):
                cb = self._hex_to_bgr(BOX_COLORS[g_idx % len(BOX_COLORS)])
                for pos, box_idx in enumerate(group):
                    x, y, w, h = self.boxes[box_idx]
                    cv2.rectangle(display, (x, y), (x + w, y + h), cb, 3)
                    lbl = f"G{g_idx+1}#{pos+1}" if len(group) > 1 else f"#{box_idx+1}"
                    (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)
                    cv2.rectangle(display, (x, y - lh - 10), (x + lw + 6, y), cb, -1)
                    cv2.putText(display, lbl, (x + 3, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            for g_idx in range(len(groups) - 1):
                last_box = groups[g_idx][-1]
                x1, y1, w1, h1 = self.boxes[last_box]
                first_box = groups[g_idx + 1][0]
                x2, y2, w2, h2 = self.boxes[first_box]
                cv2.arrowedLine(display, (x1 + w1 // 2, y1 + h1),
                                (x2 + w2 // 2, y2), (80, 80, 80), 2, tipLength=0.04)
            if self.calibrator.is_calibrated:
                ty, by = self.calibrator.top_y, self.calibrator.bottom_y
                for pv, lb in [(ty, "Top"), (by, "Bottom")]:
                    if 0 <= pv <= display.shape[0]:
                        cv2.line(display, (0, int(pv)), (display.shape[1], int(pv)),
                                 (255, 165, 0), 2)
                        cv2.putText(display, f"{lb} {pv:.0f}px",
                                    (10, int(pv) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 2)
        else:
            cv2.putText(display, "X  No red boxes found!",
                        (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        self._show_image(display)

    def _show_image(self, bgr_image):
        display_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        h_img, w_img, ch = display_rgb.shape
        qt_img = QtGui.QImage(display_rgb.data, w_img, h_img,
                              ch * w_img, QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_img)
        aw = self.scroll_area.viewport().width() - 20
        ah = self.scroll_area.viewport().height() - 20
        if aw > 0 and ah > 0:
            scaled = pixmap.scaled(aw, ah,
                                   QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                   QtCore.Qt.TransformationMode.SmoothTransformation)
        else:
            scaled = pixmap
        self.image_label.setPixmap(scaled)
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

    @staticmethod
    def _hex_to_bgr(h):
        return (int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16))


# ═══════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
