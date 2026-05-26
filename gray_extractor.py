"""
Gray Extractor v8.1 — Thickness Popup Dialog
===========================================
v8.1: Thickness mode now opens a popup dialog with 5 tabbed panels
  - Each panel = standalone FigureCanvas + NavigationToolbar
  - Zoom/pan/save on every panel individually
  - Real-time update when σ or Prom changes
v8.0: Added thickness analysis mode (View → Thickness)
  - BedBoundaryDetector: valley detection → bed boundaries → thickness
v7.0-v7.2: Clean slate, dual raw/corr CSV, inner boundary viz, info table
"""
import sys, os, csv
import numpy as np, cv2
from PyQt6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# ── Optional scipy imports (fallback to numpy if unavailable) ──
try:
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d as _gauss1d
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

    def _gauss1d(data, sigma):
        """Pure numpy Gaussian filter fallback."""
        if sigma < 0.5:
            return data.astype(np.float64)
        # Truncate at 4*sigma
        radius = int(np.ceil(4 * sigma))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        return np.convolve(data.astype(np.float64), kernel, mode='same')

    def find_peaks(data, height=None, prominence=None, distance=1):
        """Pure numpy peak detection fallback.
        Detects local maxima via simple comparison with neighbors.
        Only distance and prominence are supported; height is ignored unless
        used as a simple threshold."""
        data = np.asarray(data, dtype=np.float64)
        n = len(data)
        if n < 3:
            return np.array([], dtype=int), {}

        # Local maxima: greater than both neighbors
        peaks = np.where((data[1:-1] > data[:-2]) & (data[1:-1] > data[2:]))[0] + 1

        # Distance filter: keep highest peak in each sliding window
        if distance > 1 and len(peaks) > 0:
            kept = []
            i = 0
            while i < len(peaks):
                window_end = i
                while (window_end + 1 < len(peaks) and
                       peaks[window_end + 1] - peaks[i] < distance):
                    window_end += 1
                best = i + np.argmax(data[peaks[i:window_end + 1]])
                kept.append(peaks[best])
                i = window_end + 1
            peaks = np.array(kept, dtype=int)

        # Prominence filter (simplified)
        if prominence is not None and prominence > 0 and len(peaks) > 0:
            prom_values = np.zeros(len(peaks))
            for j, p in enumerate(peaks):
                # Find lowest valley to the left
                left_min = data[p]
                for k in range(p - 1, -1, -1):
                    if data[k] < left_min:
                        left_min = data[k]
                    if k > 0 and data[k] > data[p]:
                        break
                # Find lowest valley to the right
                right_min = data[p]
                for k in range(p + 1, n):
                    if data[k] < right_min:
                        right_min = data[k]
                    if k < n - 1 and data[k] > data[p]:
                        break
                prom_values[j] = data[p] - max(left_min, right_min)
            peaks = peaks[prom_values >= prominence]

        props = {}
        return peaks, props

# ═══════════════════════════════════════════════
#  Multi-channel extraction engine
# ═══════════════════════════════════════════════

class MultiChannelExtractor:
    CHANNELS = ["gray", "redness", "greenness", "blueness", "L_star", "a_star"]

    @staticmethod
    def inner_roi(red_mask, outer_rect, min_frac=0.05, max_scan=50):
        x, y, w, h = outer_rect
        roi = red_mask[y:y+h, x:x+w]

        def _scan(direction, limit):
            n = limit
            for i in range(limit):
                if direction == 'top':
                    frac = np.count_nonzero(roi[i, :]) / max(w, 1)
                elif direction == 'bottom':
                    frac = np.count_nonzero(roi[h-1-i, :]) / max(w, 1)
                elif direction == 'left':
                    frac = np.count_nonzero(roi[:, i]) / max(h, 1)
                else:
                    frac = np.count_nonzero(roi[:, w-1-i]) / max(h, 1)
                if frac < min_frac:
                    n = i; break
            return n

        ti = _scan('top', max_scan);   bi = _scan('bottom', max_scan)
        li = _scan('left', max_scan);  ri = _scan('right', max_scan)

        fb_h = max(4, h // 40)
        fb_w = max(4, w // 30)
        if ti >= max_scan: ti = fb_h
        if bi >= max_scan: bi = fb_h
        if li >= max_scan: li = fb_w
        if ri >= max_scan: ri = fb_w

        ix, iy = x + li, y + ti
        iw = (w - ri) - li; ih = (h - bi) - ti
        if iw < 10 or ih < 20: return outer_rect
        return (ix, iy, iw, ih)

    @staticmethod
    def extract_single(image_bgr, roi):
        x, y, w, h = roi
        roi_img = image_bgr[y:y+h, x:x+w].astype(np.float32)
        B, G, R = roi_img[:,:,0], roi_img[:,:,1], roi_img[:,:,2]
        total = R+G+B+1e-6
        profiles = {
            "gray": np.mean(0.299*R+0.587*G+0.114*B, axis=1),
            "redness": np.mean(R/total, axis=1),
            "greenness": np.mean(G/total, axis=1),
            "blueness": np.mean(B/total, axis=1),
        }
        roi_u8 = image_bgr[y:y+h, x:x+w]
        lab = cv2.cvtColor(roi_u8, cv2.COLOR_BGR2Lab).astype(np.float32)
        profiles["L_star"] = np.mean(lab[:,:,0], axis=1)
        profiles["a_star"] = np.mean(lab[:,:,1], axis=1)
        return profiles

    @staticmethod
    def group_boxes(boxes):
        if not boxes: return []
        centers = [y+h/2 for (_,y,_,h) in boxes]
        used = set(); groups = []
        for i in range(len(boxes)):
            if i in used: continue
            group = [i]; used.add(i)
            _, yi, _, hi = boxes[i]
            for j in range(len(boxes)):
                if j in used: continue
                _, yj, _, hj = boxes[j]
                dist = abs(centers[i]-centers[j])
                mh = min(hi, hj)
                if mh>0 and dist<0.35*mh:
                    group.append(j); used.add(j)
            groups.append(group)
        return groups

    @staticmethod
    def extract_all(image_bgr, boxes, red_mask=None):
        if not boxes:
            return {ch: np.array([]) for ch in MultiChannelExtractor.CHANNELS}, []
        inner_boxes = [MultiChannelExtractor.inner_roi(red_mask,b) if red_mask is not None else b for b in boxes]
        raw_profiles = [MultiChannelExtractor.extract_single(image_bgr, ib) for ib in inner_boxes]
        groups = MultiChannelExtractor.group_boxes(boxes)
        all_profiles = {ch: [] for ch in MultiChannelExtractor.CHANNELS}
        segments = []; offset = 0
        for group in groups:
            if len(group) == 1:
                prof = raw_profiles[group[0]]
            else:
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
            segments.append((offset, offset+n, group[0]))
            offset += n
        return {ch: np.concatenate(all_profiles[ch]) for ch in MultiChannelExtractor.CHANNELS}, segments


class IlluminationCorrector:
    @staticmethod
    def correct(profile, degree=2):
        n = len(profile)
        if n<10: return profile.copy(), np.full(n, np.mean(profile))
        x = np.arange(n, dtype=np.float64)
        coeffs = np.polyfit(x, profile, degree)
        trend = np.polyval(coeffs, x)
        corrected = profile - trend + np.mean(profile)
        return corrected.astype(np.float32), trend.astype(np.float32)


class DepthCalibrator:
    """Per-group depth calibration. Each group independently maps
    its pixel rows to a depth range [top_m, bottom_m] via linear interpolation."""
    def __init__(self):
        self.group_depth = {}   # {group_idx: (top_m, bottom_m)}
        self.is_calibrated = False

    def set_group(self, group_idx, top_m, bottom_m):
        self.group_depth[group_idx] = (top_m, bottom_m)
        self.is_calibrated = bool(self.group_depth)

    def remove_group(self, group_idx):
        self.group_depth.pop(group_idx, None)
        self.is_calibrated = bool(self.group_depth)

    def get_group(self, group_idx):
        return self.group_depth.get(group_idx)

    def pixel_to_depth(self, local_row, group_height, group_idx):
        if not self.is_calibrated or group_idx not in self.group_depth:
            return np.asarray(local_row, dtype=np.float32)
        top_m, bottom_m = self.group_depth[group_idx]
        y = np.asarray(local_row, dtype=np.float64)
        frac = y / max(group_height, 1)
        return (top_m + frac * (bottom_m - top_m)).astype(np.float32)

    def clear(self): self.__init__()


class RedBoxDetector:
    """Detect red rectangular bounding boxes in HSV space.

    Boxes are sorted by bottom-Y descending (i.e. stratigraphic order:
    highest in photo = deepest stratigraphically = last in list).
    This works because outcrop photos are shot top-down: the top of
    the photo is the shallowest stratigraphic level. Users who draw
    boxes out of order don't need to worry — extract_all processes
    boxes in this sorted order, and group_boxes further handles
    parallel (side-by-side) boxes via center-distance clustering.
    """
    @staticmethod
    def detect_all(image_bgr):
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, (0,80,80), (10,255,255))
        m2 = cv2.inRange(hsv, (160,80,80), (180,255,255))
        raw = cv2.bitwise_or(m1, m2)
        k = np.ones((7,7), np.uint8)
        clean = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k)
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, k)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if h>=150 and area>=5000 and h/max(w,1)>=2.0:
                boxes.append((x,y,w,h))
        if not boxes: return [], raw
        boxes.sort(key=lambda b: b[1]+b[3], reverse=True)
        return boxes, raw


# ═══════════════════════════════════════════════
#  Bed boundary detector — grayscale → thickness
# ═══════════════════════════════════════════════

class BedBoundaryDetector:
    """Detect bed boundaries from a grayscale profile and compute bed thicknesses.

    Principle:
      - Layer boundaries appear as sharp dark→light transitions in grayscale.
      - The boundary position is the PEAK of the gradient, not a specific gray value.
      - This is robust to lighting variations — a 20% overall brightness change
        doesn't shift where the gradient peaks.

    Algorithm:
      1. Gaussian smooth to suppress pixel-level noise
      2. Find local minima (dark = muddy/clay-rich layer boundaries) OR
         find gradient magnitude peaks (|d(gray)/dy| maxima)
      3. Filter by prominence and minimum distance
      4. Compute bed thickness = distance between consecutive boundaries
    """

    # We use valley detection (local minima) as default for carbonate rocks:
    # dark troughs = argillaceous/mud-rich interbeds = layer boundaries.
    # The alternative (gradient peak) works better for sharp lithologic contacts.

    def __init__(self, method="valley"):
        self.method = method
        self._boundaries = None       # pixel indices of detected boundaries
        self._thickness = None        # bed thicknesses (same unit as input)
        self._boundary_depths = None  # depth/position of each boundary

    def detect(self, gray, depth=None, smooth_sigma=3.0, prominence=None,
               min_thickness=None, min_thickness_unit='px'):
        """Detect bed boundaries from grayscale profile.

        Args:
            gray: 1D array of grayscale values
            depth: optional depth array (for calibrated output). If None, uses pixel index.
            smooth_sigma: Gaussian filter sigma (controls scale sensitivity).
                          Smaller → finer beds. Larger → only major boundaries.
            prominence: minimum valley depth for a boundary to count.
                        Default: 0.3 * std(smoothed_gray)
            min_thickness: minimum distance between boundaries.
                           Default: 3 * smooth_sigma
            min_thickness_unit: 'px' or 'cm'. If 'cm', min_thickness is in cm
                               and requires depth array.

        Returns:
            self (boundaries, thickness, boundary_depths are set as attributes)
        """
        n = len(gray)
        if n < 10:
            self._boundaries = np.array([], dtype=int)
            self._thickness = np.array([], dtype=np.float64)
            self._boundary_depths = np.array([], dtype=np.float64)
            return self

        # Step 1: Smooth
        gray_f = gray.astype(np.float64)
        smoothed = _gauss1d(gray_f, smooth_sigma)

        # Step 2: Default parameters
        if prominence is None:
            prominence = np.std(smoothed) * 0.3
        if min_thickness is None:
            min_thickness = max(3, int(smooth_sigma * 3))

        # Convert min_thickness from cm to px if depth is provided
        if min_thickness_unit == 'cm' and depth is not None:
            # Find approximate px/cm ratio
            depth_spacing = np.median(np.diff(depth))
            if depth_spacing > 0:
                min_thickness = max(3, int(min_thickness / float(depth_spacing) / 100))
            else:
                min_thickness = max(3, int(smooth_sigma * 3))

        # Step 3: Find boundaries as valleys (local minima of gray)
        # Valleys = peaks of -smoothed
        peaks, props = find_peaks(
            -smoothed,
            prominence=prominence,
            distance=max(2, min_thickness)
        )

        self._boundaries = np.asarray(peaks, dtype=int)

        # Step 4: Compute thickness and boundary positions
        if depth is not None and len(depth) == n:
            self._boundary_depths = depth[peaks]
            if len(peaks) >= 2:
                self._thickness = np.diff(self._boundary_depths)
            else:
                self._thickness = np.array([], dtype=np.float64)
        else:
            self._boundary_depths = peaks.astype(np.float64)
            if len(peaks) >= 2:
                self._thickness = np.diff(peaks).astype(np.float64)
            else:
                self._thickness = np.array([], dtype=np.float64)

        return self

    @property
    def n_beds(self):
        return max(0, len(self._boundaries) - 1)

    @property
    def boundaries(self):
        return self._boundaries if self._boundaries is not None else np.array([], dtype=int)

    @property
    def thickness(self):
        return self._thickness if self._thickness is not None else np.array([], dtype=np.float64)

    @property
    def boundary_depths(self):
        return self._boundary_depths if self._boundary_depths is not None else np.array([], dtype=np.float64)

    def stats(self):
        """Return a dict of statistics suitable for display."""
        t = self.thickness
        if len(t) < 2:
            return {"n_beds": 0, "median": 0, "mean": 0, "min": 0, "max": 0, "std": 0}
        return {
            "n_beds": self.n_beds,
            "median": float(np.median(t)),
            "mean": float(np.mean(t)),
            "min": float(np.min(t)),
            "max": float(np.max(t)),
            "std": float(np.std(t)),
        }


# ═══════════════════════════════════════════════
#  Colors
# ═══════════════════════════════════════════════

BOX_COLORS = ['#2D7DD2','#E8734A','#1A9C6E','#F0B429','#8E44AD','#E74C3C','#27AE60','#2980B9']
CHANNEL_COLORS = {"gray":'#333',"redness":'#E74C3C',"greenness":'#27AE60',"blueness":'#2980B9',"L_star":'#555',"a_star":'#E8734A'}
CHANNEL_LABELS = {"gray":"Gray (0–255)","redness":"R/(R+G+B)","greenness":"G/(R+G+B)","blueness":"B/(R+G+B)","L_star":"CIE L*","a_star":"CIE a*"}


# ═══════════════════════════════════════════════
#  Multi-channel plot widget
# ═══════════════════════════════════════════════

class MultiChannelPlotWidget(FigureCanvasQTAgg):
    """Three distinct view modes, each with its own layout."""

    LAYOUTS = {
        "all": {
            "gray":       (0.04, 0.35, 0.06, 0.90),
            "L_star":     (0.43, 0.25, 0.53, 0.43),
            "a_star":     (0.43, 0.25, 0.06, 0.43),
            "redness":    (0.72, 0.25, 0.68, 0.28),
            "greenness":  (0.72, 0.25, 0.37, 0.28),
            "blueness":   (0.72, 0.25, 0.06, 0.28),
        },
        "gray_lab": {
            "gray":       (0.06, 0.55, 0.06, 0.90),
            "L_star":     (0.65, 0.30, 0.53, 0.43),
            "a_star":     (0.65, 0.30, 0.06, 0.43),
        },
        "gray": {
            "gray":       (0.12, 0.76, 0.08, 0.84),
        },
    }

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(14, 9), dpi=100)
        self.figure.set_facecolor('#fafafa')
        super().__init__(self.figure)
        self.setParent(parent)
        self.axes = {}
        self._view_mode = "all"
        self._create_subplots()
        self._plot_cache = None

    def _create_subplots(self):
        for ax in list(self.axes.values()):
            self.figure.delaxes(ax)
        self.axes.clear()
        layout = self.LAYOUTS[self._view_mode]
        for ch, (left, width, bottom, height) in layout.items():
            self.axes[ch] = self.figure.add_axes((left, bottom, width, height))

    def set_view_mode(self, mode):
        if mode not in self.LAYOUTS: return
        if mode == self._view_mode and self.axes: return
        self._view_mode = mode
        self._create_subplots()
        if self._plot_cache:
            prof, seg, da, ic, tl = self._plot_cache
            self._do_plot(prof, seg, da, ic, tl)

    def _draw_one(self, ax, ch, data, n, segments, depth_array, illum_corrected, trend_lines):
        color = CHANNEL_COLORS[ch]
        y_vals = np.arange(n)
        for ss, se, bi in segments:
            ax.axhspan(ss, se, alpha=0.06, color=BOX_COLORS[bi % len(BOX_COLORS)])
        ax.fill_betweenx(y_vals, data, data.min(), alpha=0.15, color=color)
        ax.plot(data, y_vals, color=color, linewidth=0.45)
        if illum_corrected and trend_lines and ch in trend_lines:
            ax.plot(trend_lines[ch], y_vals, '--', color='#999', linewidth=0.7, alpha=0.7)

        ax.invert_yaxis()

        is_solo = (self._view_mode == "gray")
        fs = 9 if is_solo else (7 if ch == "gray" else 6)
        ax.text(0.03, 0.96, f" {ch} ", transform=ax.transAxes,
                fontsize=fs, fontweight='bold', color='white',
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.25', facecolor=color, alpha=0.85, edgecolor='none'))

        ax.text(0.97, 0.03, CHANNEL_LABELS.get(ch, ch), transform=ax.transAxes,
                fontsize=5 if not is_solo else 7, color='#666', ha='right', va='bottom')

        ax.axvline(float(np.mean(data)), color=color, linestyle=':', linewidth=0.7, alpha=0.5)
        ax.grid(True, alpha=0.15, linestyle='--')
        ax.tick_params(labelsize=6 if is_solo else 5)

        if depth_array is not None and len(depth_array) == n:
            n_ticks = 12 if is_solo else (10 if ch == "gray" else 5)
            step = max(1, n // n_ticks)
            tick_rows = list(range(0, n, step))
            tick_labels = [f"{depth_array[i]:.2f}" for i in tick_rows]
            ax.set_yticks(tick_rows)
            ax.set_yticklabels(tick_labels, fontsize=5 if not is_solo else 6)
            if ch == "gray" or is_solo:
                ax.set_ylabel("Depth (m)", fontsize=7, color='#999')
            for ss, se, bi in segments:
                if ss > 0:
                    ax.axhline(ss, color='#F0B429', linestyle='-', linewidth=0.8, alpha=0.5)
                    ax.annotate(f'{depth_array[min(ss, n-1)]:.2f}',
                                xy=(0.98, ss), xycoords=('axes fraction', 'data'),
                                fontsize=4, color='#F0B429', ha='right', va='center',
                                bbox=dict(boxstyle='round,pad=0.08', facecolor='white', alpha=0.7, edgecolor='none'))

    def _do_plot(self, profiles, segments, depth_array, illum_corrected, trend_lines):
        for ch in self.axes:
            ax = self.axes[ch]; ax.clear()
            data = profiles.get(ch)
            if data is None or len(data) == 0:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center',
                        fontsize=8, color='gray')
                ax.text(0.03, 0.96, f" {ch} ", transform=ax.transAxes, fontsize=6,
                        fontweight='bold', color='white', va='top',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#999', alpha=0.7))
                continue
            self._draw_one(ax, ch, data, len(data), segments, depth_array, illum_corrected, trend_lines)

    def plot(self, profiles, segments, depth_array=None, illum_corrected=False, trend_lines=None):
        self._plot_cache = (profiles, segments, depth_array, illum_corrected, trend_lines)
        self._do_plot(profiles, segments, depth_array, illum_corrected, trend_lines)
        self.draw()

    def save_plots(self, path):
        self.figure.savefig(path, dpi=200, bbox_inches='tight', facecolor='#fafafa')


# ═══════════════════════════════════════════════
#  Zoomable image viewer
# ═══════════════════════════════════════════════

class ZoomableImageView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = None
        self.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("border: 1px solid #ccc; background: #e8e8e8;")
        self.setMinimumSize(400, 300)
        self._zoom = 1.0
        self._image_rect = None

    def set_image(self, qimage):
        self._scene.clear()
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QtCore.QRectF(pixmap.rect()))
        self._image_rect = pixmap.rect()
        self.fitInView(self._scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = self.transform().m11()

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1/1.15
        new_zoom = self._zoom * factor
        if 0.05 < new_zoom < 20:
            self.scale(factor, factor)
            self._zoom = new_zoom
        event.accept()

    def fit_to_window(self):
        if self._image_rect:
            self.fitInView(self._scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = self.transform().m11()

    def reset_zoom(self):
        self.resetTransform()
        self._zoom = 1.0


# ═══════════════════════════════════════════════
#  Box list widget
# ═══════════════════════════════════════════════

class BoxListWidget(QtWidgets.QWidget):
    order_changed = QtCore.pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        l = QtWidgets.QVBoxLayout(self); l.setContentsMargins(0,0,0,0)
        h = QtWidgets.QLabel("Detected Boxes"); h.setStyleSheet("font-weight:bold;font-size:12px;padding:2px;")
        l.addWidget(h)
        self.list = QtWidgets.QListWidget()
        self.list.setAlternatingRowColors(True)
        self.list.setStyleSheet("QListWidget{border:1px solid #ccc;border-radius:4px;font-size:11px;} QListWidget::item{padding:3px;} QListWidget::item:alternate{background:#f5f5f5;}")
        l.addWidget(self.list, 1)
        bl = QtWidgets.QHBoxLayout()
        u = QtWidgets.QPushButton("↑"); d = QtWidgets.QPushButton("↓")
        u.clicked.connect(self._up); d.clicked.connect(self._down)
        u.setStyleSheet("padding:2px 10px;"); d.setStyleSheet("padding:2px 10px;")
        bl.addWidget(u); bl.addWidget(d); l.addLayout(bl)
        self._boxes = []
    def set_boxes(self, boxes, groups=None):
        self._boxes = boxes; self.list.clear()
        if groups is None: groups = [list(range(len(boxes)))]
        for gi, grp in enumerate(groups):
            c = BOX_COLORS[gi%len(BOX_COLORS)]
            for pos, bi in enumerate(grp):
                x,y,w,h = boxes[bi]
                lbl = f"G{gi+1}#{pos+1}" if len(grp)>1 else f"Box {bi+1}"
                item = QtWidgets.QListWidgetItem(f"{lbl} @({x},{y}) {w}×{h}")
                item.setForeground(QtGui.QColor(c)); self.list.addItem(item)
    def get_boxes(self): return self._boxes
    def _up(self):
        r = self.list.currentRow()
        if r<=0: return
        self._boxes[r], self._boxes[r-1] = self._boxes[r-1], self._boxes[r]
        self.set_boxes(self._boxes); self.list.setCurrentRow(r-1); self.order_changed.emit()
    def _down(self):
        r = self.list.currentRow()
        if r<0 or r>=len(self._boxes)-1: return
        self._boxes[r], self._boxes[r+1] = self._boxes[r+1], self._boxes[r]
        self.set_boxes(self._boxes); self.list.setCurrentRow(r+1); self.order_changed.emit()


# ═══════════════════════════════════════════════
#  Main window
# ═══════════════════════════════════════════════


# ============================================================
#  Thickness Analysis Dialog (v8.1 — popup with 5 tabbed panels)
# ============================================================

class ThicknessDialog(QtWidgets.QDialog):
    """Popup dialog with 5 tabbed panels for bed thickness analysis.
    Each tab is a standalone FigureCanvas with its own NavigationToolbar.
    """

    C = {'green': '#1A9C6E', 'orange': '#E8734A', 'blue': '#2D7DD2',
         'gold': '#F0B429', 'red': '#D64545', 'gray': '#666'}

    def __init__(self, parent, gray, detector, depth_array, segments):
        super().__init__(parent)
        self.gray = gray
        self.detector = detector
        self.depth_array = depth_array
        self.segments = segments
        self.n = len(gray)

        self.setWindowTitle("Bed Thickness Analysis")
        self.resize(1050, 750)
        self.setMinimumSize(800, 550)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Tab widget
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet("QTabWidget::pane{border:1px solid #ccc;} "
                                "QTabBar::tab{padding:5px 12px;}")
        layout.addWidget(self.tabs)

        # Stats bar at bottom
        self.lbl_stats = QtWidgets.QLabel("")
        self.lbl_stats.setStyleSheet("background:#f5f5f5; border:1px solid #ddd; "
                                      "padding:4px; font-size:11px; font-family:monospace;")
        layout.addWidget(self.lbl_stats)

        # Build 5 tabs
        self.tabs.addTab(self._make_tab("gray_panel", "Gray + Bounds"), "Gray+Bounds")
        self.tabs.addTab(self._make_tab("thickness_panel", "Thickness"), "Thickness")
        self.tabs.addTab(self._make_tab("cumulative_panel", "Cumulative"), "Cumulative")
        self.tabs.addTab(self._make_tab("histogram_panel", "Histogram"), "Histogram")
        self.tabs.addTab(self._make_tab("stats_panel", "Summary"), "Summary")

        self._draw_all()
        self.tabs.currentChanged.connect(lambda _: self._draw_current())

    def _make_tab(self, name, title):
        page = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout(page)
        l.setContentsMargins(0, 0, 0, 0)
        fig = Figure(figsize=(10, 7), dpi=100)
        fig.set_facecolor('#fafafa')
        canvas = FigureCanvasQTAgg(fig)
        canvas.setMinimumSize(600, 400)
        toolbar = NavigationToolbar(canvas, self)
        toolbar.setStyleSheet("QToolBar{border-bottom:1px solid #ddd; spacing:2px;}")
        l.addWidget(toolbar)
        l.addWidget(canvas, 1)
        setattr(self, f'_{name}_fig', fig)
        setattr(self, f'_{name}_canvas', canvas)
        return page

    def update_data(self, detector, depth_array=None):
        self.detector = detector
        if depth_array is not None:
            self.depth_array = depth_array
        self._draw_all()

    def _draw_all(self):
        self._draw_gray_panel()
        self._draw_thickness_panel()
        self._draw_cumulative_panel()
        self._draw_histogram_panel()
        self._draw_stats_panel()
        self._update_stats_label()

    def _draw_current(self):
        idx = self.tabs.currentIndex()
        methods = [self._draw_gray_panel, self._draw_thickness_panel,
                   self._draw_cumulative_panel, self._draw_histogram_panel,
                   self._draw_stats_panel]
        if 0 <= idx < len(methods):
            methods[idx]()
        self._update_stats_label()

    # ---- Panel (1): Gray + Boundaries ----

    def _draw_gray_panel(self):
        fig = getattr(self, '_gray_panel_fig', None)
        if fig is None:
            return
        fig.clear()
        ax = fig.add_subplot(111)
        gray = self.gray
        n = self.n
        boundaries = self.detector.boundaries if self.detector is not None else np.array([], dtype=int)

        ax.plot(gray, np.arange(n), color=self.C['blue'], linewidth=0.5)
        for ss, se, bi in self.segments:
            ax.axhspan(ss, se, alpha=0.04, color=BOX_COLORS[bi % len(BOX_COLORS)])
        for b in boundaries:
            ax.axhline(y=b, color=self.C['orange'], linewidth=0.8, alpha=0.7, linestyle='--')
        ax.invert_yaxis()
        n_beds = max(0, len(boundaries) - 1)
        ax.set_title(f'Gray Profile + {n_beds} Bed Boundaries',
                     fontsize=9, loc='left', fontweight='bold')
        ax.set_xlabel('Gray value', fontsize=7, color=self.C['gray'])
        ax.tick_params(labelsize=6)
        has_depth = self.depth_array is not None and len(self.depth_array) == n
        if has_depth:
            da = self.depth_array
            n_ticks = 10
            step = max(1, n // n_ticks)
            tick_rows = list(range(0, n, step))
            ax.set_yticks(tick_rows)
            ax.set_yticklabels([f"{da[i]:.2f}" for i in tick_rows], fontsize=5)
            ax.set_ylabel('Depth (m)', fontsize=7, color=self.C['gray'])
        ax.grid(True, alpha=0.12, linestyle='--')
        fig.canvas.draw()

    # ---- Panel (2): Bed Thickness Bars ----

    def _draw_thickness_panel(self):
        fig = getattr(self, '_thickness_panel_fig', None)
        if fig is None:
            return
        fig.clear()
        ax = fig.add_subplot(111)
        d = self.detector
        if d is None or d.n_beds < 1:
            ax.text(0.5, 0.5, 'No beds detected', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray')
            fig.canvas.draw()
            return

        boundaries = d.boundaries
        thickness = d.thickness
        n_beds = d.n_beds
        has_depth = self.depth_array is not None and len(self.depth_array) == self.n

        for i in range(n_beds):
            b0, b1 = boundaries[i], boundaries[i + 1]
            mid = int((b0 + b1) / 2)
            y_pos = self.depth_array[mid] if has_depth else mid
            # Bar height MUST be in Y-axis units: meters (depth) or pixels
            bar_height = thickness[i] if has_depth else (b1 - b0)
            ax.barh(y_pos, thickness[i], height=bar_height, color=self.C['blue'],
                    alpha=0.55, edgecolor=self.C['blue'], linewidth=0.3)

        ax.axvline(x=np.median(thickness), color=self.C['orange'], linewidth=0.8,
                   linestyle='--', label=f"median={np.median(thickness):.3f}")
        ax.invert_yaxis()
        ax.set_xlabel('Thickness (m)' if has_depth else 'Thickness (px)',
                      fontsize=7, color=self.C['gray'])
        ax.set_title(f'Bed Thickness ({n_beds} beds)', fontsize=9, loc='left',
                     fontweight='bold')
        ax.legend(fontsize=7, loc='lower right')
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.12, linestyle='--')
        if has_depth:
            ax.set_ylim(self.depth_array[-1], self.depth_array[0])
        else:
            ax.set_ylim(self.n, 0)
        fig.canvas.draw()

    # ---- Panel (3): Cumulative Deviation ----

    def _draw_cumulative_panel(self):
        fig = getattr(self, '_cumulative_panel_fig', None)
        if fig is None:
            return
        fig.clear()
        ax = fig.add_subplot(111)
        d = self.detector
        if d is None or d.n_beds < 2:
            ax.text(0.5, 0.5, 'Need >= 2 beds', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray')
            fig.canvas.draw()
            return

        thickness = d.thickness
        cum = np.cumsum(thickness)
        x_cum = np.arange(len(cum), dtype=np.float64)
        slope, intercept = np.polyfit(x_cum, cum, 1)
        trend = slope * x_cum + intercept
        cum_resid = cum - trend

        ax.plot(x_cum, cum_resid, color=self.C['gold'], linewidth=0.8)
        ax.fill_between(x_cum, 0, cum_resid, where=(cum_resid > 0),
                        color=self.C['orange'], alpha=0.25)
        ax.fill_between(x_cum, 0, cum_resid, where=(cum_resid < 0),
                        color=self.C['blue'], alpha=0.25)
        ax.axhline(y=0, color='gray', linewidth=0.4, linestyle='--')
        ax.set_xlabel('Bed number', fontsize=7, color=self.C['gray'])
        unit = 'm' if self.depth_array is not None else 'px'
        ax.set_ylabel(f'Deviation ({unit})', fontsize=7, color=self.C['gray'])
        ax.set_title('Cumulative Thickness Deviation', fontsize=9, loc='left',
                     fontweight='bold')
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.12, linestyle='--')
        fig.canvas.draw()

    # ---- Panel (4): Histogram ----

    def _draw_histogram_panel(self):
        fig = getattr(self, '_histogram_panel_fig', None)
        if fig is None:
            return
        fig.clear()
        ax = fig.add_subplot(111)
        d = self.detector
        if d is None or d.n_beds < 3:
            ax.text(0.5, 0.5, 'Need >= 3 beds', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray')
            fig.canvas.draw()
            return

        thickness = d.thickness
        n_beds = d.n_beds
        bins = max(6, min(30, int(np.sqrt(n_beds))))
        ax.hist(thickness, bins=bins, density=True, color=self.C['blue'],
                alpha=0.35, edgecolor=self.C['blue'], linewidth=0.3)
        ax.axvline(x=np.median(thickness), color=self.C['orange'], linewidth=0.8,
                   linestyle='--', label=f"median={np.median(thickness):.3f}")
        unit = 'm' if self.depth_array is not None else 'px'
        ax.set_xlabel(f'Thickness ({unit})', fontsize=7, color=self.C['gray'])
        ax.set_ylabel('Density', fontsize=7, color=self.C['gray'])
        ax.set_title(f'Thickness Distribution (n={n_beds})', fontsize=9, loc='left',
                     fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=6)
        fig.canvas.draw()

    # ---- Panel (5): Summary Text ----

    def _draw_stats_panel(self):
        fig = getattr(self, '_stats_panel_fig', None)
        if fig is None:
            return
        fig.clear()
        ax = fig.add_subplot(111)
        ax.axis('off')

        d = self.detector
        if d is None or d.n_beds < 1:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
            fig.canvas.draw()
            return

        s = d.stats()
        u = 'm' if self.depth_array is not None else 'px'
        text = (
            "Bed Thickness Statistics\n"
            "=======================\n\n"
            f"  Beds detected:  {s['n_beds']}\n"
            f"  Median:         {s['median']:.4f} {u}\n"
            f"  Mean:           {s['mean']:.4f} {u}\n"
            f"  Std dev:        {s['std']:.4f} {u}\n"
            f"  Min:            {s['min']:.4f} {u}\n"
            f"  Max:            {s['max']:.4f} {u}\n\n"
            "Parameter Guide\n"
            "===============\n\n"
            "  sigma     smaller -> finer beds\n"
            "  Prom      lower   -> more boundaries\n"
            "  Target:   median 3-5 cm for PL carbonate\n"
            "  Target:   n ~ 30-50 for 1.5 m section"
        )
        ax.text(0.08, 0.98, text, transform=ax.transAxes, fontsize=10,
                family='monospace', ha='left', va='top', color=self.C['gray'],
                linespacing=1.6)
        fig.canvas.draw()

    def _update_stats_label(self):
        d = self.detector
        if d is None or d.n_beds < 1:
            self.lbl_stats.setText("No beds detected")
            return
        s = d.stats()
        u = 'm' if self.depth_array is not None else 'px'
        self.lbl_stats.setText(
            f"  n = {s['n_beds']}  |  median = {s['median']:.3f} {u}  |  "
            f"mean = {s['mean']:.3f}  |  sd = {s['std']:.3f}  |  "
            f"range = [{s['min']:.3f}, {s['max']:.3f}]"
        )


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_bgr = None; self.boxes = []; self.profiles = {}; self.segments = []
        self.current_path = ""; self.calibrator = DepthCalibrator()
        self.illum_corrected = False; self.trend_lines = {}; self.red_mask = None
        self.inner_boxes = []  # inner ROI after border trim (for visualization)

        self.setWindowTitle("Gray Extractor v8.1")
        self.setMinimumSize(1200, 750); self.resize(1600, 900)
        self._setup_ui(); self._setup_toolbar(); self._setup_statusbar()
        self.setAcceptDrops(True); self._apply_style()

    # ── UI setup ──────────────────────────────

    def _setup_ui(self):
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central); root.setContentsMargins(4,4,4,4)

        # ── Left: image + depth bar ──
        left = QtWidgets.QVBoxLayout()
        self.image_view = ZoomableImageView()
        self.image_view.setMinimumWidth(450)
        left.addWidget(self.image_view, 1)

        # Depth calibration bar
        depth_bar = QtWidgets.QHBoxLayout()
        depth_bar.addWidget(QtWidgets.QLabel("Group:"))
        self.combo_group = QtWidgets.QComboBox()
        self.combo_group.setMinimumWidth(120)
        self.combo_group.currentIndexChanged.connect(self._on_group_selected)
        depth_bar.addWidget(self.combo_group)

        depth_bar.addWidget(QtWidgets.QLabel("  Top:"))
        self.spin_top_m = QtWidgets.QDoubleSpinBox()
        self.spin_top_m.setDecimals(3); self.spin_top_m.setRange(-10, 1000)
        self.spin_top_m.setSuffix(" m"); self.spin_top_m.setMinimumWidth(90)
        depth_bar.addWidget(self.spin_top_m)

        depth_bar.addWidget(QtWidgets.QLabel("  Bottom:"))
        self.spin_bot_m = QtWidgets.QDoubleSpinBox()
        self.spin_bot_m.setDecimals(3); self.spin_bot_m.setRange(-10, 1000)
        self.spin_bot_m.setSuffix(" m"); self.spin_bot_m.setMinimumWidth(90)
        depth_bar.addWidget(self.spin_bot_m)

        self.btn_calib = QtWidgets.QPushButton("Set")
        self.btn_calib.clicked.connect(self._apply_depth)
        self.btn_calib.setEnabled(False)
        self.btn_calib.setStyleSheet(
            "padding:2px 10px; background:#1A9C6E; color:white; border-radius:3px; font-weight:bold;")
        depth_bar.addWidget(self.btn_calib)

        self.btn_clear_depth = QtWidgets.QPushButton("Clear")
        self.btn_clear_depth.clicked.connect(self._clear_depth)
        self.btn_clear_depth.setEnabled(False)
        self.btn_clear_depth.setStyleSheet(
            "padding:2px 8px; background:#ccc; color:#333; border-radius:3px;")
        depth_bar.addWidget(self.btn_clear_depth)

        self.btn_reset_all = QtWidgets.QPushButton("Reset All")
        self.btn_reset_all.clicked.connect(self._reset_all_depths)
        self.btn_reset_all.setEnabled(False)
        self.btn_reset_all.setStyleSheet(
            "padding:2px 8px; background:#E74C3C; color:white; border-radius:3px;")
        depth_bar.addWidget(self.btn_reset_all)

        depth_bar.addStretch()
        left.addLayout(depth_bar)

        # Info table — replaces old info_label
        self.info_table = QtWidgets.QTableWidget()
        self.info_table.setColumnCount(4)
        self.info_table.setHorizontalHeaderLabels(["Group", "Depth", "Position", "Size"])
        self.info_table.horizontalHeader().setStretchLastSection(True)
        self.info_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.info_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.info_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.info_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.info_table.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.info_table.setAlternatingRowColors(True)
        self.info_table.setMaximumHeight(160)
        self.info_table.setStyleSheet(
            "QTableWidget{font-size:10px;border:1px solid #ddd;gridline-color:#eee;}"
            "QTableWidget::item{padding:1px 4px;}"
            "QHeaderView::section{background:#f0f0f0;font-weight:bold;font-size:10px;padding:2px;}")
        left.addWidget(self.info_table)

        # ── Right: plot + box list ──
        right = QtWidgets.QVBoxLayout()
        self.plot_widget = MultiChannelPlotWidget()
        self.box_list = BoxListWidget()
        self.box_list.order_changed.connect(self._on_order_changed)
        right.addWidget(self.plot_widget, 4)
        right.addWidget(self.box_list, 1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        lc = QtWidgets.QWidget(); lc.setLayout(left)
        rc = QtWidgets.QWidget(); rc.setLayout(right)
        splitter.addWidget(lc); splitter.addWidget(rc)
        splitter.setStretchFactor(0, 2); splitter.setStretchFactor(1, 3)
        root.addWidget(splitter)

    def _setup_toolbar(self):
        tb = self.addToolBar("Main"); tb.setMovable(False)

        a = QtGui.QAction("Open Image", self); a.setShortcut("Ctrl+O")
        a.triggered.connect(self._open_image); tb.addAction(a)
        tb.addSeparator()

        self.act_illum = QtGui.QAction("Illum Correct", self)
        self.act_illum.setShortcut("Ctrl+I"); self.act_illum.setCheckable(True)
        self.act_illum.toggled.connect(self._toggle_illum)
        self.act_illum.setEnabled(False); tb.addAction(self.act_illum)
        tb.addSeparator()

        self.act_csv = QtGui.QAction("Export CSV", self)
        self.act_csv.setShortcut("Ctrl+S"); self.act_csv.triggered.connect(self._export_csv)
        self.act_csv.setEnabled(False); tb.addAction(self.act_csv)

        self.act_plots = QtGui.QAction("Export Plots", self)
        self.act_plots.setShortcut("Ctrl+P"); self.act_plots.triggered.connect(self._export_plots)
        self.act_plots.setEnabled(False); tb.addAction(self.act_plots)
        tb.addSeparator()

        a2 = QtGui.QAction("Re-detect", self); a2.setShortcut("F5")
        a2.triggered.connect(self._detect_and_extract); tb.addAction(a2)

        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel("View: "))
        self.combo_view = QtWidgets.QComboBox()
        self.combo_view.addItems(["All 6", "Gray + L*/a*", "Gray only", "Thickness"])
        self.combo_view.setCurrentIndex(0)
        self.combo_view.currentIndexChanged.connect(self._on_view_changed)
        self.combo_view.setMinimumWidth(120)
        tb.addWidget(self.combo_view)

        tb.addSeparator()
        self.btn_detect_beds = QtWidgets.QPushButton("Detect Beds")
        self.btn_detect_beds.clicked.connect(self._run_thickness_detection)
        self.btn_detect_beds.setEnabled(False)
        self.btn_detect_beds.setStyleSheet("padding:2px 8px; background:#E8734A; color:white; border-radius:3px;")
        tb.addWidget(self.btn_detect_beds)

        tb.addWidget(QtWidgets.QLabel(" σ:"))
        self.spin_sigma = QtWidgets.QDoubleSpinBox()
        self.spin_sigma.setDecimals(1); self.spin_sigma.setRange(1, 20); self.spin_sigma.setValue(3.0)
        self.spin_sigma.setToolTip("Smoothing (sigma)\nSmaller = finer beds, Larger = only major boundaries")
        self.spin_sigma.setSuffix(" px"); self.spin_sigma.setMinimumWidth(70)
        self.spin_sigma.valueChanged.connect(self._run_thickness_detection)
        tb.addWidget(self.spin_sigma)

        tb.addWidget(QtWidgets.QLabel(" Prom:"))
        self.spin_prom = QtWidgets.QDoubleSpinBox()
        self.spin_prom.setDecimals(2); self.spin_prom.setRange(0.05, 50); self.spin_prom.setValue(3.0)
        self.spin_prom.setToolTip("Prominence (minimum valley depth)\nLower = more boundaries")
        self.spin_prom.setMinimumWidth(65)
        self.spin_prom.valueChanged.connect(self._run_thickness_detection)
        tb.addWidget(self.spin_prom)

        self.act_export_beds = QtGui.QAction("Export Beds", self)
        self.act_export_beds.setShortcut("Ctrl+B"); self.act_export_beds.triggered.connect(self._export_beds_csv)
        self.act_export_beds.setEnabled(False); tb.addAction(self.act_export_beds)

        # Hide thickness controls initially (only visible in Thickness view mode)
        self.btn_detect_beds.setVisible(False)
        self.spin_sigma.setVisible(False)
        self.spin_prom.setVisible(False)

    def _setup_statusbar(self):
        self.sb = self.statusBar()
        self.sb.showMessage("Ready — Open an image (Ctrl+O) or drag & drop")

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow{background:#fafafa}
            QToolBar{background:#fff;border-bottom:1px solid #ddd;padding:3px;spacing:5px}
            QStatusBar{background:#fff;border-top:1px solid #ddd;color:#444;font-size:11px}
            QDoubleSpinBox{padding:2px;font-size:11px}
        """)

    # ── Image loading ─────────────────────────

    def _open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All (*)")
        if path: self._load_image(path)

    def _load_image(self, path):
        self.image_bgr = cv2.imread(path)
        if self.image_bgr is None:
            try:
                from PIL import Image
                pil = Image.open(path)
                if pil.mode in ('RGBA','LA','P'): pil = pil.convert('RGB')
                self.image_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            except: pass
        if self.image_bgr is None:
            QtWidgets.QMessageBox.critical(self, "Error", f"Cannot open:\n{path}")
            return
        self.current_path = path
        self.sb.showMessage(f"Loaded: {os.path.basename(path)} ({self.image_bgr.shape[1]}×{self.image_bgr.shape[0]})")
        self._detect_and_extract()

    # ── Detection + extraction ───────────────

    def _detect_and_extract(self):
        if self.image_bgr is None: return
        self.boxes, self.red_mask = RedBoxDetector.detect_all(self.image_bgr)
        has = bool(self.boxes)
        self.act_illum.setEnabled(has); self.act_csv.setEnabled(has); self.act_plots.setEnabled(has)
        self.btn_calib.setEnabled(has); self.btn_clear_depth.setEnabled(False)
        self.btn_reset_all.setEnabled(False)
        self.btn_detect_beds.setEnabled(has)
        self.illum_corrected = False; self.act_illum.setChecked(False); self.trend_lines = {}
        if has:
            self._update_depth_bar()

        display = self.image_bgr.copy()
        if not self.boxes:
            cv2.putText(display, "X No red boxes found!", (40,60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            self.profiles = {ch: np.array([]) for ch in MultiChannelExtractor.CHANNELS}
            self.segments = []; self.box_list.set_boxes([])
            self.info_table.setRowCount(0); self.sb.showMessage("No red boxes")
        else:
            groups = MultiChannelExtractor.group_boxes(self.boxes)
            self._draw_boxes_on(display, groups)

            oh = sum(b[3] for b in self.boxes)
            self.inner_boxes = [MultiChannelExtractor.inner_roi(self.red_mask, b) for b in self.boxes]
            self.profiles, self.segments = MultiChannelExtractor.extract_all(self.image_bgr, self.boxes, self.red_mask)
            ih = sum(s[1]-s[0] for s in self.segments) if self.segments else 0
            tp = oh-ih
            self.box_list.set_boxes(self.boxes, groups)
            gray = self.profiles.get("gray", np.array([]))
            n = len(gray)
            self._update_info_table(groups, n, tp)
            self.sb.showMessage(f"{len(groups)} group(s) | {n} rows | 6 channels")

        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h_img, w_img, ch = display_rgb.shape
        qt_img = QtGui.QImage(display_rgb.data, w_img, h_img, ch*w_img, QtGui.QImage.Format.Format_RGB888)
        self.image_view.set_image(qt_img)
        self._refresh_plot()

    def _draw_boxes_on(self, display, groups):
        """Draw outer (red frame bounding rect) and inner (data region after border trim)
        boundaries on the image, with cropped zone shown as a shaded overlay."""
        for gi, grp in enumerate(groups):
            cb = self._h2b(BOX_COLORS[gi % len(BOX_COLORS)])
            # Lighter version of the group color for cropped-zone overlay
            cb_light = tuple(min(255, c + 80) for c in cb)
            for pos, bi in enumerate(grp):
                # ── Outer box: bounding rect of the red frame (thick line) ──
                x, y, w, h = self.boxes[bi]
                cv2.rectangle(display, (x, y), (x+w, y+h), cb, 3)

                # ── Inner box: data region after border trim (dashed thin line) ──
                if self.inner_boxes and bi < len(self.inner_boxes):
                    ix, iy, iw, ih = self.inner_boxes[bi]
                    # Draw inner rectangle with dashed lines
                    dash_len = 8
                    for _y in range(iy, iy+ih, dash_len*2):
                        y1 = _y; y2 = min(_y + dash_len, iy+ih)
                        cv2.line(display, (ix, y1), (ix, y2), cb_light, 1)       # left
                        cv2.line(display, (ix+iw, y1), (ix+iw, y2), cb_light, 1)  # right
                    for _x in range(ix, ix+iw, dash_len*2):
                        x1 = _x; x2 = min(_x + dash_len, ix+iw)
                        cv2.line(display, (x1, iy), (x2, iy), cb_light, 1)        # top
                        cv2.line(display, (x1, iy+ih), (x2, iy+ih), cb_light, 1)  # bottom

                    # ── Cropped zone overlay (semi-transparent shading) ──
                    overlay = display.copy()
                    # Top strip: y..iy
                    if iy > y:
                        cv2.rectangle(overlay, (x, y), (x+w, iy), cb_light, -1)
                    # Bottom strip: iy+ih..y+h
                    if iy+ih < y+h:
                        cv2.rectangle(overlay, (x, iy+ih), (x+w, y+h), cb_light, -1)
                    # Left strip: x..ix
                    if ix > x:
                        cv2.rectangle(overlay, (x, iy), (ix, iy+ih), cb_light, -1)
                    # Right strip: ix+iw..x+w
                    if ix+iw < x+w:
                        cv2.rectangle(overlay, (ix+iw, iy), (x+w, iy+ih), cb_light, -1)
                    display = cv2.addWeighted(display, 0.70, overlay, 0.30, 0)

                # ── Label ──
                lbl = f"G{gi+1}#{pos+1}" if len(grp)>1 else f"#{bi+1}"
                (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)
                cv2.rectangle(display, (x, y-lh-10), (x+lw+6, y), cb, -1)
                cv2.putText(display, lbl, (x+3, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        # Legend
        cv2.rectangle(display, (12, 12), (290, 62), (255,255,255), -1)
        cv2.rectangle(display, (12, 12), (290, 62), (180,180,180), 1)
        cv2.line(display, (20, 28), (60, 28), (100,100,100), 3)
        cv2.putText(display, "Outer (red frame)", (68, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80,80,80), 1)

        # Dashed line for inner
        for _x in range(20, 50, 12):
            cv2.line(display, (_x, 48), (_x+6, 48), (120,160,200), 1)
        cv2.putText(display, "Inner (data region)", (68, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80,80,80), 1)

        # Arrows between consecutive groups
        for gi in range(len(groups)-1):
            lb = groups[gi][-1]; fb = groups[gi+1][0]
            x1, y1, w1, h1 = self.boxes[lb]
            x2, y2, w2, h2 = self.boxes[fb]
            cv2.arrowedLine(display, (x1+w1//2, y1+h1), (x2+w2//2, y2), (80,80,80), 2, tipLength=0.04)

        # Depth labels for calibrated groups
        if self.calibrator.is_calibrated:
            for gi in range(len(groups)):
                d = self.calibrator.get_group(gi)
                if d:
                    rep_box = self.boxes[groups[gi][0]]
                    xb = rep_box[0] - 30
                    yb = rep_box[1] + rep_box[3] // 2
                    cv2.putText(display, f"G{gi+1}: {d[0]:.1f}–{d[1]:.1f}m",
                                (max(5, xb), yb), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,140,0), 2)

    # ── Depth array building (deduplicated) ──

    def _build_depth_array(self):
        """Build per-row depth array from calibrator + segments.
        Returns (depth_array or None)."""
        if not self.calibrator.is_calibrated or not self.segments:
            return None
        if not self.boxes:
            return None
        n = len(self.profiles.get("gray", []))
        if n == 0:
            return None
        depth_arr = np.zeros(n, dtype=np.float32)
        groups = MultiChannelExtractor.group_boxes(self.boxes)
        box2group = {}
        for gi, grp in enumerate(groups):
            for bi in grp:
                box2group[bi] = gi
        for ss, se, bi in self.segments:
            gi = box2group.get(bi, 0)
            h = se - ss
            local = np.arange(h, dtype=np.float64)
            depth_arr[ss:se] = self.calibrator.pixel_to_depth(local, h, gi)
        return depth_arr

    def _update_info_table(self, groups=None, n_rows=0, trimmed=0):
        """Populate the info table with group details.
        Call with no args to just refresh from current state."""
        if groups is None:
            if not self.boxes: return
            groups = MultiChannelExtractor.group_boxes(self.boxes)
        if self.profiles:
            gray = self.profiles.get("gray", np.array([]))
            if n_rows == 0:
                n_rows = len(gray)

        self.info_table.setRowCount(0)
        for gi, grp in enumerate(groups):
            row = self.info_table.rowCount(); self.info_table.insertRow(row)
            # Col 0: Group name
            if len(grp) == 1:
                gname = f"G{gi+1}"
            else:
                gname = f"G{gi+1} ({len(grp)} parallel)"
            self.info_table.setItem(row, 0, QtWidgets.QTableWidgetItem(gname))
            # Col 1: Depth
            d = self.calibrator.get_group(gi)
            if d:
                depth_str = f"{d[0]:.3f} – {d[1]:.3f} m"
            else:
                depth_str = "—"
            item = QtWidgets.QTableWidgetItem(depth_str)
            item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.info_table.setItem(row, 1, item)
            # Col 2: Position
            if len(grp) == 1:
                x, y, w, h = self.boxes[grp[0]]
                pos_str = f"({x}, {y})"
            else:
                pos_str = f"{len(grp)} boxes"
            item = QtWidgets.QTableWidgetItem(pos_str)
            item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.info_table.setItem(row, 2, item)
            # Col 3: Size
            if len(grp) == 1:
                x, y, w, h = self.boxes[grp[0]]
                size_str = f"{w}×{h}"
            else:
                max_h = max(self.boxes[bi][3] for bi in grp)
                size_str = f"avg → {max_h}"
            item = QtWidgets.QTableWidgetItem(size_str)
            item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.info_table.setItem(row, 3, item)

        # Summary row
        row = self.info_table.rowCount(); self.info_table.insertRow(row)
        summary = f"{len(self.boxes)} boxes, {len(groups)} groups, {n_rows} rows"
        if trimmed:
            summary += f"  (trimmed: {trimmed} px)"
        item = QtWidgets.QTableWidgetItem(summary)
        item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        font = item.font(); font.setItalic(True); item.setFont(font)
        item.setForeground(QtGui.QColor("#888"))
        self.info_table.setItem(row, 0, item)
        self.info_table.setSpan(row, 0, 1, 4)

    def _refresh_plot(self):
        profiles_to_plot = self.profiles.copy(); trend = {}
        if self.illum_corrected and self.profiles:
            for ch in MultiChannelExtractor.CHANNELS:
                if ch in self.profiles and len(self.profiles[ch])>0:
                    profiles_to_plot[ch], trend[ch] = IlluminationCorrector.correct(self.profiles[ch])
            self.trend_lines = trend
        depth_arr = self._build_depth_array()
        self.plot_widget.plot(profiles_to_plot, self.segments, depth_arr,
                              illum_corrected=self.illum_corrected, trend_lines=self.trend_lines)

    # ── Depth calibration ────────────────────

    def _update_depth_bar(self):
        groups = MultiChannelExtractor.group_boxes(self.boxes)
        self.combo_group.blockSignals(True)
        self.combo_group.clear()
        for gi, grp in enumerate(groups):
            labels = ','.join(str(b+1) for b in grp)
            self.combo_group.addItem(f"Group {gi+1} (Box {labels})")
        self.combo_group.blockSignals(False)
        self._on_group_selected(0)

    def _on_group_selected(self, idx):
        if idx < 0: return
        d = self.calibrator.get_group(idx)
        self.spin_top_m.blockSignals(True)
        self.spin_bot_m.blockSignals(True)
        if d:
            self.spin_top_m.setValue(d[0]); self.spin_bot_m.setValue(d[1])
        else:
            self.spin_top_m.setValue(0); self.spin_bot_m.setValue(1)
        self.spin_top_m.blockSignals(False)
        self.spin_bot_m.blockSignals(False)
        # Enable clear button if this group has a depth set
        self.btn_clear_depth.setEnabled(d is not None)

    def _apply_depth(self):
        """Set depth range for the selected group.
        Simple: just save the calibration and refresh. No merge logic."""
        if not self.boxes: return
        gi = self.combo_group.currentIndex()
        if gi < 0: return
        tm = self.spin_top_m.value(); bm = self.spin_bot_m.value()
        if tm >= bm:
            QtWidgets.QMessageBox.warning(self, "Depth", "Top must be < bottom depth")
            return
        self.calibrator.set_group(gi, tm, bm)
        self.btn_clear_depth.setEnabled(True)
        self.btn_reset_all.setEnabled(True)

        # ── Warn if adjacent groups have depth gaps ──
        groups = MultiChannelExtractor.group_boxes(self.boxes)
        all_cal = sorted([g for g in range(len(groups)) if self.calibrator.get_group(g)],
                         key=lambda g: self.calibrator.get_group(g)[0])

        warnings = []
        for a, b in zip(all_cal, all_cal[1:]):
            da = self.calibrator.get_group(a); db = self.calibrator.get_group(b)
            gap = db[0] - da[1]
            if abs(gap) > 0.001:  # gap > 1mm
                direction = "gap" if gap > 0 else "OVERLAP"
                warnings.append(f"  G{a+1}→G{b+1}: {abs(gap):.3f}m {direction}")

        self._refresh_plot()
        self._redraw_image()
        self._update_info_table()

        info = f"Group {gi+1}: {tm:.3f} - {bm:.3f} m"
        if warnings:
            info += "\n⚠ " + "\n⚠ ".join(warnings)
        self.sb.showMessage(info)

    def _clear_depth(self):
        """Clear depth calibration for the selected group."""
        gi = self.combo_group.currentIndex()
        if gi < 0: return
        self.calibrator.remove_group(gi)
        self.btn_clear_depth.setEnabled(False)
        # Update Reset All state
        has_any = any(self.calibrator.get_group(g) for g in range(
            len(MultiChannelExtractor.group_boxes(self.boxes))))
        self.btn_reset_all.setEnabled(has_any)
        self.spin_top_m.setValue(0); self.spin_bot_m.setValue(1)
        self._refresh_plot()
        self._redraw_image()
        self.sb.showMessage(f"Depth cleared for Group {gi+1}")

    def _reset_all_depths(self):
        """Clear ALL group depth calibrations at once."""
        if not self.calibrator.is_calibrated: return
        n_groups = len(MultiChannelExtractor.group_boxes(self.boxes))
        self.calibrator.clear()
        self.btn_clear_depth.setEnabled(False)
        self.btn_reset_all.setEnabled(False)
        self.spin_top_m.setValue(0); self.spin_bot_m.setValue(1)
        self._refresh_plot()
        self._redraw_image()
        self._update_info_table()
        self.sb.showMessage(f"All {n_groups} group depth calibrations cleared")

    # ── Illumination toggle ──────────────────

    def _toggle_illum(self, checked):
        self.illum_corrected = checked; self._refresh_plot()
        self.sb.showMessage("Illumination correction " + ("ON" if checked else "OFF"))

    # ── View toggle ──────────────────────────

    def _on_view_changed(self, idx):
        modes = ["all", "gray_lab", "gray", "thickness"]
        mode = modes[idx] if 0 <= idx < len(modes) else "all"
        if mode == "thickness":
            # Popup dialog instead of inline view
            self._show_thickness_dialog()
            # Reset combo back to previous view
            self.combo_view.blockSignals(True)
            self.combo_view.setCurrentIndex(0)
            self.combo_view.blockSignals(False)
            return
        self.plot_widget.set_view_mode(mode)
        # Hide thickness-specific controls when not in thickness mode
        self.spin_sigma.setVisible(False)
        self.spin_prom.setVisible(False)
        self.btn_detect_beds.setVisible(False)
        self.act_export_beds.setVisible(False)

    # ── Thickness dialog ────────────────────────

    def _show_thickness_dialog(self):
        gray = self.profiles.get("gray", np.array([]))
        if len(gray) == 0:
            self.sb.showMessage("No grayscale data")
            return
        depth_arr = self._build_depth_array()
        sigma = self.spin_sigma.value()
        prom = self.spin_prom.value()

        detector = BedBoundaryDetector()
        detector.detect(gray, depth=depth_arr, smooth_sigma=sigma, prominence=prom)

        # Store for export and parameter updates
        self._detector = detector
        self._depth_arr = depth_arr

        # Show toolbar controls
        self.spin_sigma.setVisible(True)
        self.spin_prom.setVisible(True)
        self.btn_detect_beds.setVisible(True)
        self.act_export_beds.setVisible(True)
        self.act_export_beds.setEnabled(detector.n_beds >= 1)

        # Create and show dialog (non-modal so user can still adjust sigma/prom)
        dlg = ThicknessDialog(self, gray, detector, depth_arr, self.segments)
        dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
        dlg.finished.connect(self._on_thickness_dialog_closed)
        dlg.show()
        self._thickness_dialog = dlg

    def _on_thickness_dialog_closed(self):
        self.spin_sigma.setVisible(False)
        self.spin_prom.setVisible(False)
        self.btn_detect_beds.setVisible(False)
        self.act_export_beds.setVisible(False)
        if self._thickness_dialog is not None:
            self._thickness_dialog.deleteLater()
        self._thickness_dialog = None

    def _run_thickness_detection(self):
        """Re-run detection (called when sigma/prom changes) and update dialog."""
        dlg = getattr(self, '_thickness_dialog', None)
        if dlg is None or not dlg.isVisible():
            return
        gray = self.profiles.get("gray", np.array([]))
        if len(gray) == 0:
            return
        depth_arr = self._build_depth_array()
        sigma = self.spin_sigma.value()
        prom = self.spin_prom.value()

        detector = BedBoundaryDetector()
        detector.detect(gray, depth=depth_arr, smooth_sigma=sigma, prominence=prom)
        self._detector = detector
        self._depth_arr = depth_arr

        dlg.update_data(detector, depth_arr)
        stats = detector.stats()
        unit = "m" if depth_arr is not None else "px"
        self.sb.showMessage(
            f"Bed detection: {detector.n_beds} beds, "
            f"median thickness={stats['median']:.3f} {unit}, "
            f"range=[{stats['min']:.3f}, {stats['max']:.3f}]"
        )
        self.act_export_beds.setEnabled(detector.n_beds >= 1)

    def _export_beds_csv(self):
        """Export bed boundary and thickness data as CSV."""
        detector = getattr(self, '_detector', None)
        if detector is None or detector.n_beds < 1:
            self.sb.showMessage("No beds detected — run Thickness view first")
            return
        depth_array = getattr(self, '_depth_arr', None)

        dn = "beds.csv"
        if self.current_path:
            dn = f"{os.path.splitext(os.path.basename(self.current_path))[0]}_beds.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Bed Data", dn, "CSV (*.csv);;All (*)")
        if not path: return

        # Build rows
        boundaries_depth = detector.boundary_depths
        thickness = detector.thickness
        n_beds = detector.n_beds
        has_depth = depth_array is not None

        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            hdrs = ["bed_id", "top", "bottom", "thickness"]
            if has_depth:
                hdrs[1] = "top_m"; hdrs[2] = "bottom_m"; hdrs[3] = "thickness_m"
            w.writerow(hdrs)
            if has_depth:
                for i in range(n_beds):
                    w.writerow([i+1,
                               round(float(boundaries_depth[i]), 4),
                               round(float(boundaries_depth[i+1]), 4),
                               round(float(thickness[i]), 4)])
            else:
                for i in range(n_beds):
                    w.writerow([i+1,
                               int(boundaries_depth[i]),
                               int(boundaries_depth[i+1]),
                               round(float(thickness[i]), 2)])

        self.sb.showMessage(f"Exported {n_beds} beds: {os.path.basename(path)}")

    # ── Box reorder ───────────────────────────

    def _on_order_changed(self):
        self.boxes = self.box_list.get_boxes()
        if self.image_bgr is not None and self.boxes:
            self.inner_boxes = [MultiChannelExtractor.inner_roi(
                getattr(self, 'red_mask', None), b) for b in self.boxes]
            self.profiles, self.segments = MultiChannelExtractor.extract_all(
                self.image_bgr, self.boxes, getattr(self, 'red_mask', None))
            self._update_depth_bar()   # refresh group dropdown after reorder
            self._refresh_plot()
            display = self.image_bgr.copy()
            groups = MultiChannelExtractor.group_boxes(self.boxes)
            self._draw_boxes_on(display, groups)
            drgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            hh, ww, chh = drgb.shape
            qi = QtGui.QImage(drgb.data, ww, hh, chh*ww, QtGui.QImage.Format.Format_RGB888)
            self.image_view.set_image(qi)
            self.sb.showMessage(f"Order updated — {len(self.boxes)} box(es)")

    # ── Export CSV ────────────────────────────

    def _export_csv(self):
        gd = self.profiles.get("gray", np.array([]))
        if len(gd)==0: return
        dn = "gray_profile.csv"
        if self.current_path:
            dn = f"{os.path.splitext(os.path.basename(self.current_path))[0]}_profile.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export CSV", dn, "CSV (*.csv);;All (*)")
        if not path: return

        # Box ID per row
        bid = np.full(len(gd), -1, dtype=int)
        for ss, se, bi in self.segments:
            bid[ss:se] = bi+1

        # Depth column
        dm = self._build_depth_array()

        # Compute illumination-corrected values (always)
        corrected = {}
        for ch in MultiChannelExtractor.CHANNELS:
            if ch in self.profiles and len(self.profiles[ch]) > 0:
                corrected[ch], _ = IlluminationCorrector.correct(self.profiles[ch])

        # Build headers: row_px, depth_m, box_id, [raw channels], [corrected channels]
        hdrs = ["row_px"]
        if dm is not None: hdrs.append("depth_m")
        hdrs.append("box_id")
        for ch in MultiChannelExtractor.CHANNELS:
            hdrs.append(ch)
        for ch in MultiChannelExtractor.CHANNELS:
            hdrs.append(f"{ch}_corr")

        n = len(gd)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f); w.writerow(hdrs)
            for i in range(n):
                row = [i]
                if dm is not None: row.append(round(float(dm[i]), 4))
                row.append(int(bid[i]))
                # Raw values
                for ch in MultiChannelExtractor.CHANNELS:
                    row.append(round(float(self.profiles[ch][i]), 4))
                # Corrected values
                for ch in MultiChannelExtractor.CHANNELS:
                    row.append(round(float(corrected[ch][i]), 4))
                w.writerow(row)

        self.sb.showMessage(f"CSV exported: {os.path.basename(path)} ({n} rows, 6 raw + 6 corrected channels)")

    # ── Export plots ──────────────────────────

    def _export_plots(self):
        if not self.profiles or len(self.profiles.get("gray",[]))==0: return
        dn = "plots.png"
        if self.current_path:
            dn = f"{os.path.splitext(os.path.basename(self.current_path))[0]}_plots.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Plots", dn,
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;All (*)")
        if not path: return
        self.plot_widget.save_plots(path)
        self.sb.showMessage(f"Plots exported: {os.path.basename(path)}")

    # ── Drag & drop ───────────────────────────

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.accept()
        else: e.ignore()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if p.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff','.tif')):
                self._load_image(p); break

    # ── Image redraw (unified) ────────────────

    def _redraw_image(self):
        """Redraw the annotated image with boxes, groups, and depth labels."""
        if self.image_bgr is None: return
        display = self.image_bgr.copy()
        if not self.boxes: return
        groups = MultiChannelExtractor.group_boxes(self.boxes)
        self._draw_boxes_on(display, groups)
        drgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        hh, ww, chh = drgb.shape
        qi = QtGui.QImage(drgb.data, ww, hh, chh*ww, QtGui.QImage.Format.Format_RGB888)
        self.image_view.set_image(qi)

    @staticmethod
    def _h2b(h): return (int(h[5:7],16), int(h[3:5],16), int(h[1:3],16))


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    w = MainWindow(); w.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
