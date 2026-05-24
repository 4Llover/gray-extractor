"""
Gray Extractor v5.0 — Zoomable Image + Depth Ruler + Plot Export
================================================================
New in v5.0:
  - QGraphicsView-based zoomable/pannable image viewer
  - Simplified depth calibration (toolbar spinboxes)
  - Depth ruler on profile plots with key marker lines
  - Export 6-channel plots as PNG/SVG/PDF
  - Depth-aware overlap verification
  - All v4.0 features retained
"""
import sys, os, csv
import numpy as np, cv2
from PyQt6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# ═══════════════════════════════════════════════
#  Multi-channel extraction engine (unchanged)
# ═══════════════════════════════════════════════

class MultiChannelExtractor:
    CHANNELS = ["gray", "redness", "greenness", "blueness", "L_star", "a_star"]

    @staticmethod
    def inner_roi(red_mask, outer_rect, min_frac=0.05, max_scan=50):
        """Scan inward from each edge using FULL row/column to find
        where the red border ends. Designed for thick (>8px), straight borders.

        For each edge, scans inward until <5% of the row/column is red.
        Proportional fallback: max(4, w/30) px when can't detect border.
        """
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
                else:  # 'right'
                    frac = np.count_nonzero(roi[:, w-1-i]) / max(h, 1)
                if frac < min_frac:
                    n = i; break
            return n

        ti = _scan('top', max_scan);   bi = _scan('bottom', max_scan)
        li = _scan('left', max_scan);  ri = _scan('right', max_scan)

        # Proportional fallback
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
    """Per-group depth calibration. Each box group (strata segment)
    gets its own top/bottom depth in meters. When multiple groups are
    calibrated, overlapping depth ranges are averaged."""
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
        """Convert local pixel row (0..group_height) to depth for a specific group.
        Returns float or array."""
        if not self.is_calibrated or group_idx not in self.group_depth:
            return np.asarray(local_row, dtype=np.float32)
        top_m, bottom_m = self.group_depth[group_idx]
        y = np.asarray(local_row, dtype=np.float64)
        frac = y / max(group_height, 1)
        return (top_m + frac * (bottom_m - top_m)).astype(np.float32)

    def clear(self): self.__init__()


class RedBoxDetector:
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
#  Colors
# ═══════════════════════════════════════════════

BOX_COLORS = ['#2D7DD2','#E8734A','#1A9C6E','#F0B429','#8E44AD','#E74C3C','#27AE60','#2980B9']
CHANNEL_COLORS = {"gray":'#333',"redness":'#E74C3C',"greenness":'#27AE60',"blueness":'#2980B9',"L_star":'#555',"a_star":'#E8734A'}
CHANNEL_LABELS = {"gray":"Gray (0–255)","redness":"R/(R+G+B)","greenness":"G/(R+G+B)","blueness":"B/(R+G+B)","L_star":"CIE L*","a_star":"CIE a*"}


# ═══════════════════════════════════════════════
#  Multi-channel plot widget (v5.2: 1-column layout)
# ═══════════════════════════════════════════════

class MultiChannelPlotWidget(FigureCanvasQTAgg):
    """Asymmetric 3-column layout:
       Left (wide): gray only
       Middle (medium): L* + a* stacked
       Right (narrow): redness, greenness, blueness stacked"""

    LAYOUT = {
        "gray":       (0, 0.40, 0.58),   # (left, width_ratio, height)
        "L_star":     (0.42, 0.28, 0.47),
        "a_star":     (0.42, 0.28, 0.47),
        "redness":    (0.72, 0.26, 0.30),
        "greenness":  (0.72, 0.26, 0.30),
        "blueness":   (0.72, 0.26, 0.30),
    }

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(14, 9), dpi=100)
        self.figure.set_facecolor('#fafafa')
        super().__init__(self.figure)
        self.setParent(parent)
        self.axes = {}
        self._create_subplots()
        self._depth_data = None

    def _create_subplots(self):
        # Left: gray (full height)
        self.axes["gray"] = self.figure.add_axes((0.04, 0.06, 0.34, 0.90))
        # Middle: L_star (top half), a_star (bottom half)
        self.axes["L_star"] = self.figure.add_axes((0.42, 0.53, 0.24, 0.43))
        self.axes["a_star"] = self.figure.add_axes((0.42, 0.06, 0.24, 0.43))
        # Right: redness, greenness, blueness
        self.axes["redness"]   = self.figure.add_axes((0.70, 0.68, 0.27, 0.28))
        self.axes["greenness"] = self.figure.add_axes((0.70, 0.37, 0.27, 0.28))
        self.axes["blueness"]  = self.figure.add_axes((0.70, 0.06, 0.27, 0.28))

    def _draw_one(self, ax, ch, data, n, segments, depth_array, illum_corrected, trend_lines):
        color = CHANNEL_COLORS[ch]
        y_vals = np.arange(n)
        for ss, se, bi in segments:
            ax.axhspan(ss, se, alpha=0.06, color=BOX_COLORS[bi % len(BOX_COLORS)])
        ax.fill_betweenx(y_vals, data, data.min(), alpha=0.15, color=color)
        ax.plot(data, y_vals, color=color, linewidth=0.5)
        if illum_corrected and trend_lines and ch in trend_lines:
            ax.plot(trend_lines[ch], y_vals, '--', color='#999', linewidth=0.8, alpha=0.7)
        ax.axvline(float(np.mean(data)), color=color, linestyle=':', linewidth=0.8, alpha=0.6)
        ax.invert_yaxis()
        title_size = 9 if ch == "gray" else 8
        label_size = 7 if ch == "gray" else 6
        ax.set_xlabel(CHANNEL_LABELS.get(ch, ch), fontsize=label_size)
        ax.set_title(ch, fontsize=title_size, fontweight='bold', color=color)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.tick_params(labelsize=label_size - 1)

        # Depth ruler
        if depth_array is not None and len(depth_array) == n:
            n_ticks = 10 if ch == "gray" else 6
            step = max(1, n // n_ticks)
            tick_rows = list(range(0, n, step))
            tick_labels = [f"{depth_array[i]:.2f}" for i in tick_rows]
            ax.set_yticks(tick_rows)
            ax.set_yticklabels(tick_labels, fontsize=5)
            if ch == "gray":
                ax.set_ylabel("Depth (m)", fontsize=7)
            # Segment boundary markers — fix: label beside the LINE
            for ss, se, bi in segments:
                if ss > 0:
                    ax.axhline(ss, color='#F0B429', linestyle='-', linewidth=1.0, alpha=0.5)
                    # Place label at the golden line position, not at inverted-axes position
                    ax.annotate(f'{depth_array[min(ss, n-1)]:.2f}m',
                                xy=(0.98, ss), xycoords=('axes fraction', 'data'),
                                fontsize=5, color='#F0B429', ha='right', va='center',
                                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7, edgecolor='none'))

    def plot(self, profiles, segments, depth_array=None, illum_corrected=False, trend_lines=None):
        self._depth_data = (profiles, segments, depth_array, illum_corrected, trend_lines)
        for ch in MultiChannelExtractor.CHANNELS:
            ax = self.axes[ch]; ax.clear()
            data = profiles.get(ch)
            if data is None or len(data) == 0:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center',
                        fontsize=10, color='gray')
                ax.set_title(ch, fontsize=8, fontweight='bold')
                continue
            self._draw_one(ax, ch, data, len(data), segments, depth_array, illum_corrected, trend_lines)
        self.draw()

    def save_plots(self, path):
        self.figure.savefig(path, dpi=200, bbox_inches='tight', facecolor='#fafafa')


# ═══════════════════════════════════════════════
#  Zoomable image viewer (QGraphicsView)
# ═══════════════════════════════════════════════

class ZoomableImageView(QtWidgets.QGraphicsView):
    """QGraphicsView with mouse-wheel zoom and click-drag pan."""

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
#  Overlap merge dialog
# ═══════════════════════════════════════════════

class OverlapMergeDialog(QtWidgets.QDialog):
    """Shown when two box groups have overlapping depth ranges.
    User chooses: mean (default), keep A, keep B."""

    def __init__(self, overlaps, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Depth Overlap Detected")
        self.setMinimumWidth(450)
        self._choice = "mean"

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(
            "<b>Overlapping depth ranges found between groups!</b>"))
        layout.addWidget(QtWidgets.QLabel(
            "This is expected when you intentionally overlap boxes to\n"
            "connect discontinuous outcrop sections."))

        for ga, gb, om in overlaps:
            layout.addWidget(QtWidgets.QLabel(
                f"  Group {ga+1} & Group {gb+1}: overlap {om:.2f} m"))

        layout.addWidget(QtWidgets.QLabel(
            "<br><b>How to handle the overlap?</b>"))
        self.radio_mean = QtWidgets.QRadioButton("Take Mean of both (recommended)")
        self.radio_mean.setChecked(True)
        self.radio_a = QtWidgets.QRadioButton("Keep Group A only")
        self.radio_b = QtWidgets.QRadioButton("Keep Group B only")

        bg = QtWidgets.QButtonGroup(self)
        bg.addButton(self.radio_mean); bg.addButton(self.radio_a); bg.addButton(self.radio_b)

        layout.addWidget(self.radio_mean)
        layout.addWidget(self.radio_a)
        layout.addWidget(self.radio_b)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _on_accept(self):
        if self.radio_mean.isChecked(): self._choice = "mean"
        elif self.radio_a.isChecked(): self._choice = "keep_a"
        else: self._choice = "keep_b"
        self.accept()

    def choice(self):
        return self._choice


# ═══════════════════════════════════════════════
#  Overlap merge engine
# ═══════════════════════════════════════════════

class OverlapMerger:
    """Merge overlapping depth profiles into a continuous sequence."""

    @staticmethod
    def merge(profiles, segments, calibrator, groups, method="mean"):
        """Build a single continuous profile from per-group depth data.
        Returns (merged_profiles, merged_segments, depth_array, merged_info)."""
        if not calibrator.is_calibrated:
            return profiles, segments, None, {}

        # Build per-row depth + group membership
        gray = profiles.get("gray", np.array([]))
        n = len(gray)
        depth = np.zeros(n, dtype=np.float32)
        grp_id = np.full(n, -1, dtype=int)

        box2grp = {}
        for gi, grp in enumerate(groups):
            for bi in grp:
                box2grp[bi] = gi

        for ss, se, bi in segments:
            gi = box2grp.get(bi, 0)
            h = se - ss
            local = np.arange(h, dtype=np.float64)
            depth[ss:se] = calibrator.pixel_to_depth(local, h, gi)
            grp_id[ss:se] = gi

        # Detect overlap zones
        sorted_groups = sorted(
            [gi for gi in range(len(groups)) if calibrator.get_group(gi)],
            key=lambda gi: calibrator.get_group(gi)[0])

        overlap_info = {}
        merged_results = {ch: [] for ch in MultiChannelExtractor.CHANNELS}
        merged_segments = []
        offset = 0

        for a, b in zip(sorted_groups, sorted_groups[1:]):
            da = calibrator.get_group(a)
            db = calibrator.get_group(b)
            overlap_start = max(da[0], db[0])
            overlap_end = min(da[1], db[1])
            if overlap_end <= overlap_start:
                continue
            overlap_info[(a, b)] = overlap_end - overlap_start

        # Simple concatenation with overlap handling
        ordered_segments = []
        for gi in sorted_groups:
            rows = np.where(grp_id == gi)[0]
            if len(rows) == 0:
                continue
            ss, se = int(rows[0]), int(rows[-1]) + 1
            ordered_segments.append((gi, ss, se, depth[ss], depth[se - 1]))

        # Process overlaps: for each adjacent pair, merge the overlap zone
        final_data = {ch: [] for ch in MultiChannelExtractor.CHANNELS}
        final_depth = []
        final_segs = []
        off = 0

        i = 0
        while i < len(ordered_segments):
            gi, ss, se, d_start, d_end = ordered_segments[i]
            data_block = {ch: profiles[ch][ss:se].copy() for ch in MultiChannelExtractor.CHANNELS}
            depth_block = depth[ss:se].copy()

            if i + 1 < len(ordered_segments):
                gj, ss2, se2, d2_start, d2_end = ordered_segments[i + 1]
                key = (gi, gj) if (gi, gj) in overlap_info else (gj, gi)
                ov = overlap_info.get(key, 0)
                if ov > 0:
                    # Find overlap rows in each block
                    # In gi block: rows where depth >= d2_start
                    # In gj block: rows where depth <= d_end
                    mask_i = depth_block >= d2_start
                    mask_j = depth[ss2:se2] <= d_end

                    if method == "mean":
                        # Average the overlapping portions
                        for ch in MultiChannelExtractor.CHANNELS:
                            data_i = profiles[ch][ss:se][mask_i]
                            data_j = profiles[ch][ss2:se2][mask_j]
                            mn = min(len(data_i), len(data_j))
                            if mn > 0:
                                avg = (data_i[:mn] + data_j[:mn]) / 2.0
                                data_block[ch][mask_i][:mn] = avg
                        # Remove the overlap portion from block j
                        ordered_segments[i + 1] = (gj, ss2 + sum(mask_j),
                                                    se2, depth[se2 - 1], d2_end)

            # Append block i
            for ch in MultiChannelExtractor.CHANNELS:
                final_data[ch].append(data_block[ch])
            final_depth.append(depth_block)
            se_len = len(data_block["gray"])
            final_segs.append((off, off + se_len, gi))
            off += se_len
            i += 1

        if not final_data["gray"]:
            return profiles, segments, depth, overlap_info

        merged = {ch: np.concatenate(final_data[ch]) for ch in MultiChannelExtractor.CHANNELS}
        merged_depth = np.concatenate(final_depth)

        return merged, final_segs, merged_depth, overlap_info


# ═══════════════════════════════════════════════
#  Main window
# ═══════════════════════════════════════════════

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_bgr = None; self.boxes = []; self.profiles = {}; self.segments = []
        self.current_path = ""; self.calibrator = DepthCalibrator()
        self.illum_corrected = False; self.trend_lines = {}; self.red_mask = None
        self._has_merged = False; self._merge_choice = "mean"
        self.raw_profiles = None; self.raw_segments = None; self._merged_depth = None

        self.setWindowTitle("Gray Extractor v5.0")
        self.setMinimumSize(1200, 750); self.resize(1600, 900)
        self._setup_ui(); self._setup_toolbar(); self._setup_statusbar()
        self.setAcceptDrops(True); self._apply_style()

    # ── UI ────────────────────────────────────

    def _setup_ui(self):
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central); root.setContentsMargins(4,4,4,4)

        # Left: zoomable image
        left = QtWidgets.QVBoxLayout()
        self.image_view = ZoomableImageView()
        self.image_view.setMinimumWidth(450)
        left.addWidget(self.image_view, 1)

        # Quick depth bar below image — per-group calibration
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
        self.btn_calib.setStyleSheet("padding:2px 8px; background:#1A9C6E; color:white; border-radius:3px;")
        depth_bar.addWidget(self.btn_calib)
        depth_bar.addStretch()
        left.addLayout(depth_bar)

        self.info_label = QtWidgets.QLabel("")
        self.info_label.setStyleSheet("color:#555;font-size:11px;padding:2px;")
        left.addWidget(self.info_label)

        # Right: plot + box list
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
        self.btn_calib.setEnabled(has)
        self.illum_corrected = False; self.act_illum.setChecked(False); self.trend_lines = {}
        if has:
            self._update_depth_bar()

        display = self.image_bgr.copy()
        if not self.boxes:
            cv2.putText(display, "X No red boxes found!", (40,60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            self.profiles = {ch: np.array([]) for ch in MultiChannelExtractor.CHANNELS}
            self.segments = []; self.box_list.set_boxes([])
            self.info_label.setText("No red boxes found."); self.sb.showMessage("No red boxes")
        else:
            groups = MultiChannelExtractor.group_boxes(self.boxes)
            for gi, grp in enumerate(groups):
                cb = self._h2b(BOX_COLORS[gi%len(BOX_COLORS)])
                for pos, bi in enumerate(grp):
                    x,y,w,h = self.boxes[bi]
                    cv2.rectangle(display, (x,y), (x+w,y+h), cb, 3)
                    lbl = f"G{gi+1}#{pos+1}" if len(grp)>1 else f"#{bi+1}"
                    (lw,lh),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)
                    cv2.rectangle(display, (x,y-lh-10), (x+lw+6,y), cb, -1)
                    cv2.putText(display, lbl, (x+3,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            for gi in range(len(groups)-1):
                lb = groups[gi][-1]; fb = groups[gi+1][0]
                x1,y1,w1,h1 = self.boxes[lb]; x2,y2,w2,h2 = self.boxes[fb]
                cv2.arrowedLine(display, (x1+w1//2,y1+h1), (x2+w2//2,y2), (80,80,80), 2, tipLength=0.04)
            if self.calibrator.is_calibrated:
                for gi in range(len(groups)):
                    d = self.calibrator.get_group(gi)
                    if d:
                        # Show group label on image
                        grp_boxes = groups[gi]
                        rep_box = self.boxes[grp_boxes[0]]
                        xb, yb = rep_box[0] - 30, rep_box[1] + rep_box[3] // 2
                        cv2.putText(display, f"G{gi+1}: {d[0]:.1f}–{d[1]:.1f}m",
                                    (max(5, xb), yb),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 140, 0), 2)

            oh = sum(b[3] for b in self.boxes)
            self.profiles, self.segments = MultiChannelExtractor.extract_all(self.image_bgr, self.boxes, self.red_mask)
            ih = sum(s[1]-s[0] for s in self.segments) if self.segments else 0
            tp = oh-ih
            self.box_list.set_boxes(self.boxes, groups)
            gray = self.profiles.get("gray", np.array([]))
            n = len(gray)
            info = f"Boxes: {len(self.boxes)} → {len(groups)} group(s) → {n} rows\n"
            info += f"Border trimmed: {tp} px total\n"
            info += "─"*35+"\n"
            if self.calibrator.is_calibrated:
                cal_info = []
                for gi in range(len(groups)):
                    d = self.calibrator.get_group(gi)
                    if d:
                        cal_info.append(f"  G{gi+1}: {d[0]:.3f} – {d[1]:.3f} m")
                if cal_info:
                    info += "\n".join(cal_info) + "\n"
            for gi, grp in enumerate(groups):
                if len(grp)==1:
                    x,y,w,h = self.boxes[grp[0]]
                    info += f"  G{gi+1}: ({x},{y}) {w}×{h}\n"
                else:
                    info += f"  G{gi+1}: {len(grp)} parallel → averaged\n"
                    for pos, bi in enumerate(grp):
                        x,y,w,h = self.boxes[bi]
                        info += f"    #{bi+1}: ({x},{y}) {w}×{h}\n"
            self.info_label.setText(info)
            self.sb.showMessage(f"{len(groups)} group(s) | {n} rows | 6 channels")

        # Show in zoomable view
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h_img, w_img, ch = display_rgb.shape
        qt_img = QtGui.QImage(display_rgb.data, w_img, h_img, ch*w_img, QtGui.QImage.Format.Format_RGB888)
        self.image_view.set_image(qt_img)
        self._refresh_plot()

    def _refresh_plot(self):
        profiles_to_plot = self.profiles.copy(); trend = {}
        if self.illum_corrected and self.profiles:
            for ch in MultiChannelExtractor.CHANNELS:
                if ch in self.profiles and len(self.profiles[ch])>0:
                    profiles_to_plot[ch], trend[ch] = IlluminationCorrector.correct(self.profiles[ch])
            self.trend_lines = trend
        # Build per-group depth array
        depth_arr = None
        if self.calibrator.is_calibrated and self.segments:
            n = len(self.profiles.get("gray", []))
            depth_arr = np.zeros(n, dtype=np.float32)
            groups = MultiChannelExtractor.group_boxes(self.boxes)
            # Map box→group
            box2group = {}
            for gi, grp in enumerate(groups):
                for bi in grp:
                    box2group[bi] = gi
            for ss, se, bi in self.segments:
                gi = box2group.get(bi, 0)
                h = se - ss
                local = np.arange(h, dtype=np.float64)
                depth_arr[ss:se] = self.calibrator.pixel_to_depth(local, h, gi)
        self.plot_widget.plot(profiles_to_plot, self.segments, depth_arr,
                              illum_corrected=self.illum_corrected, trend_lines=self.trend_lines)

    # ── Depth calibration ────────────────────

    def _update_depth_bar(self):
        """Populate group dropdown and show current depth for selected group."""
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

    def _apply_depth(self):
        if not self.boxes: return
        gi = self.combo_group.currentIndex()
        if gi < 0: return
        tm = self.spin_top_m.value(); bm = self.spin_bot_m.value()
        if tm >= bm:
            QtWidgets.QMessageBox.warning(self, "Depth", "Top must be < bottom depth")
            return
        self.calibrator.set_group(gi, tm, bm)

        groups = MultiChannelExtractor.group_boxes(self.boxes)
        all_cal = sorted([g for g in range(len(groups)) if self.calibrator.get_group(g)],
                         key=lambda g: self.calibrator.get_group(g)[0])

        # ── Check for overlaps between adjacent groups ──
        overlaps = []
        for a, b in zip(all_cal, all_cal[1:]):
            da = self.calibrator.get_group(a); db = self.calibrator.get_group(b)
            ov = min(da[1], db[1]) - max(da[0], db[0])
            if ov > 0:
                overlaps.append((a, b, ov))

        if overlaps and len(all_cal) >= 2:
            dlg = OverlapMergeDialog(overlaps, self)
            if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                self._merge_choice = dlg.choice()
                # Store raw profiles before merging
                self.raw_profiles = {ch: self.profiles[ch].copy()
                                     for ch in MultiChannelExtractor.CHANNELS if len(self.profiles.get(ch, [])) > 0}
                self.raw_segments = list(self.segments)
                merged_prof, merged_seg, merged_depth, ov_info = OverlapMerger.merge(
                    self.profiles, self.segments, self.calibrator, groups, self._merge_choice)
                self.profiles = merged_prof
                self.segments = merged_seg
                self._merged_depth = merged_depth
                self._has_merged = True
        else:
            self._has_merged = False

        self._refresh_plot()
        info = f"Group {gi+1}: {tm:.3f} – {bm:.3f} m"
        if overlaps:
            for a, b, ov in overlaps:
                info += f"\n  Overlap G{a+1} & G{b+1}: {ov:.3f}m"
            if hasattr(self, '_merge_choice'):
                info += f"\n  Merged: {self._merge_choice}"
        self.sb.showMessage(info)

    # ── Illumination toggle ──────────────────

    def _toggle_illum(self, checked):
        self.illum_corrected = checked; self._refresh_plot()
        self.sb.showMessage("Illumination correction " + ("ON" if checked else "OFF"))

    # ── Box reorder ───────────────────────────

    def _on_order_changed(self):
        self.boxes = self.box_list.get_boxes()
        if self.image_bgr is not None and self.boxes:
            self.profiles, self.segments = MultiChannelExtractor.extract_all(
                self.image_bgr, self.boxes, getattr(self, 'red_mask', None))
            self._refresh_plot()
            groups = MultiChannelExtractor.group_boxes(self.boxes)
            display = self.image_bgr.copy()
            for gi, grp in enumerate(groups):
                cb = self._h2b(BOX_COLORS[gi%len(BOX_COLORS)])
                for pos, bi in enumerate(grp):
                    x,y,w,h = self.boxes[bi]
                    cv2.rectangle(display, (x,y), (x+w,y+h), cb, 3)
                    lbl = f"G{gi+1}#{pos+1}" if len(grp)>1 else f"#{bi+1}"
                    (lw,lh),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)
                    cv2.rectangle(display, (x,y-lh-10), (x+lw+6,y), cb, -1)
                    cv2.putText(display, lbl, (x+3,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            for gi in range(len(groups)-1):
                x1,y1,w1,h1 = self.boxes[groups[gi][-1]]
                x2,y2,w2,h2 = self.boxes[groups[gi+1][0]]
                cv2.arrowedLine(display, (x1+w1//2,y1+h1), (x2+w2//2,y2), (80,80,80), 2, tipLength=0.04)
            drgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            hh,ww,chh = drgb.shape
            qi = QtGui.QImage(drgb.data, ww, hh, chh*ww, QtGui.QImage.Format.Format_RGB888)
            self.image_view.set_image(qi)
            self.sb.showMessage(f"Order updated — {len(self.boxes)} box(es)")

    # ── Export CSV ────────────────────────────

    def _export_csv(self):
        src = self.profiles
        if self.illum_corrected:
            src = {}
            for ch in MultiChannelExtractor.CHANNELS:
                if ch in self.profiles and len(self.profiles[ch])>0:
                    src[ch], _ = IlluminationCorrector.correct(self.profiles[ch])
        gd = src.get("gray", np.array([]))
        if len(gd)==0: return
        dn = "gray_profile.csv"
        if self.current_path:
            dn = f"{os.path.splitext(os.path.basename(self.current_path))[0]}_profile.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export CSV", dn, "CSV (*.csv);;All (*)")
        if not path: return
        bid = np.full(len(gd), -1, dtype=int)
        for ss, se, bi in self.segments: bid[ss:se] = bi+1
        # Per-group depth
        dm = None
        if self.calibrator.is_calibrated:
            dm = np.zeros(len(gd), dtype=np.float32)
            groups = MultiChannelExtractor.group_boxes(self.boxes)
            b2g = {}
            for gi, grp in enumerate(groups):
                for bi in grp: b2g[bi] = gi
            for ss, se, bi in self.segments:
                gi = b2g.get(bi, 0); h = se-ss
                local = np.arange(h, dtype=np.float64)
                dm[ss:se] = self.calibrator.pixel_to_depth(local, h, gi)
        hdrs = ["row_px"]; hdrs += (["depth_m"] if dm is not None else [])
        hdrs += ["box_id"]
        if self._has_merged:
            for ch in MultiChannelExtractor.CHANNELS:
                hdrs.append(f"raw_{ch}")
        for ch in MultiChannelExtractor.CHANNELS:
            hdrs.append(ch if not self._has_merged else f"merged_{ch}")
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f); w.writerow(hdrs)
            for i in range(len(gd)):
                row = [i]
                if dm is not None: row.append(round(float(dm[i]),4))
                row.append(int(bid[i]))
                if self._has_merged and self.raw_profiles:
                    for ch in MultiChannelExtractor.CHANNELS:
                        raw_data = self.raw_profiles.get(ch, np.array([]))
                        row.append(round(float(raw_data[i]), 4) if i < len(raw_data) else "")
                for ch in MultiChannelExtractor.CHANNELS:
                    row.append(round(float(src[ch][i]),4))
                w.writerow(row)
        extra = " (raw + merged)" if self._has_merged else ""
        self.sb.showMessage(f"CSV exported{extra}: {os.path.basename(path)} ({len(gd)} rows)")

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

    @staticmethod
    def _h2b(h): return (int(h[5:7],16), int(h[3:5],16), int(h[1:3],16))


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    w = MainWindow(); w.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
