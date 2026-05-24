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
#  Multi-channel plot widget (v5: depth ruler)
# ═══════════════════════════════════════════════

class MultiChannelPlotWidget(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(8, 10), dpi=100)
        self.figure.set_facecolor('#fafafa')
        super().__init__(self.figure)
        self.setParent(parent)
        self.axes = {}
        self._create_subplots()
        self._depth_data = None  # stored for export

    def _create_subplots(self):
        for i, ch in enumerate(MultiChannelExtractor.CHANNELS):
            self.axes[ch] = self.figure.add_subplot(2, 3, i+1)

    def plot(self, profiles, segments, depth_array=None, illum_corrected=False, trend_lines=None):
        self._depth_data = (profiles, segments, depth_array, illum_corrected, trend_lines)
        for ch in MultiChannelExtractor.CHANNELS:
            ax = self.axes[ch]; ax.clear()
            data = profiles.get(ch)
            if data is None or len(data)==0:
                ax.text(0.5,0.5,'No data',transform=ax.transAxes,ha='center',va='center',fontsize=10,color='gray')
                ax.set_title(ch, fontsize=9, fontweight='bold'); continue
            n = len(data); color = CHANNEL_COLORS[ch]
            y_vals = np.arange(n)
            for ss, se, bi in segments:
                ax.axhspan(ss, se, alpha=0.06, color=BOX_COLORS[bi%len(BOX_COLORS)])
            ax.fill_betweenx(y_vals, data, data.min(), alpha=0.15, color=color)
            ax.plot(data, y_vals, color=color, linewidth=0.5)
            if illum_corrected and trend_lines and ch in trend_lines:
                ax.plot(trend_lines[ch], y_vals, '--', color='#999', linewidth=0.8, alpha=0.7)
            ax.axvline(float(np.mean(data)), color=color, linestyle=':', linewidth=0.8, alpha=0.6)
            ax.invert_yaxis()
            ax.set_xlabel(CHANNEL_LABELS.get(ch, ch), fontsize=7)
            ax.set_title(ch, fontsize=9, fontweight='bold', color=color)
            ax.grid(True, alpha=0.2, linestyle='--'); ax.tick_params(labelsize=7)

            # ── v5.1: Per-group depth ruler ──
            if depth_array is not None and len(depth_array) == n:
                n_ticks = 8
                step = max(1, n // n_ticks)
                tick_rows = list(range(0, n, step))
                tick_labels = [f"{depth_array[i]:.2f}" for i in tick_rows]
                ax.set_yticks(tick_rows)
                ax.set_yticklabels(tick_labels, fontsize=6)
                ax.set_ylabel("Depth (m)", fontsize=7)
                # Segment boundary markers
                for ss, se, bi in segments:
                    if ss > 0:
                        ax.axhline(ss, color='#F0B429', linestyle='-', linewidth=1.0, alpha=0.5)
                        ax.text(0.02, ss/n, f'{depth_array[min(ss,n-1)]:.2f}m',
                                transform=ax.transAxes, fontsize=6, color='#F0B429',
                                va='center', clip_on=False)

        self.figure.tight_layout(pad=2.0)
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
#  Main window
# ═══════════════════════════════════════════════

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_bgr = None; self.boxes = []; self.profiles = {}; self.segments = []
        self.current_path = ""; self.calibrator = DepthCalibrator()
        self.illum_corrected = False; self.trend_lines = {}; self.red_mask = None

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
        self._refresh_plot()
        groups = MultiChannelExtractor.group_boxes(self.boxes)
        info = f"Group {gi+1} depth: {tm:.3f} – {bm:.3f} m"
        all_cal = [g for g in range(len(groups)) if self.calibrator.get_group(g)]
        if len(all_cal) > 1:
            # Cross-group alignment: merge overlapping depth ranges
            merged = []
            for g in sorted(all_cal):
                d = self.calibrator.get_group(g)
                merged.append((g, d[0], d[1]))
            merged.sort(key=lambda x: x[1])
            for a, b in zip(merged, merged[1:]):
                if a[2] > b[1]:
                    overlap = a[2] - b[1]
                    info += f"\n⚠ Group {a[0]+1} & {b[0]+1} overlap {overlap:.3f}m"
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
        hdrs += ["box_id"] + MultiChannelExtractor.CHANNELS
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f); w.writerow(hdrs)
            for i in range(len(gd)):
                row = [i]
                if dm is not None: row.append(round(float(dm[i]),4))
                row.append(int(bid[i]))
                for ch in MultiChannelExtractor.CHANNELS:
                    row.append(round(float(src[ch][i]),4))
                w.writerow(row)
        self.sb.showMessage(f"CSV exported: {os.path.basename(path)} ({len(gd)} rows)")

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
