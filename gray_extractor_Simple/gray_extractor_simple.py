"""
Gray Extractor Simple v2 — Preprocessing + Standalone Plot
============================================================
Load image -> detect red boxes -> extract gray -> preprocess -> export
Preprocessing: illumination correction, resample, smoothing
"""
import sys, os, csv
import numpy as np, cv2
from PyQt6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# ── Optional scipy import (fallback to numpy) ──
try:
    from scipy.interpolate import interp1d
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    def interp1d(x, y, kind='linear', bounds_error=False, fill_value=0):
        """Pure numpy linear interpolation fallback."""
        x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
        idx = np.argsort(x); x, y = x[idx], y[idx]
        def _interp(x_new):
            x_new = np.asarray(x_new, dtype=np.float64)
            result = np.full_like(x_new, fill_value if np.isscalar(fill_value) else fill_value[0], dtype=np.float64)
            lo = np.searchsorted(x, x_new, side='right') - 1
            lo = np.clip(lo, 0, len(x)-2)
            hi = lo + 1
            dx = x[hi] - x[lo]
            valid = dx > 0
            t = np.zeros_like(x_new); t[valid] = (x_new[valid]-x[lo][valid])/dx[valid]
            result = y[lo] + t*(y[hi]-y[lo])
            return result
        return _interp

# ═══════════════════════ Data Classes ═══════════════════════

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
            if h>=150 and cv2.contourArea(cnt)>=5000 and h/max(w,1)>=2.0:
                boxes.append((x,y,w,h))
        if not boxes: return [], raw
        boxes.sort(key=lambda b: b[1]+b[3], reverse=True)
        return boxes, raw


class GrayExtractor:
    @staticmethod
    def _inner_roi(red_mask, outer_rect):
        x,y,w,h = outer_rect
        roi = red_mask[y:y+h, x:x+w]
        t,b,l,r = 0,0,0,0
        for i in range(50):
            if np.count_nonzero(roi[i,:])/max(w,1)<0.05: t=i; break
        else: t = max(4, h//40)
        for i in range(50):
            if np.count_nonzero(roi[h-1-i,:])/max(w,1)<0.05: b=i; break
        else: b = max(4, h//40)
        for i in range(50):
            if np.count_nonzero(roi[:,i])/max(h,1)<0.05: l=i; break
        else: l = max(4, w//30)
        for i in range(50):
            if np.count_nonzero(roi[:,w-1-i])/max(h,1)<0.05: r=i; break
        else: r = max(4, w//30)
        ix,iy = x+l, y+t
        iw,ih = (w-r)-l, (h-b)-t
        return (ix,iy,iw,ih) if iw>=10 and ih>=20 else outer_rect

    @staticmethod
    def extract(image_bgr, boxes, red_mask):
        if not boxes: return np.array([]), []
        all_gray, segs = [], []
        off = 0
        for bi, (x,y,w,h) in enumerate(boxes):
            ix,iy,iw,ih = GrayExtractor._inner_roi(red_mask, (x,y,w,h))
            roi = image_bgr[iy:iy+ih, ix:ix+iw].astype(np.float32)
            gray = np.mean(0.299*roi[:,:,2]+0.587*roi[:,:,1]+0.114*roi[:,:,0], axis=1)
            all_gray.append(gray)
            n = len(gray)
            segs.append((off, off+n, bi))
            off += n
        return np.concatenate(all_gray) if all_gray else np.array([]), segs


class DepthCalibrator:
    def __init__(self): self._g = {}
    def set(self, gi, top, bottom): self._g[gi] = (top, bottom)
    def get(self, gi): return self._g.get(gi)
    def has_any(self): return bool(self._g)
    def clear(self): self._g.clear()
    def build_depth_array(self, n_total, boxes, segments):
        if not self.has_any(): return None
        depth = np.zeros(n_total, dtype=np.float32)
        for ss, se, bi in segments:
            h = se-ss; local = np.arange(h, dtype=np.float64)
            t,b = self._g.get(bi,(0.0,1.0))
            frac = local/max(h,1)
            depth[ss:se] = (t+frac*(b-t)).astype(np.float32)
        return depth


# ═══════════════════════ Zoomable Image View ═══════════════════════

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
        self.setStyleSheet("border:1px solid #ccc; background:#e8e8e8;")
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
        factor = 1.15 if event.angleDelta().y()>0 else 1/1.15
        new_zoom = self._zoom*factor
        if 0.05 < new_zoom < 20:
            self.scale(factor, factor)
            self._zoom = new_zoom
        event.accept()

    def fit_to_window(self):
        if self._image_rect:
            self.fitInView(self._scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = self.transform().m11()


# ═══════════════════════ Standalone Gray Plot Dialog ═══════════════════════

class GrayPlotDialog(QtWidgets.QDialog):
    """Independent, resizable gray profile window with zoom/pan/save toolbar."""
    def __init__(self, parent, gray, depth_arr, title="Gray Profile"):
        super().__init__(parent)
        self.gray = gray; self.depth_arr = depth_arr
        self.setWindowTitle(title)
        self.resize(900, 700); self.setMinimumSize(500, 400)

        layout = QtWidgets.QVBoxLayout(self); layout.setContentsMargins(0,0,0,0)
        self.figure = Figure(figsize=(8, 9), dpi=100); self.figure.set_facecolor('#fafafa')
        self.canvas = FigureCanvasQTAgg(self.figure)
        toolbar = NavigationToolbar(self.canvas, self)
        toolbar.setStyleSheet("QToolBar{border-bottom:1px solid #ddd;}")
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas, 1)
        self._draw()

    def _draw(self):
        self.figure.clear(); ax = self.figure.add_subplot(111)
        n = len(self.gray)
        if n==0:
            ax.text(0.5,0.5,'No data',transform=ax.transAxes,ha='center',va='center')
            self.canvas.draw(); return
        if self.depth_arr is not None and len(self.depth_arr)==n:
            ax.plot(self.gray, self.depth_arr, color='#2D7DD2', linewidth=0.5)
            ax.set_ylabel('Depth (m)', fontsize=8, color='#999')
            ax.invert_yaxis()
        else:
            ax.plot(self.gray, np.arange(n), color='#2D7DD2', linewidth=0.5)
            ax.invert_yaxis()
            ax.set_ylabel('Pixel row', fontsize=8, color='#999')
        ax.set_xlabel('Gray value', fontsize=8, color='#666')
        ax.set_title(f'Gray Profile — {n} points', fontsize=9, loc='left', fontweight='bold')
        ax.tick_params(labelsize=6); ax.grid(True, alpha=0.12, linestyle='--')
        self.canvas.draw()

    def update_data(self, gray, depth_arr):
        self.gray = gray; self.depth_arr = depth_arr; self._draw()


# ═══════════════════════ Main Window ═══════════════════════

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gray Extractor Simple v2")
        self.setMinimumSize(900,600); self.resize(1300,800)

        self.image_bgr = None; self.boxes = []
        self.gray_raw = np.array([]); self.segments = []
        self.red_mask = None; self.calibrator = DepthCalibrator()
        self.current_path = ""; self._depth_arr = None
        self._plot_dialog = None

        self._setup_ui(); self._setup_toolbar(); self.setAcceptDrops(True)
        self._apply_style()
        self.statusBar().showMessage("Ready — Open image (Ctrl+O) or drag & drop")

    # ── UI ────────────────────────────────────

    def _setup_ui(self):
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central); root.setContentsMargins(4,4,4,4)

        # Left: image
        left = QtWidgets.QVBoxLayout()
        self.image_view = ZoomableImageView()
        self.image_view.setMinimumWidth(380)
        left.addWidget(self.image_view, 1)

        # Depth bar
        db = QtWidgets.QHBoxLayout()
        db.addWidget(QtWidgets.QLabel("Box:"))
        self.combo_group = QtWidgets.QComboBox(); self.combo_group.setMinimumWidth(80)
        self.combo_group.currentIndexChanged.connect(self._on_group_changed)
        db.addWidget(self.combo_group)
        db.addWidget(QtWidgets.QLabel("Top:"))
        self.spin_top = QtWidgets.QDoubleSpinBox()
        self.spin_top.setDecimals(3); self.spin_top.setRange(-10,1000)
        self.spin_top.setSuffix(" m"); self.spin_top.setMinimumWidth(80)
        db.addWidget(self.spin_top)
        db.addWidget(QtWidgets.QLabel("Bot:"))
        self.spin_bot = QtWidgets.QDoubleSpinBox()
        self.spin_bot.setDecimals(3); self.spin_bot.setRange(-10,1000)
        self.spin_bot.setSuffix(" m"); self.spin_bot.setMinimumWidth(80)
        db.addWidget(self.spin_bot)
        self.btn_set = QtWidgets.QPushButton("Set")
        self.btn_set.clicked.connect(self._apply_depth); self.btn_set.setEnabled(False)
        self.btn_set.setStyleSheet("background:#1A9C6E;color:white;padding:2px 10px;border-radius:3px;")
        db.addWidget(self.btn_set); db.addStretch()
        left.addLayout(db)

        self.lbl_info = QtWidgets.QLabel("")
        self.lbl_info.setStyleSheet("color:#555;font-size:11px;padding:2px;")
        left.addWidget(self.lbl_info)

        # Right: preview plot + preprocessing controls
        right = QtWidgets.QVBoxLayout()

        # Preprocessing bar
        pp = QtWidgets.QHBoxLayout()
        pp.addWidget(QtWidgets.QLabel("Preprocess:"))
        self.chk_illum = QtWidgets.QCheckBox("Illum Corr")
        self.chk_illum.toggled.connect(self._reprocess)
        self.chk_illum.setToolTip("2nd-order polynomial detrending — removes lighting gradient")
        pp.addWidget(self.chk_illum)

        pp.addWidget(QtWidgets.QLabel("Resample:"))
        self.combo_resample = QtWidgets.QComboBox()
        self.combo_resample.addItems(["none", "0.5 cm", "1 cm", "2 cm", "5 cm"])
        self.combo_resample.setCurrentIndex(2)  # default 1cm
        self.combo_resample.currentIndexChanged.connect(self._reprocess)
        self.combo_resample.setToolTip("Resample to uniform depth grid (requires depth calibration)")
        pp.addWidget(self.combo_resample)

        self.chk_smooth = QtWidgets.QCheckBox("Smooth")
        self.chk_smooth.toggled.connect(self._reprocess)
        self.chk_smooth.setToolTip("Gaussian smoothing — reduces pixel-level noise")
        pp.addWidget(self.chk_smooth)
        pp.addWidget(QtWidgets.QLabel("σ:"))
        self.spin_sigma = QtWidgets.QDoubleSpinBox()
        self.spin_sigma.setDecimals(1); self.spin_sigma.setRange(0.5, 20); self.spin_sigma.setValue(2.0)
        self.spin_sigma.setSuffix(" px"); self.spin_sigma.setMinimumWidth(60)
        self.spin_sigma.valueChanged.connect(self._reprocess)
        pp.addWidget(self.spin_sigma)
        pp.addStretch()

        self.btn_popout = QtWidgets.QPushButton("Pop-out Plot")
        self.btn_popout.clicked.connect(self._open_plot_window)
        self.btn_popout.setEnabled(False)
        self.btn_popout.setStyleSheet("padding:2px 8px; background:#2D7DD2; color:white; border-radius:3px;")
        pp.addWidget(self.btn_popout)
        right.addLayout(pp)

        # Preview plot
        self.figure = Figure(figsize=(7, 6), dpi=100); self.figure.set_facecolor('#fafafa')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumWidth(350)
        right.addWidget(self.canvas, 1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        lc = QtWidgets.QWidget(); lc.setLayout(left)
        rc = QtWidgets.QWidget(); rc.setLayout(right)
        splitter.addWidget(lc); splitter.addWidget(rc)
        splitter.setStretchFactor(0,1); splitter.setStretchFactor(1,2)
        root.addWidget(splitter)

    def _setup_toolbar(self):
        tb = self.addToolBar("Main"); tb.setMovable(False)
        a = QtGui.QAction("Open", self); a.setShortcut("Ctrl+O")
        a.triggered.connect(self._open_image); tb.addAction(a); tb.addSeparator()

        self.act_csv = QtGui.QAction("Export CSV", self); self.act_csv.setShortcut("Ctrl+S")
        self.act_csv.triggered.connect(self._export_csv); self.act_csv.setEnabled(False)
        tb.addAction(self.act_csv)

        self.act_svg = QtGui.QAction("Export SVG", self); self.act_svg.setShortcut("Ctrl+P")
        self.act_svg.triggered.connect(self._export_svg); self.act_svg.setEnabled(False)
        tb.addAction(self.act_svg)

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow{background:#fafafa}
            QToolBar{background:#fff;border-bottom:1px solid #ddd;padding:3px;spacing:5px}
            QStatusBar{background:#fff;border-top:1px solid #ddd;color:#444}
            QCheckBox{padding:2px 4px;}
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
            QtWidgets.QMessageBox.critical(self,"Error",f"Cannot open:\n{path}"); return
        self.current_path = path; self._run_detection()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.accept()
        else: e.ignore()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if p.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff','.tif')):
                self._load_image(p); break

    # ── Detection ─────────────────────────────

    def _run_detection(self):
        if self.image_bgr is None: return
        self.boxes, self.red_mask = RedBoxDetector.detect_all(self.image_bgr)
        has = bool(self.boxes)
        self.act_csv.setEnabled(has); self.act_svg.setEnabled(has)
        self.btn_set.setEnabled(has); self.btn_popout.setEnabled(has)
        self.calibrator.clear()
        if has:
            self._update_combo()
            self.gray_raw, self.segments = GrayExtractor.extract(self.image_bgr, self.boxes, self.red_mask)
        else:
            self.gray_raw = np.array([]); self.segments = []
        self._redraw_image(); self._reprocess()

    def _update_combo(self):
        self.combo_group.blockSignals(True); self.combo_group.clear()
        for bi in range(len(self.boxes)): self.combo_group.addItem(f"Box {bi+1}")
        self.combo_group.blockSignals(False); self._on_group_changed(0)

    def _redraw_image(self):
        if self.image_bgr is None: return
        display = self.image_bgr.copy()
        if self.boxes:
            for bi, (x,y,w,h) in enumerate(self.boxes):
                cv2.rectangle(display,(x,y),(x+w,y+h),(45,125,210),3)
                lbl=f"#{bi+1}"; (lw,lh),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.8,3)
                cv2.rectangle(display,(x,y-lh-8),(x+lw+4,y),(45,125,210),-1)
                cv2.putText(display,lbl,(x+2,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                if self.calibrator.get(bi):
                    d=self.calibrator.get(bi)
                    cv2.putText(display,f"{d[0]:.2f}-{d[1]:.2f}m",(x+5,y+h//2),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,140,0),2)
        rgb=cv2.cvtColor(display,cv2.COLOR_BGR2RGB)
        h,w,ch=rgb.shape
        qi=QtGui.QImage(rgb.data,w,h,ch*w,QtGui.QImage.Format.Format_RGB888)
        self.image_view.set_image(qi)

    # ── Preprocessing pipeline ────────────────

    def _reprocess(self):
        """Apply selected preprocessing steps and refresh both plots."""
        if len(self.gray_raw)==0: return
        gray = self.gray_raw.astype(np.float64).copy()
        n = len(gray)
        depth_arr = self.calibrator.build_depth_array(n, self.boxes, self.segments)
        self._depth_arr = depth_arr

        # Step 1: Illumination correction (polynomial detrending)
        if self.chk_illum.isChecked() and depth_arr is not None:
            coeffs = np.polyfit(depth_arr, gray, 2)
            trend = np.polyval(coeffs, depth_arr)
            gray = gray - trend + np.mean(gray)

        # Step 2: Resample to uniform depth grid
        if self.combo_resample.currentIndex()>0 and depth_arr is not None:
            cm_per_step = [0.5, 1.0, 2.0, 5.0][self.combo_resample.currentIndex()-1]
            step_m = cm_per_step/100.0
            d_min, d_max = depth_arr[0], depth_arr[-1]
            if d_min > d_max: d_min, d_max = d_max, d_min
            grid = np.arange(d_min, d_max+step_m/2, step_m)
            f_interp = interp1d(depth_arr, gray, kind='linear', bounds_error=False,
                                fill_value=(gray[0], gray[-1]))
            gray = f_interp(grid)
            depth_arr = grid

        # Step 3: Gaussian smoothing
        if self.chk_smooth.isChecked() and len(gray)>5:
            sigma = self.spin_sigma.value()
            radius = int(np.ceil(3*sigma))
            x = np.arange(-radius, radius+1, dtype=np.float64)
            kernel = np.exp(-0.5*(x/sigma)**2); kernel /= kernel.sum()
            gray = np.convolve(gray, kernel, mode='same')

        self._gray_processed = gray
        self._depth_processed = depth_arr
        self._redraw_preview(gray, depth_arr)

        # Update pop-out window if open
        if self._plot_dialog is not None and self._plot_dialog.isVisible():
            self._plot_dialog.update_data(gray, depth_arr)

        n_info = len(gray)
        info = f"Raw: {len(self.gray_raw)} rows → Processed: {n_info} pts"
        if depth_arr is not None and n_info>1:
            info += f" | span: {abs(depth_arr[-1]-depth_arr[0]):.3f}m"
        self.lbl_info.setText(info)

    def _redraw_preview(self, gray, depth_arr):
        self.figure.clear(); ax = self.figure.add_subplot(111)
        n=len(gray)
        if n==0:
            ax.text(0.5,0.5,'No data',transform=ax.transAxes,ha='center',va='center')
            self.canvas.draw(); return
        if depth_arr is not None and len(depth_arr)==n:
            ax.plot(gray, depth_arr, color='#2D7DD2', linewidth=0.5)
            ax.set_ylabel('Depth (m)', fontsize=8, color='#999')
            ax.invert_yaxis()
        else:
            ax.plot(gray, np.arange(n), color='#2D7DD2', linewidth=0.5)
            ax.invert_yaxis(); ax.set_ylabel('Pixel row', fontsize=8, color='#999')
        ax.set_xlabel('Gray', fontsize=8, color='#666')
        active = []
        if self.chk_illum.isChecked(): active.append("IllumCorr")
        if self.combo_resample.currentIndex()>0: active.append(f"Resample@{self.combo_resample.currentText()}")
        if self.chk_smooth.isChecked(): active.append(f"Smooth(σ={self.spin_sigma.value():.1f})")
        tag = " | ".join(active) if active else "Raw"
        ax.set_title(f'Gray Profile — {n} pts — [{tag}]', fontsize=9, loc='left', fontweight='bold')
        ax.tick_params(labelsize=6); ax.grid(True, alpha=0.12, linestyle='--')
        self.canvas.draw()

    # ── Depth calibration ────────────────────

    def _on_group_changed(self, idx):
        if idx<0: return
        d=self.calibrator.get(idx)
        if d: self.spin_top.setValue(d[0]); self.spin_bot.setValue(d[1])
        else: self.spin_top.setValue(0); self.spin_bot.setValue(1)

    def _apply_depth(self):
        if not self.boxes: return
        gi=self.combo_group.currentIndex()
        if gi<0: return
        t=self.spin_top.value(); b=self.spin_bot.value()
        if t>=b: QtWidgets.QMessageBox.warning(self,"Depth","Top must be < bottom"); return
        self.calibrator.set(gi, t, b)
        self._redraw_image(); self._reprocess()
        nc=sum(1 for gi2 in range(len(self.boxes)) if self.calibrator.get(gi2))
        self.statusBar().showMessage(f"Box {gi+1}: {t:.3f}–{b:.3f}m ({nc}/{len(self.boxes)} calibrated)")

    # ── Pop-out plot window ──────────────────

    def _open_plot_window(self):
        g = getattr(self, '_gray_processed', self.gray_raw)
        d = getattr(self, '_depth_processed', self._depth_arr)
        if self._plot_dialog is not None and self._plot_dialog.isVisible():
            self._plot_dialog.update_data(g, d)
            self._plot_dialog.raise_()
        else:
            self._plot_dialog = GrayPlotDialog(self, g, d)
            self._plot_dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
            self._plot_dialog.destroyed.connect(lambda: setattr(self, '_plot_dialog', None))
            self._plot_dialog.show()

    # ── Export ────────────────────────────────

    def _export_csv(self):
        g = getattr(self, '_gray_processed', self.gray_raw)
        d = getattr(self, '_depth_processed', self._depth_arr)
        if len(g)==0: return
        dn = "gray_profile.csv"
        if self.current_path:
            dn = f"{os.path.splitext(os.path.basename(self.current_path))[0]}_gray.csv"
        path,_ = QtWidgets.QFileDialog.getSaveFileName(self,"Export CSV",dn,"CSV (*.csv);;All (*)")
        if not path: return

        with open(path,'w',newline='',encoding='utf-8') as f:
            w = csv.writer(f)
            hdrs = ["index"]
            if d is not None and len(d)==len(g): hdrs.append("depth_m")
            hdrs.append("gray")
            w.writerow(hdrs)
            for i in range(len(g)):
                row = [i+1]
                if d is not None and len(d)==len(g): row.append(round(float(d[i]),4))
                row.append(round(float(g[i]),4))
                w.writerow(row)
        self.statusBar().showMessage(f"CSV: {os.path.basename(path)} ({len(g)} rows)")

    def _export_svg(self):
        g = getattr(self, '_gray_processed', self.gray_raw)
        d = getattr(self, '_depth_processed', self._depth_arr)
        if len(g)==0: return
        dn = "gray_profile.svg"
        if self.current_path:
            dn = f"{os.path.splitext(os.path.basename(self.current_path))[0]}_gray.svg"
        path,_ = QtWidgets.QFileDialog.getSaveFileName(self,"Export SVG",dn,"SVG (*.svg);;All (*)")
        if not path: return
        self._redraw_preview(g, d)
        self.figure.savefig(path, dpi=200, bbox_inches='tight', facecolor='#fafafa')
        self.statusBar().showMessage(f"SVG: {os.path.basename(path)}")


# ═══════════════════════ Entry ═══════════════════════

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    w = MainWindow(); w.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
