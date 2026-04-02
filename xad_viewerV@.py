import sys, os, re
import numpy as np
from itertools import cycle
from scipy.ndimage import gaussian_filter
import pydicom as dcm
import xad_models as XM

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QTabWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QSpinBox,
    QDoubleSpinBox, QFileDialog, QMessageBox, QSizePolicy, QAction,
    QFrame, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QComboBox, QColorDialog, QButtonGroup, QActionGroup,
    QCheckBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsItem, QGraphicsRectItem, QGraphicsEllipseItem,
    QGraphicsPolygonItem, QGraphicsPathItem, QRadioButton, QGroupBox,
    QScrollArea, QListWidget, QListWidgetItem,
)
from PyQt5.QtCore import (Qt, QThread, pyqtSignal, QTimer, QRectF,
                          QPointF, QSizeF, QLineF)
from PyQt5.QtGui import (QColor, QPixmap, QImage, QPainter, QPen, QBrush,
                         QPainterPath, QPolygonF, QCursor, QFont,
                         QTransform, QKeySequence, QFontMetrics, QPalette,
                         QLinearGradient)


def get_HUpecs(keVHUhi, keVHUlo, keVref, scaling="Water"):
    khi, hhi = keVHUhi;  klo, hlo = keVHUlo
    muhi = XM.MU("Water", khi) * (hhi / 1000.0 + 1.0)
    mulo = XM.MU("Water", klo) * (hlo / 1000.0 + 1.0)
    if isinstance(scaling, (float, int)):
        g_pe = [np.power(float(k) / keVref, scaling) for k in [khi, klo]]
        g_cs = [XM.kleinNishina(k) / XM.kleinNishina(keVref) for k in [khi, klo]]
    else:
        g_pe = [(XM.dPE(scaling, k) + XM.dRA(scaling, k)) /
                (XM.dPE(scaling, keVref) + XM.dRA(scaling, keVref)) for k in [khi, klo]]
        g_cs = [XM.dCS(scaling, k) / XM.dCS(scaling, keVref) for k in [khi, klo]]
    cs = (muhi - mulo * g_pe[0] / g_pe[1]) / g_cs[0] / \
         (1.0 - g_cs[1] / g_cs[0] * g_pe[0] / g_pe[1])
    pe = (mulo - cs * g_cs[1]) / g_pe[1]
    return (1000 * (pe - (XM.dPE("Water", keVref) + XM.dRA("Water", keVref))) /
            XM.MU("Water", keVref),
            1000 * (cs - XM.dCS("Water", keVref)) / XM.MU("Water", keVref))


def _read_pixel_array(ds):
    try:
        return ds.pixel_array.astype(np.float32)
    except Exception as e0:
        for pkg in ("pylibjpeg", "gdcm"):
            try:
                __import__(pkg)
                return ds.pixel_array.astype(np.float32)
            except ImportError:
                pass
        raise RuntimeError(
            f"Cannot decode DICOM pixels.\n"
            f"Run: pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg\n{e0}"
        ) from e0


class DicomVolume:
    def __init__(self, vol, z_locs, ps, ss):
        self.vol    = vol
        self.z_locs = z_locs
        self.ps     = ps
        self.ss     = ss
        self.ax_aspect  = float(ps[0]) / float(ps[1])
        self.cor_aspect = float(ss)    / float(ps[1])
        self.sag_aspect = float(ss)    / float(ps[0])

    @property
    def shape_hwn(self): return self.vol.shape


def read_dicom_path(path):
    if os.path.isfile(path):
        if not path.lower().endswith(".dcm"):
            raise ValueError(f"Not a .dcm file: {path}")
        files = [path]
    else:
        files = [os.path.join(path, f)
                 for f in os.listdir(path) if f.lower().endswith(".dcm")]
        if not files:
            raise FileNotFoundError(f"No DICOM files in {path}")
    return files


def read_dicom_folder(folder):
    files = read_dicom_path(folder)
    headers = [(dcm.dcmread(fp, stop_before_pixels=True), fp) for fp in files]

    def _z(ds_fp):
        ds, fp = ds_fp
        try:    return float(ds.SliceLocation)
        except: pass
        try:    return float(ds.ImagePositionPatient[2])
        except: pass
        return None

    headers = [(ds, fp) for ds, fp in headers if _z((ds, fp)) is not None]
    headers.sort(key=lambda p: _z(p))
    slices, z_locs = [], []
    for ds, fp in headers:
        full = dcm.dcmread(fp)
        arr  = _read_pixel_array(full)
        if arr.ndim == 3: arr = arr[..., 0]
        slices.append(full.RescaleIntercept + full.RescaleSlope * arr)
        z_locs.append(_z((ds, fp)))
    vol_hwn = np.moveaxis(np.stack(slices, axis=0), 0, -1)
    z_arr   = np.array(z_locs, dtype=np.float64)
    ds0 = headers[0][0]
    try:    ps = [float(x) for x in ds0.PixelSpacing]
    except: ps = [1.0, 1.0]
    try:    ss = float(ds0.SliceThickness)
    except: ss = abs(z_arr[1] - z_arr[0]) if len(z_arr) > 1 else 1.0
    return DicomVolume(vol_hwn, z_arr, ps, ss)


def z_match_volumes(dv_hi, dv_lo):
    z_hi, z_lo = dv_hi.z_locs, dv_lo.z_locs
    z_ref = z_lo
    tol   = np.mean(np.abs(np.diff(z_ref))) * 0.6

    def _resample(vol, z_src, z_dst):
        idxs  = np.argmin(np.abs(z_src[:, None] - z_dst[None, :]), axis=0)
        good  = np.abs(z_src[idxs] - z_dst) < tol
        return vol[:, :, idxs], good

    vol_hi_rs, good = _resample(dv_hi.vol, z_hi, z_ref)
    vol_hi_out = vol_hi_rs[:, :, good]
    vol_lo_out = dv_lo.vol[:, :, good]
    z_out      = z_ref[good]
    if vol_hi_out.shape[2] == 0:
        raise RuntimeError(
            "Z-matching: no overlapping slices.\n"
            f"Hi z: {z_hi[0]:.1f}--{z_hi[-1]:.1f} mm\n"
            f"Lo z: {z_lo[0]:.1f}--{z_lo[-1]:.1f} mm")
    return vol_hi_out, vol_lo_out, z_out


# ── Material loading (flat folder) ────────────────────────────────────────────
class Material:
    def __init__(self, path):
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.rho  = 1.0
        self._kev = np.array([]); self._ra = np.array([])
        self._cs  = np.array([]); self._pe = np.array([])
        self._load(path)

    def _load(self, path):
        with open(path, "r") as f:
            lines = f.read().splitlines()
        for line in lines:
            parts = [p.strip() for p in line.split("\t")]
            if parts[0].lower() == "material" and len(parts) >= 2:
                self.name = parts[1]
            elif parts[0].lower() == "rho" and len(parts) >= 2:
                try: self.rho = float(parts[1])
                except: pass
        kevs, ras, css, pes = [], [], [], []
        for line in lines:
            parts = [p.strip() for p in line.split("\t")]
            try:
                kevs.append(float(parts[0])); ras.append(float(parts[1]))
                css.append(float(parts[2]));  pes.append(float(parts[3]))
            except: pass
        self._kev = np.array(kevs); self._ra = np.array(ras)
        self._cs  = np.array(css);  self._pe = np.array(pes)

    def mu(self, keV):
        ra = np.interp(keV, self._kev, self._ra)
        cs = np.interp(keV, self._kev, self._cs)
        pe = np.interp(keV, self._kev, self._pe)
        return self.rho * (ra + cs + pe)

    def HU(self, keV):
        return 1000.0 * (self.mu(keV) / XM.MU("Water", keV) - 1.0)

    def xad_point(self, kh, kl, kr):
        result = get_HUpecs((kh, self.HU(kh)), (kl, self.HU(kl)), kr)
        return float(result[0]), float(result[1])


def load_materials_from_folder(folder):
    """Flat scan — all .tsv files directly in folder, no subfolders assumed."""
    mats = []
    if not os.path.isdir(folder): return mats
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith(".tsv"):
            try: mats.append(Material(os.path.join(folder, fn)))
            except Exception as e: print(f"Could not load {fn}: {e}")
    return mats


# ── constants ─────────────────────────────────────────────────────────────────
ROI_POLYGON   = "polygon"
ROI_ELLIPSE   = "ellipse"
ROI_RECTANGLE = "rectangle"
VIEW_AX = "AX";  VIEW_CO = "CO";  VIEW_SA = "SA"
ALL_VIEWS = (VIEW_AX, VIEW_CO, VIEW_SA)
PHILIPS_ROWS  = 50
SCATTER_R     = 1.5

DEFAULT_COLORS = [
    "#e6194b","#3cb44b","#4363d8","#f58231","#911eb4",
    "#42d4f4","#f032e6","#bfef45","#fabed4","#469990",
]
WW_PRESETS = {
    "Soft tissue": (40, 400),
    "Lung":        (-600, 1500),
    "Bone":        (400, 1800),
    "Brain":       (40, 80),
    "Liver":       (60, 160),
}
XAD_CS_LIM = (-1000, 500)
XAD_PE_LIM = (-200,  500)

TICK_COLOR  = QColor(180, 180, 180)
TICK_FONT   = QFont("Monospace", 8)
TICK_LEN    = 5


def _vline():
    f = QFrame(); f.setFrameShape(QFrame.VLine); f.setFrameShadow(QFrame.Sunken)
    return f


def _nice_ticks(lo, hi, n=6):
    span = hi - lo
    if span == 0: return [lo]
    raw  = span / n
    mag  = 10 ** np.floor(np.log10(raw))
    step = mag * min([1,2,5,10], key=lambda s: abs(s*mag - raw))
    first = np.ceil(lo / step) * step
    ticks = []; v = first
    while v <= hi + 1e-9:
        ticks.append(round(v, 10)); v += step
    return ticks


def hu_to_uint8(arr2d, wc, ww):
    lo, hi = wc - ww / 2, wc + ww / 2
    return np.clip((arr2d - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def gray_u8_to_qimage(arr):
    h, w = arr.shape; arr_c = np.ascontiguousarray(arr)
    return QImage(arr_c.data, w, h, w, QImage.Format_Grayscale8).copy()


def rgba_to_qimage(arr):
    h, w = arr.shape[:2]; arr_c = np.ascontiguousarray(arr)
    return QImage(arr_c.data, w, h, w * 4, QImage.Format_RGBA8888).copy()


def colormap_magma(t):
    t = np.clip(t, 0, 1)
    r2 = np.where(t < 0.5, t * 2 * 0.9, 0.9 + (t - 0.5) * 2 * 0.1)
    g2 = np.where(t < 0.33, t * 3 * 0.05,
         np.where(t < 0.67, 0.05 + (t - 0.33) * 3 * 0.4,
                  0.45 + (t - 0.67) * 3 * 0.55))
    b2 = np.where(t < 0.25, t * 4 * 0.5,
         np.where(t < 0.5,  0.5  - (t - 0.25) * 4 * 0.35,
         np.where(t < 0.75, 0.15 - (t - 0.5)  * 4 * 0.1,
                  0.05 + (t - 0.75) * 4 * 0.05)))
    out = np.zeros((*t.shape, 4), dtype=np.uint8)
    out[..., 0] = (np.clip(r2, 0, 1) * 255).astype(np.uint8)
    out[..., 1] = (np.clip(g2, 0, 1) * 255).astype(np.uint8)
    out[..., 2] = (np.clip(b2, 0, 1) * 255).astype(np.uint8)
    out[..., 3] = 255
    return out


def apply_aspect_to_pixmap(arr2d_u8, aspect_h_over_w: float) -> QPixmap:
    qi = gray_u8_to_qimage(arr2d_u8); pm = QPixmap.fromImage(qi)
    rows, cols = arr2d_u8.shape
    new_h = max(1, int(round(rows * aspect_h_over_w)))
    if new_h == rows: return pm
    return pm.scaled(cols, new_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)


# ── XAD histogram ─────────────────────────────────────────────────────────────
class XADHistogram:
    def __init__(self):
        self.qimg = None; self.cs_lim = XAD_CS_LIM; self.pe_lim = XAD_PE_LIM
        self.bins = 512

    def build(self, HUpe, HUcs, gauss, sl=None):
        H, W, N = HUpe.shape
        rows = slice(PHILIPS_ROWS, None) if H > PHILIPS_ROWS else slice(None)
        pf = HUpe[rows, :, sl].reshape(-1) if sl is not None else HUpe[rows].reshape(-1)
        cf = HUcs[rows, :, sl].reshape(-1) if sl is not None else HUcs[rows].reshape(-1)
        ok = np.isfinite(pf) & np.isfinite(cf); pf, cf = pf[ok], cf[ok]
        xl, xh = XAD_CS_LIM; yl, yh = XAD_PE_LIM
        self.cs_lim = XAD_CS_LIM; self.pe_lim = XAD_PE_LIM
        h, _, _ = np.histogram2d(cf, pf, bins=self.bins, range=[[xl, xh], [yl, yh]])
        if gauss: h = gaussian_filter(h, sigma=1.5)
        h = h.T; log_h = np.log10(1 + h); mx = log_h.max()
        t = log_h / mx if mx > 0 else log_h
        rgba = colormap_magma(t[::-1, :])
        self.qimg = rgba_to_qimage(rgba)

    def cs_to_px(self, cs, img_w):
        xl, xh = self.cs_lim; return (cs - xl) / (xh - xl) * img_w
    def pe_to_py(self, pe, img_h):
        yl, yh = self.pe_lim; return (1.0 - (pe - yl) / (yh - yl)) * img_h
    def px_to_cs(self, px, img_w):
        xl, xh = self.cs_lim; return xl + px / img_w * (xh - xl)
    def py_to_pe(self, py, img_h):
        yl, yh = self.pe_lim; return yl + (1.0 - py / img_h) * (yh - yl)


def _fast_mask(poly_xy, pts):
    from matplotlib.path import Path as MPath
    p = MPath(poly_xy); ext = p.get_extents()
    inbox = ((pts[:, 0] >= ext.x0) & (pts[:, 0] <= ext.x1) &
             (pts[:, 1] >= ext.y0) & (pts[:, 1] <= ext.y1))
    idx = np.where(inbox)[0]
    if len(idx) == 0: return idx
    return idx[p.contains_points(pts[idx])]


# ── CT Histogram Colorbar ─────────────────────────────────────────────────────
class CTColorBar(QWidget):
    window_changed = pyqtSignal(float, float)

    _CBAR_W   = 18
    _HIST_W   = 70
    _HANDLE_H = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(self._CBAR_W + self._HIST_W + 4)
        self.setMinimumHeight(200)
        self._wc  = 40.0;  self._ww = 400.0
        self._hist_counts = np.zeros(200, dtype=np.float32)
        self._hist_edges  = np.linspace(-1000, 2000, 201)
        self._drag = None

    def set_data(self, arr2d):
        flat = arr2d[np.isfinite(arr2d)].ravel()
        if len(flat) == 0: return
        if len(flat) > 20000:
            flat = flat[::len(flat)//20000]
        lo, hi = np.percentile(flat, [0.1, 99.9])
        counts, edges = np.histogram(flat, bins=200, range=(lo, hi))
        self._hist_counts = counts.astype(np.float32)
        self._hist_edges  = edges
        self.update()

    def set_window(self, wc, ww):
        self._wc = wc; self._ww = ww; self.update()

    def _data_range(self):
        return float(self._hist_edges[0]), float(self._hist_edges[-1])

    def _hu_to_y(self, hu):
        lo, hi = self._data_range(); h = self.height()
        return int(h - (hu - lo) / max(hi - lo, 1) * h)

    def _y_to_hu(self, y):
        lo, hi = self._data_range(); h = self.height()
        return lo + (h - y) / h * (hi - lo)

    def paintEvent(self, ev):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing, False)
        w = self.width(); h = self.height()
        cbar_x = w - self._CBAR_W

        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0.0, QColor(255, 255, 255))
        grad.setColorAt(1.0, QColor(0,   0,   0))
        p.fillRect(cbar_x, 0, self._CBAR_W, h, grad)

        lo_d, hi_d = self._data_range()
        span = max(hi_d - lo_d, 1)
        mx = self._hist_counts.max()
        if mx > 0:
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(74, 127, 165, 180)))
            n = len(self._hist_counts)
            for i, cnt in enumerate(self._hist_counts):
                bin_lo = self._hist_edges[i]
                bin_hi = self._hist_edges[i+1]
                y1 = h - int((bin_hi - lo_d) / span * h)
                y2 = h - int((bin_lo - lo_d) / span * h)
                bh = max(1, y2 - y1)
                bw = int(cnt / mx * self._HIST_W)
                p.drawRect(0, y1, bw, bh)

        wlo = self._wc - self._ww / 2; whi = self._wc + self._ww / 2
        y_hi = self._hu_to_y(whi); y_lo = self._hu_to_y(wlo)
        p.fillRect(0, y_hi, w, y_lo - y_hi, QColor(255, 255, 255, 30))

        pen = QPen(QColor("#2a82da"), 2); p.setPen(pen)
        p.drawLine(0, y_hi, w, y_hi)
        p.drawLine(0, y_lo, w, y_lo)

        p.setPen(QColor(200, 200, 200))
        p.setFont(QFont("Monospace", 8))
        p.drawText(2, h - 24, f"WL {int(self._wc)}")
        p.drawText(2, h - 10, f"WW {int(self._ww)}")
        p.end()

    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton: return
        y = ev.pos().y()
        wlo = self._wc - self._ww / 2; whi = self._wc + self._ww / 2
        y_hi = self._hu_to_y(whi); y_lo = self._hu_to_y(wlo)
        if abs(y - y_hi) < 8: self._drag = 'hi'
        elif abs(y - y_lo) < 8: self._drag = 'lo'
        else: self._drag = 'mid'; self._drag_y0 = y; self._drag_wc0 = self._wc

    def mouseMoveEvent(self, ev):
        if self._drag is None: return
        hu = self._y_to_hu(ev.pos().y())
        wlo = self._wc - self._ww / 2; whi = self._wc + self._ww / 2
        if self._drag == 'hi':
            whi = hu; wlo_new = min(wlo, whi - 1)
            self._ww = whi - wlo_new; self._wc = (whi + wlo_new) / 2
        elif self._drag == 'lo':
            wlo = hu; whi_new = max(whi, wlo + 1)
            self._ww = whi_new - wlo; self._wc = (whi_new + wlo) / 2
        elif self._drag == 'mid':
            dy = ev.pos().y() - self._drag_y0
            lo_d, hi_d = self._data_range()
            span = max(hi_d - lo_d, 1)
            self._wc = self._drag_wc0 - dy / self.height() * span
        self.update(); self.window_changed.emit(self._wc, self._ww)

    def mouseReleaseEvent(self, ev): self._drag = None

    def setCursor_hint(self, ev):
        y = ev.pos().y()
        wlo = self._wc - self._ww / 2; whi = self._wc + self._ww / 2
        if abs(y - self._hu_to_y(whi)) < 8 or abs(y - self._hu_to_y(wlo)) < 8:
            self.setCursor(Qt.SizeVerCursor)
        else:
            self.setCursor(Qt.ArrowCursor)


# ── compute worker ────────────────────────────────────────────────────────────
class ComputeWorker(QThread):
    done  = pyqtSignal(object, object, object, object, object)
    error = pyqtSignal(str)

    def __init__(self, dv_hi, dv_lo, kh, kl, kr):
        super().__init__()
        self.dv_hi = dv_hi; self.dv_lo = dv_lo
        self.kh = kh; self.kl = kl; self.kr = kr

    def run(self):
        try:
            vh, vl, z_common = z_match_volumes(self.dv_hi, self.dv_lo)
            H, W, N = vl.shape
            pe = np.empty((H, W, N), dtype=np.float32)
            cs = np.empty_like(pe)
            for i in range(N):
                pe[:, :, i], cs[:, :, i] = get_HUpecs(
                    (self.kh, vh[:, :, i]), (self.kl, vl[:, :, i]), self.kr)
            self.done.emit(pe, cs, vl, z_common, self.dv_lo)
        except Exception as exc:
            self.error.emit(str(exc))


# ── ROI ───────────────────────────────────────────────────────────────────────
class ROI:
    _counter = 1

    def __init__(self, source, kind, color, view):
        self.source = source; self.kind = kind; self.color = color; self.view = view
        self.label  = f"ROI {ROI._counter}"; ROI._counter += 1
        self.verts  = None; self._cache = {}

    def clear_cache(self): self._cache.clear()


# ── Graphics items ────────────────────────────────────────────────────────────
class PixelOverlayItem(QGraphicsItem):
    def __init__(self, xi, yi, color, opacity=0.6):
        super().__init__()
        self._xi = xi.astype(np.int32); self._yi = yi.astype(np.int32)
        self._color = QColor(color)
        if len(xi):
            self._rect = QRectF(float(xi.min()), float(yi.min()),
                                float(xi.max()-xi.min()+1), float(yi.max()-yi.min()+1))
        else: self._rect = QRectF()
        self.setZValue(10); self.setOpacity(opacity)

    def boundingRect(self): return self._rect

    def paint(self, painter, option, widget=None):
        painter.setPen(Qt.NoPen); painter.setBrush(QBrush(self._color))
        path = QPainterPath()
        for x, y in zip(self._xi, self._yi):
            path.addRect(QRectF(float(x), float(y), 1.0, 1.0))
        painter.drawPath(path)

    def update_color(self, color): self._color = QColor(color); self.update()
    def update_opacity(self, opacity): self.setOpacity(opacity); self.update()


class ScatterItem(QGraphicsItem):
    def __init__(self, xs, ys, color, radius=SCATTER_R):
        super().__init__()
        self._pts = list(zip(xs.tolist(), ys.tolist()))
        self._pen = QPen(Qt.NoPen); self._brush = QBrush(QColor(color)); self._r = radius
        if self._pts:
            ax = [p[0] for p in self._pts]; ay = [p[1] for p in self._pts]
            self._rect = QRectF(min(ax)-radius, min(ay)-radius,
                                max(ax)-min(ax)+2*radius, max(ay)-min(ay)+2*radius)
        else: self._rect = QRectF()
        self.setZValue(10)

    def boundingRect(self): return self._rect

    def paint(self, painter, option, widget=None):
        painter.setPen(self._pen); painter.setBrush(self._brush); r = self._r
        for x, y in self._pts: painter.drawEllipse(QRectF(x-r, y-r, 2*r, 2*r))

    def update_color(self, color): self._brush = QBrush(QColor(color)); self.update()
    def update_opacity(self, opacity): self.setOpacity(opacity); self.update()


class MaterialPointItem(QGraphicsItem):
    _R    = 5.0
    _FONT = QFont("Monospace", 8)

    def __init__(self, px, py, name):
        super().__init__()
        self._px = float(px); self._py = float(py); self._name = name
        fm = QFontMetrics(self._FONT); tw = fm.horizontalAdvance(name); th = fm.height()
        self._rect = QRectF(px - self._R - 1, py - self._R - 1,
                            self._R * 2 + 10 + tw, max(self._R * 2 + 2, th + 2))
        self.setZValue(20)

    def boundingRect(self): return self._rect

    def paint(self, painter, option, widget=None):
        r = self._R
        painter.setPen(QPen(QColor("#222222"), 1.0))
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawEllipse(QRectF(self._px - r, self._py - r, 2*r, 2*r))
        pen = QPen(QColor("#ffffff"), 1); pen.setCosmetic(True); painter.setPen(pen)
        painter.setFont(self._FONT); fm = QFontMetrics(self._FONT)
        painter.drawText(int(self._px + r + 5), int(self._py + fm.ascent() / 2), self._name)


class MixtureLineItem(QGraphicsItem):
    _FONT = QFont("Monospace", 8)

    def __init__(self, px0, py0, px1, py1, name, color="#ffd166"):
        super().__init__()
        self._p0 = QPointF(float(px0), float(py0))
        self._p1 = QPointF(float(px1), float(py1))
        self._name = name
        self._color = QColor(color)
        fm = QFontMetrics(self._FONT)
        x0 = min(self._p0.x(), self._p1.x()); x1 = max(self._p0.x(), self._p1.x())
        y0 = min(self._p0.y(), self._p1.y()); y1 = max(self._p0.y(), self._p1.y())
        self._rect = QRectF(x0-12, y0-12, (x1-x0)+fm.horizontalAdvance(name)+24, (y1-y0)+fm.height()+24)
        self.setZValue(18)

    def boundingRect(self): return self._rect

    def paint(self, painter, option, widget=None):
        pen = QPen(self._color, 2.0, Qt.DashLine); pen.setCosmetic(True)
        painter.setPen(pen); painter.setBrush(Qt.NoBrush)
        painter.drawLine(self._p0, self._p1)
        painter.setPen(QPen(self._color, 1.2)); painter.setBrush(QBrush(self._color))
        r = 3.0
        painter.drawEllipse(self._p0, r, r); painter.drawEllipse(self._p1, r, r)
        painter.setFont(self._FONT); fm = QFontMetrics(self._FONT)
        mx = 0.5*(self._p0.x()+self._p1.x()); my = 0.5*(self._p0.y()+self._p1.y())
        tx = mx+6; ty = my-6
        bg = QRectF(tx-2, ty-fm.ascent()-1, fm.horizontalAdvance(self._name)+4, fm.height()+2)
        painter.setPen(Qt.NoPen); painter.setBrush(QColor(20,20,20,180)); painter.drawRect(bg)
        painter.setPen(QPen(QColor("#ffffff"), 1))
        painter.drawText(QPointF(tx, ty), self._name)

class ROIOutlineItem(QGraphicsPathItem):
    def __init__(self, verts, color):
        super().__init__()
        path = QPainterPath(); path.moveTo(verts[0, 0], verts[0, 1])
        for v in verts[1:]: path.lineTo(v[0], v[1])
        path.closeSubpath(); self.setPath(path)
        pen = QPen(QColor(color), 1.5); pen.setCosmetic(True)
        self.setPen(pen); self.setBrush(QBrush(Qt.NoBrush)); self.setZValue(9)

    def update_color(self, color):
        pen = QPen(QColor(color), 1.5); pen.setCosmetic(True); self.setPen(pen)
    def update_opacity(self, opacity): pass


# ── ImageView (zoom-persistent) ───────────────────────────────────────────────
class ImageView(QGraphicsView):
    roi_finished = pyqtSignal(np.ndarray)

    def __init__(self, mode="ct", parent=None):
        super().__init__(parent)
        self._mode = mode; self._histo = None
        self._wc = 40.0; self._ww = 400.0; self._img_w = 1; self._img_h = 1
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QBrush(QColor("#111111")))
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._bg_item = QGraphicsPixmapItem(); self._bg_item.setZValue(0)
        self.scene().addItem(self._bg_item)
        self._tool = ROI_POLYGON; self._drawing = False; self._poly_pts = []
        self._rubber = None; self._pan_start = None; self._overlays = []
        self._fitted = False

    def set_pixmap(self, pm):
        self._bg_item.setPixmap(pm); self._img_w = pm.width(); self._img_h = pm.height()
        self.scene().setSceneRect(QRectF(0, 0, self._img_w, self._img_h))
        if not self._fitted:
            self.fitInView(self._bg_item, Qt.KeepAspectRatio)
            self._fitted = True
        self.viewport().update()

    def reset_zoom(self):
        self._fitted = False
        self.fitInView(self._bg_item, Qt.KeepAspectRatio)
        self._fitted = True

    def set_ct_window(self, wc, ww): self._wc = wc; self._ww = ww; self.viewport().update()

    def add_overlay(self, item): self.scene().addItem(item); self._overlays.append(item)

    def remove_overlay(self, item):
        try: self.scene().removeItem(item); self._overlays.remove(item)
        except: pass

    def clear_overlays(self):
        for item in list(self._overlays):
            try: self.scene().removeItem(item)
            except: pass
        self._overlays.clear()

    def wheelEvent(self, ev):
        factor = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor); self.viewport().update()

    def set_tool(self, t): self._tool = t; self._cancel_drawing()

    def mousePressEvent(self, ev):
        if ev.button() in (Qt.MiddleButton,) or (
                ev.button() == Qt.RightButton and not self._drawing):
            self._pan_start = ev.pos(); self.setCursor(QCursor(Qt.ClosedHandCursor)); return
        if ev.button() == Qt.LeftButton:
            sp = self.mapToScene(ev.pos())
            if self._tool == ROI_POLYGON:
                self._poly_pts.append(sp); self._update_rubber()
            elif self._tool in (ROI_ELLIPSE, ROI_RECTANGLE):
                self._poly_pts = [sp]; self._drawing = True
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._pan_start is not None:
            d = ev.pos() - self._pan_start; self._pan_start = ev.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - d.y()); return
        sp = self.mapToScene(ev.pos())
        if self._tool == ROI_POLYGON and self._poly_pts: self._update_rubber(sp)
        elif self._drawing and self._poly_pts: self._update_rubber(sp)
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._pan_start is not None and ev.button() in (Qt.MiddleButton, Qt.RightButton):
            self._pan_start = None; self.setCursor(QCursor(Qt.ArrowCursor)); return
        if ev.button() == Qt.LeftButton:
            if self._tool in (ROI_ELLIPSE, ROI_RECTANGLE) and self._drawing:
                self._poly_pts.append(self.mapToScene(ev.pos())); self._finish_shape()
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == Qt.LeftButton and self._tool == ROI_POLYGON:
            if len(self._poly_pts) >= 3: self._finish_shape()
        super().mouseDoubleClickEvent(ev)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape: self._cancel_drawing()
        super().keyPressEvent(ev)

    def _update_rubber(self, cursor=None):
        if self._rubber:
            try: self.scene().removeItem(self._rubber)
            except: pass
            self._rubber = None
        pen = QPen(QColor("#ffffff"), 1.0, Qt.DashLine); pen.setCosmetic(True)
        path = QPainterPath()
        if self._tool == ROI_POLYGON:
            pts = self._poly_pts + ([cursor] if cursor else [])
            if len(pts) < 2: return
            path.moveTo(pts[0])
            for p in pts[1:]: path.lineTo(p)
            if cursor and len(self._poly_pts) >= 2: path.lineTo(self._poly_pts[0])
        elif self._tool == ROI_RECTANGLE and len(self._poly_pts) == 1 and cursor:
            path.addRect(QRectF(self._poly_pts[0], cursor).normalized())
        elif self._tool == ROI_ELLIPSE and len(self._poly_pts) == 1 and cursor:
            path.addEllipse(QRectF(self._poly_pts[0], cursor).normalized())
        item = self.scene().addPath(path, pen); item.setZValue(100); self._rubber = item

    def _finish_shape(self):
        if self._rubber:
            try: self.scene().removeItem(self._rubber)
            except: pass
            self._rubber = None
        pts = self._poly_pts
        if self._tool == ROI_POLYGON:
            verts = np.array([[p.x(), p.y()] for p in pts])
        elif self._tool == ROI_RECTANGLE and len(pts) == 2:
            x0,y0 = pts[0].x(),pts[0].y(); x1,y1 = pts[1].x(),pts[1].y()
            verts = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
        elif self._tool == ROI_ELLIPSE and len(pts) == 2:
            x0,y0=pts[0].x(),pts[0].y(); x1,y1=pts[1].x(),pts[1].y()
            cx,cy=(x0+x1)/2,(y0+y1)/2; rx,ry=abs(x1-x0)/2,abs(y1-y0)/2
            t=np.linspace(0,2*np.pi,120)
            verts=np.column_stack([cx+rx*np.cos(t),cy+ry*np.sin(t)])
        else:
            self._cancel_drawing(); return
        self._poly_pts=[]; self._drawing=False; self.roi_finished.emit(verts)

    def _cancel_drawing(self):
        self._poly_pts=[]; self._drawing=False
        if self._rubber:
            try: self.scene().removeItem(self._rubber)
            except: pass
            self._rubber=None

    def paintEvent(self, ev):
        super().paintEvent(ev)
        if self._bg_item.pixmap() is None or self._bg_item.pixmap().isNull(): return
        painter = QPainter(self.viewport())
        painter.setFont(TICK_FONT); painter.setPen(QPen(TICK_COLOR, 1))
        fm = QFontMetrics(TICK_FONT)
        vp_h = self.viewport().height()
        tl = self.mapFromScene(QPointF(0, 0))
        br = self.mapFromScene(QPointF(self._img_w, self._img_h))
        ix0=tl.x(); iy0=tl.y(); ix1=br.x(); iy1=br.y(); iw=ix1-ix0; ih=iy1-iy0
        if iw < 2 or ih < 2: painter.end(); return
        x_lo = 0 if self._mode=="ct" else (self._histo.cs_lim[0] if self._histo else XAD_CS_LIM[0])
        x_hi = self._img_w if self._mode=="ct" else (self._histo.cs_lim[1] if self._histo else XAD_CS_LIM[1])
        y_base = min(vp_h-2, iy1+2)
        for v in _nice_ticks(x_lo, x_hi, n=7):
            frac=(v-x_lo)/(x_hi-x_lo) if x_hi!=x_lo else 0; px=ix0+frac*iw
            if px<ix0-2 or px>ix1+2: continue
            painter.drawLine(int(px),int(iy1),int(px),int(iy1)+TICK_LEN)
            label=f"{int(v)}" if abs(v)<1e4 else f"{v:.1e}"; lw=fm.horizontalAdvance(label)
            painter.drawText(int(px-lw/2),int(y_base+TICK_LEN+fm.ascent()),label)
        if self._mode=="ct":
            y_lo=self._wc-self._ww/2; y_hi=self._wc+self._ww/2
        else:
            y_lo=self._histo.pe_lim[0] if self._histo else XAD_PE_LIM[0]
            y_hi=self._histo.pe_lim[1] if self._histo else XAD_PE_LIM[1]
        x_base = max(0, ix0-2)
        for v in _nice_ticks(y_lo, y_hi, n=6):
            frac=(y_hi-v)/(y_hi-y_lo) if (self._mode=="ct" and y_hi!=y_lo) else (1-(v-y_lo)/(y_hi-y_lo) if y_hi!=y_lo else 0)
            py=iy0+frac*ih
            if py<iy0-2 or py>iy1+2: continue
            painter.drawLine(int(ix0)-TICK_LEN,int(py),int(ix0),int(py))
            label=f"{int(v)}" if abs(v)<1e4 else f"{v:.1e}"; lw=fm.horizontalAdvance(label)
            painter.drawText(int(x_base-lw-TICK_LEN-1),int(py+fm.ascent()/2),label)
        painter.setPen(QPen(QColor(200,200,200),1))
        bold=QFont(TICK_FONT); bold.setBold(True); painter.setFont(bold); fm2=QFontMetrics(bold)
        xl=("voxels" if self._mode=="ct" else "HUcs")
        yl=("voxels" if self._mode=="ct" else "HUpe")
        lw=fm2.horizontalAdvance(xl)
        painter.drawText(int(ix0+iw/2-lw/2),int(y_base+TICK_LEN+fm.ascent()+fm2.height()+2),xl)
        painter.save(); painter.translate(max(2,ix0-TICK_LEN-fm.maxWidth()-4),iy0+ih/2)
        painter.rotate(-90); lw2=fm2.horizontalAdvance(yl)
        painter.drawText(int(-lw2/2),int(fm2.ascent()/2),yl); painter.restore()
        painter.setFont(TICK_FONT); painter.end()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)


# ── XADViewer ─────────────────────────────────────────────────────────────────
class XADViewer(QWidget):
    roi_added = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._HUpe=self._HUcs=self._HUlo=None; self._z_locs=None
        self._dv_lo=None; self._dv_ct=None; self._histo=XADHistogram()
        self._sl=0; self._view=VIEW_AX; self._wc=40.0; self._ww=400.0
        self._gauss=True; self._per_slice=False; self._tool=ROI_POLYGON
        self._overlay_opacity=0.6
        self._color_cycle=cycle(DEFAULT_COLORS); self._next_color=next(self._color_cycle)
        self._rois=[]; self._roi_ct_outlines={}; self._roi_xad_outlines={}
        self._roi_ct_pixels={}; self._roi_xad_scatter={}
        self._mat_items: dict[str, MaterialPointItem] = {}
        self._fit_lines: dict = {}
        self._mixture_lines: dict = {}
        self._kh=100; self._kl=80; self._kr=80
        self._build_ui()

    def _build_ui(self):
        lay=QHBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(2)
        self.ct_view=ImageView(mode="ct"); self.xad_view=ImageView(mode="xad")
        self.xad_view._histo=self._histo
        self.ct_view.roi_finished.connect(self._on_ct_roi)
        self.xad_view.roi_finished.connect(self._on_xad_roi)

        self.colorbar = CTColorBar()
        self.colorbar.window_changed.connect(self._on_colorbar_window)

        ct_panel = QWidget()
        ct_lay = QHBoxLayout(ct_panel); ct_lay.setContentsMargins(0,0,0,0); ct_lay.setSpacing(2)
        ct_lay.addWidget(self.ct_view, stretch=1)
        ct_lay.addWidget(self.colorbar)

        splitter=QSplitter(Qt.Horizontal)
        splitter.addWidget(self._wrap(ct_panel,"CT"))
        splitter.addWidget(self._wrap(self.xad_view,"XAD Space"))
        splitter.setSizes([600,600]); lay.addWidget(splitter)

    def _on_colorbar_window(self, wc, ww):
        self._wc = wc; self._ww = ww
        self.ct_view.set_ct_window(wc, ww)
        self._refresh_ct_image_only()

    def _wrap(self, view, title):
        w=QWidget(); v=QVBoxLayout(w); v.setContentsMargins(0,0,0,0); v.setSpacing(1)
        lbl=QLabel(title); lbl.setAlignment(Qt.AlignCenter); lbl.setStyleSheet("color:#aaa;font-size:11px;")
        v.addWidget(lbl); v.addWidget(view,stretch=1); return w

    def _active_dv(self): return self._dv_ct if self._dv_ct else self._dv_lo

    def _aspect_for_view(self):
        dv=self._active_dv()
        if dv is None: return 1.0
        return {VIEW_AX:dv.ax_aspect, VIEW_CO:dv.cor_aspect, VIEW_SA:dv.sag_aspect}[self._view]

    def load_data(self, HUpe, HUcs, HUlo, z_locs, dv_lo, dv_ct=None, kh=100, kl=80, kr=80):
        self.clear_rois()
        self.clear_mixture_lines()
        self._HUpe=HUpe; self._HUcs=HUcs; self._HUlo=HUlo
        self._z_locs=z_locs; self._dv_lo=dv_lo; self._dv_ct=dv_ct
        self._kh=kh; self._kl=kl; self._kr=kr
        self._sl=0; self._view=VIEW_AX
        self._rebuild_xad_bg(); self._refresh_ct()
        self.ct_view.reset_zoom(); self.xad_view.reset_zoom()

    def set_tool(self, t):
        self._tool=t; self.ct_view.set_tool(t); self.xad_view.set_tool(t)

    def set_next_color(self, c): self._next_color=c

    def set_window(self, wc, ww):
        self._wc=wc; self._ww=ww
        self.ct_view.set_ct_window(wc,ww)
        self.colorbar.set_window(wc, ww)
        self._refresh_ct_image_only()

    def set_slice(self, idx):
        if self._HUlo is None: return
        idx=int(np.clip(idx,0,self.n_slices()-1))
        if idx==self._sl: return
        self._sl=idx
        if self._per_slice: self._rebuild_xad_bg()
        self._refresh_ct_image_only()
        self._refresh_roi_scatter()

    def set_view(self, v):
        if v==self._view: return
        self._view=v; self._sl=self.n_slices()//2
        self._refresh_ct()
        self._refresh_roi_overlays_for_view()
        self.ct_view.reset_zoom()

    def toggle_gauss(self, on):
        if on==self._gauss: return
        self._gauss=on
        if self._HUpe is not None: self._rebuild_xad_bg()

    def set_per_slice(self, on):
        if on==self._per_slice: return
        self._per_slice=on
        if self._HUpe is not None: self._rebuild_xad_bg()

    def set_overlay_opacity(self, opacity):
        self._overlay_opacity=opacity
        for d in (self._roi_ct_pixels, self._roi_xad_scatter):
            for item in d.values(): item.update_opacity(opacity)

    def n_slices(self):
        if self._HUlo is None: return 0
        H,W,N=self._HUlo.shape
        return {VIEW_AX:N, VIEW_SA:W, VIEW_CO:H}[self._view]

    def show_mixture_line(self, name, mat1, mat2, color="#ffd166"):
        self.hide_mixture_line(name)
        if self._histo.qimg is None: return
        try:
            hu_pe_1, hu_cs_1 = mat1.xad_point(self._kh, self._kl, self._kr)
            hu_pe_2, hu_cs_2 = mat2.xad_point(self._kh, self._kl, self._kr)
        except Exception as e:
            print(f"Mixture line error ({name}): {e}"); return
        bw = self._histo.qimg.width(); bh = self._histo.qimg.height()
        px0 = self._histo.cs_to_px(hu_cs_1, bw); py0 = self._histo.pe_to_py(hu_pe_1, bh)
        px1 = self._histo.cs_to_px(hu_cs_2, bw); py1 = self._histo.pe_to_py(hu_pe_2, bh)
        item = MixtureLineItem(px0, py0, px1, py1, name, color=color)
        self.xad_view.add_overlay(item); self._mixture_lines[name] = item

    def hide_mixture_line(self, name):
        item = self._mixture_lines.pop(name, None)
        if item: self.xad_view.remove_overlay(item)

    def clear_mixture_lines(self):
        for name in list(self._mixture_lines.keys()): self.hide_mixture_line(name)

    def draw_fit_line(self, roi, slope, intercept, x_lo, x_hi):
        old = self._fit_lines.pop(roi, None)
        if old: self.xad_view.remove_overlay(old)
        if self._histo.qimg is None: return
        bw = self._histo.qimg.width(); bh = self._histo.qimg.height()
        margin = (x_hi - x_lo) * 0.1
        x0 = x_lo - margin; x1 = x_hi + margin
        y0 = slope * x0 + intercept; y1 = slope * x1 + intercept
        px0 = self._histo.cs_to_px(x0, bw); py0 = self._histo.pe_to_py(y0, bh)
        px1 = self._histo.cs_to_px(x1, bw); py1 = self._histo.pe_to_py(y1, bh)
        path = QPainterPath(); path.moveTo(px0, py0); path.lineTo(px1, py1)
        item = QGraphicsPathItem(path)
        pen = QPen(QColor(roi.color), 1.5, Qt.DotLine); pen.setCosmetic(True)
        item.setPen(pen); item.setZValue(15)
        self.xad_view.add_overlay(item); self._fit_lines[roi] = item

    def remove_fit_line(self, roi):
        old = self._fit_lines.pop(roi, None)
        if old: self.xad_view.remove_overlay(old)

    def show_material(self, mat: Material):
        self._remove_mat_item(mat.name)
        if self._histo.qimg is None: return
        try:
            hu_pe, hu_cs = mat.xad_point(self._kh, self._kl, self._kr)
        except Exception as e:
            print(f"Material point error ({mat.name}): {e}"); return
        bw=self._histo.qimg.width(); bh=self._histo.qimg.height()
        px=self._histo.cs_to_px(hu_cs, bw); py=self._histo.pe_to_py(hu_pe, bh)
        item=MaterialPointItem(px, py, mat.name)
        self.xad_view.add_overlay(item); self._mat_items[mat.name]=item

    def hide_material(self, name): self._remove_mat_item(name)

    def _remove_mat_item(self, name):
        item=self._mat_items.pop(name,None)
        if item: self.xad_view.remove_overlay(item)

    def remove_roi(self, roi):
        for d, view in ((self._roi_ct_outlines,self.ct_view),(self._roi_xad_outlines,self.xad_view),
                        (self._roi_ct_pixels,self.ct_view),(self._roi_xad_scatter,self.xad_view)):
            item=d.pop(roi,None)
            if item: view.remove_overlay(item)
        self.remove_fit_line(roi)
        roi.clear_cache()
        if roi in self._rois: self._rois.remove(roi)

    def update_roi_color(self, roi, c):
        roi.color=c
        for d in (self._roi_ct_outlines,self._roi_xad_outlines,
                  self._roi_ct_pixels,self._roi_xad_scatter):
            item=d.get(roi)
            if item: item.update_color(c)

    def clear_rois(self):
        for roi in list(self._rois): self.remove_roi(roi)
        self._rois.clear(); self._fit_lines.clear(); ROI._counter=1

    def save_figure(self, path):
        pm_ct=self.ct_view.grab(); pm_xad=self.xad_view.grab()
        w=pm_ct.width()+pm_xad.width(); h=max(pm_ct.height(),pm_xad.height())
        out=QPixmap(w,h); out.fill(QColor("#1e1e1e")); p=QPainter(out)
        p.drawPixmap(0,0,pm_ct); p.drawPixmap(pm_ct.width(),0,pm_xad); p.end(); out.save(path)

    def _xad_slice_for_view(self):
        if self._HUlo is None: return None
        H,W,N = self._HUlo.shape
        return self._sl if self._view==VIEW_AX else N//2

    def _rebuild_xad_bg(self):
        sl=self._xad_slice_for_view() if self._per_slice else None
        self._histo.build(self._HUpe,self._HUcs,self._gauss,sl=sl)
        self.xad_view._histo=self._histo
        pm=QPixmap.fromImage(self._histo.qimg); self.xad_view.set_pixmap(pm)
        for roi in self._rois:
            if roi.source=="xad": roi.clear_cache()

    def _get_ct_arr(self, sl=None):
        sl=self._sl if sl is None else sl
        vol=self._dv_ct.vol if self._dv_ct else self._HUlo
        if vol is None: return np.zeros((512,512),dtype=np.float32)
        H,W,N=vol.shape
        if self._view==VIEW_AX:
            arr=vol[:,:,np.clip(sl,0,N-1)].copy()
            if H>PHILIPS_ROWS: arr[:PHILIPS_ROWS,:] = -2000
        elif self._view==VIEW_CO:
            arr=vol[np.clip(sl,0,H-1),:,:].T
        else:
            arr=vol[:,np.clip(sl,0,W-1),:].T
        return arr

    def _refresh_ct(self):
        arr=self._get_ct_arr(); u8=hu_to_uint8(arr,self._wc,self._ww)
        pm=apply_aspect_to_pixmap(u8,self._aspect_for_view()); self.ct_view.set_pixmap(pm)
        self.colorbar.set_data(arr); self.colorbar.set_window(self._wc, self._ww)

    def _refresh_ct_image_only(self):
        arr=self._get_ct_arr(); u8=hu_to_uint8(arr,self._wc,self._ww)
        pm=apply_aspect_to_pixmap(u8,self._aspect_for_view()); self.ct_view.set_pixmap(pm)
        self.colorbar.set_data(arr)

    def _slice_cs_pe(self, sl):
        H,W,N = self._HUcs.shape
        cs=self._HUcs[:,:,sl].copy().flatten(); pe=self._HUpe[:,:,sl].copy().flatten()
        if H>PHILIPS_ROWS: cs[:PHILIPS_ROWS*W]=np.nan; pe[:PHILIPS_ROWS*W]=np.nan
        return cs, pe

    def _refresh_roi_overlays_for_view(self):
        for roi in self._rois:
            for d, view in ((self._roi_xad_scatter,self.xad_view),(self._roi_ct_pixels,self.ct_view)):
                item=d.pop(roi,None)
                if item: view.remove_overlay(item)
        for roi in self._rois:
            vis=(roi.view==self._view)
            for d in (self._roi_ct_outlines,self._roi_xad_outlines):
                item=d.get(roi)
                if item: item.setVisible(vis)
        self._refresh_roi_scatter()

    def _refresh_roi_scatter(self):
        sl=self._sl
        for roi in self._rois:
            for d, view in ((self._roi_xad_scatter,self.xad_view),(self._roi_ct_pixels,self.ct_view)):
                item=d.pop(roi,None)
                if item: view.remove_overlay(item)
            if roi.view!=self._view: continue
            if roi.source=="ct":
                xs,ys=self._ct_roi_to_xad_pts(roi,sl)
                if len(xs):
                    item=self._make_xad_scatter(xs,ys,roi.color)
                    self.xad_view.add_overlay(item); self._roi_xad_scatter[roi]=item
            else:
                xi,yi=self._xad_roi_to_ct_pts(roi,sl)
                if len(xi):
                    item=self._make_ct_pixels(xi,yi,roi.color)
                    self.ct_view.add_overlay(item); self._roi_ct_pixels[roi]=item

    def _pm_to_vol_coords_co(self, v):
        dv=self._active_dv(); asp=dv.cor_aspect if dv else 1.0
        return np.column_stack([v[:,0], v[:,1]/asp])

    def _pm_to_vol_coords_sa(self, v):
        dv=self._active_dv(); asp=dv.sag_aspect if dv else 1.0
        return np.column_stack([v[:,0], v[:,1]/asp])

    def _ct_roi_to_xad_pts(self, roi, sl):
        ck=(sl,self._view)
        if ck in roi._cache: return roi._cache[ck]
        H,W,N = self._HUcs.shape
        if self._view==VIEW_AX:
            px,py=np.meshgrid(np.arange(W),np.arange(H))
            pc=np.vstack((px.flatten(),py.flatten())).T
            cs_f,pe_f=self._slice_cs_pe(sl)
            ind=_fast_mask(roi.verts,pc); v=np.isfinite(cs_f[ind])&np.isfinite(pe_f[ind])
            xs,ys=cs_f[ind][v],pe_f[ind][v]
        elif self._view==VIEW_CO:
            vc=self._pm_to_vol_coords_co(roi.verts)
            cc,aa=np.meshgrid(np.arange(W),np.arange(N))
            pts=np.vstack((cc.flatten(),aa.flatten())).T
            ind=_fast_mask(vc,pts)
            if len(ind)==0: roi._cache[ck]=(np.array([]),np.array([])); return np.array([]),np.array([])
            cs_list,pe_list=[],[]
            for fi in ind:
                a_sl=int(pts[fi,1]); col=int(pts[fi,0])
                cv=self._HUcs[sl,col,a_sl]; pv=self._HUpe[sl,col,a_sl]
                if np.isfinite(cv) and np.isfinite(pv): cs_list.append(cv); pe_list.append(pv)
            xs=np.array(cs_list,dtype=np.float32); ys=np.array(pe_list,dtype=np.float32)
        else:
            vs=self._pm_to_vol_coords_sa(roi.verts)
            rr,aa=np.meshgrid(np.arange(H),np.arange(N))
            pts=np.vstack((rr.flatten(),aa.flatten())).T
            ind=_fast_mask(vs,pts)
            if len(ind)==0: roi._cache[ck]=(np.array([]),np.array([])); return np.array([]),np.array([])
            cs_list,pe_list=[],[]
            for fi in ind:
                a_sl=int(pts[fi,1]); row=int(pts[fi,0])
                cv=self._HUcs[row,sl,a_sl]; pv=self._HUpe[row,sl,a_sl]
                if np.isfinite(cv) and np.isfinite(pv): cs_list.append(cv); pe_list.append(pv)
            xs=np.array(cs_list,dtype=np.float32); ys=np.array(pe_list,dtype=np.float32)
        roi._cache[ck]=(xs,ys); return xs,ys

    def _xad_roi_to_ct_pts(self, roi, sl):
        ck=(sl,self._view)
        if ck in roi._cache: return roi._cache[ck]
        H,W,N = self._HUcs.shape
        bw=self._histo.qimg.width(); bh=self._histo.qimg.height()
        vcp=np.column_stack([self._histo.px_to_cs(roi.verts[:,0],bw),
                             self._histo.py_to_pe(roi.verts[:,1],bh)])
        if self._view==VIEW_AX:
            cs_f,pe_f=self._slice_cs_pe(sl); xc=np.vstack((cs_f,pe_f)).T
            ind=_fast_mask(vcp,xc); ok=np.isfinite(cs_f[ind])&np.isfinite(pe_f[ind])
            ind=ind[ok]; xi,yi=ind%W,ind//W
        elif self._view==VIEW_CO:
            dv=self._active_dv(); asp=dv.cor_aspect if dv else 1.0
            xi_l,yi_l=[],[]
            for a_sl in range(N):
                cs_f=self._HUcs[sl,:,a_sl]; pe_f=self._HUpe[sl,:,a_sl]
                pts=np.vstack((cs_f,pe_f)).T; ind=_fast_mask(vcp,pts)
                ok=np.isfinite(cs_f[ind])&np.isfinite(pe_f[ind])
                for col in ind[ok]: xi_l.append(int(col)); yi_l.append(int(round(a_sl*asp)))
            xi=np.array(xi_l,dtype=np.int32); yi=np.array(yi_l,dtype=np.int32)
        else:
            dv=self._active_dv(); asp=dv.sag_aspect if dv else 1.0
            xi_l,yi_l=[],[]
            for a_sl in range(N):
                cs_f=self._HUcs[:,sl,a_sl]; pe_f=self._HUpe[:,sl,a_sl]
                pts=np.vstack((cs_f,pe_f)).T; ind=_fast_mask(vcp,pts)
                ok=np.isfinite(cs_f[ind])&np.isfinite(pe_f[ind])
                for row in ind[ok]: xi_l.append(int(row)); yi_l.append(int(round(a_sl*asp)))
            xi=np.array(xi_l,dtype=np.int32); yi=np.array(yi_l,dtype=np.int32)
        roi._cache[ck]=(xi,yi); return xi,yi

    def _make_xad_scatter(self, xs_cs, ys_pe, color):
        bw=self._histo.qimg.width(); bh=self._histo.qimg.height()
        px=np.array([self._histo.cs_to_px(v,bw) for v in xs_cs])
        py=np.array([self._histo.pe_to_py(v,bh) for v in ys_pe])
        item=ScatterItem(px,py,color); item.setOpacity(self._overlay_opacity); return item

    def _make_ct_pixels(self, xi, yi, color):
        return PixelOverlayItem(xi,yi,color,self._overlay_opacity)

    def _new_color(self):
        c=self._next_color; self._next_color=next(self._color_cycle); return c

    def _on_ct_roi(self, verts):
        if self._HUlo is None: return
        color=self._new_color(); roi=ROI("ct",self._tool,color,self._view)
        roi.verts=verts; self._rois.append(roi)
        outline=ROIOutlineItem(verts,color)
        self.ct_view.add_overlay(outline); self._roi_ct_outlines[roi]=outline
        xs,ys=self._ct_roi_to_xad_pts(roi,self._sl)
        if len(xs):
            item=self._make_xad_scatter(xs,ys,color)
            self.xad_view.add_overlay(item); self._roi_xad_scatter[roi]=item
        self.roi_added.emit(roi)

    def _on_xad_roi(self, verts):
        if self._HUlo is None: return
        color=self._new_color(); roi=ROI("xad",self._tool,color,self._view)
        roi.verts=verts; self._rois.append(roi)
        outline=ROIOutlineItem(verts,color)
        self.xad_view.add_overlay(outline); self._roi_xad_outlines[roi]=outline
        xi,yi=self._xad_roi_to_ct_pts(roi,self._sl)
        if len(xi):
            item=self._make_ct_pixels(xi,yi,color)
            self.ct_view.add_overlay(item); self._roi_ct_pixels[roi]=item
        self.roi_added.emit(roi)





class FitResultsPanel(QWidget):
    fit_computed = pyqtSignal(object, float, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(4,4,4,4); lay.setSpacing(4)
        lay.addWidget(QLabel("Linear Fits"))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["ROI", "Slope", "Intercept", "R²", "N"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1,2,3,4):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        lay.addWidget(self.table)
        self._rows: dict = {}

    def update_fit(self, roi, xs_cs, ys_pe):
        if len(xs_cs) < 2:
            self.remove_roi(roi)
            return

        x = np.array(xs_cs, dtype=np.float64)
        y = np.array(ys_pe, dtype=np.float64)
        xm = x.mean(); ym = y.mean()
        ssxx = ((x - xm)**2).sum()
        ssxy = ((x - xm)*(y - ym)).sum()
        slope = ssxy / ssxx if ssxx > 0 else 0.0
        intercept = ym - slope * xm
        y_pred = slope * x + intercept
        sst = ((y - ym)**2).sum()
        sse = ((y - y_pred)**2).sum()
        r2 = 1 - sse/sst if sst > 0 else 1.0
        n = len(x)

        if roi in self._rows:
            r = self._rows[roi]
        else:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self._rows[roi] = r

        color = QColor(roi.color)
        def _item(txt):
            it = QTableWidgetItem(txt)
            it.setForeground(color)
            return it

        self.table.setItem(r, 0, _item(roi.label))
        self.table.setItem(r, 1, _item(f"{slope:.4f}"))
        self.table.setItem(r, 2, _item(f"{intercept:.1f}"))
        self.table.setItem(r, 3, _item(f"{r2:.4f}"))
        self.table.setItem(r, 4, _item(str(n)))

        x_lo = float(x.min())
        x_hi = float(x.max())
        self.fit_computed.emit(roi, slope, intercept, x_lo, x_hi)

    def remove_roi(self, roi):
        r = self._rows.pop(roi, None)
        if r is None:
            return
        self.table.removeRow(r)
        self._rows = {k: (v-1 if v > r else v) for k, v in self._rows.items()}

    def update_label(self, roi, label):
        r = self._rows.get(roi)
        if r is not None:
            it = self.table.item(r, 0)
            if it:
                it.setText(label)

    def clear(self):
        self.table.setRowCount(0)
        self._rows.clear()

# ── ROI widgets ───────────────────────────────────────────────────────────────
RCOL_COLOR,RCOL_LABEL,RCOL_SOURCE,RCOL_KIND,RCOL_FIT,RCOL_DEL=0,1,2,3,4,5

class ROITable(QWidget):
    color_changed=pyqtSignal(object,str); label_changed=pyqtSignal(object,str)
    delete_clicked=pyqtSignal(object); fit_requested=pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent); self._rois=[]
        lay=QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self.table=QTableWidget(0,6)
        self.table.setHorizontalHeaderLabels(["","Label","Src","Kind","Fit",""])
        hh=self.table.horizontalHeader()
        hh.setSectionResizeMode(RCOL_COLOR,QHeaderView.Fixed); self.table.setColumnWidth(RCOL_COLOR,30)
        hh.setSectionResizeMode(RCOL_LABEL,QHeaderView.Stretch)
        hh.setSectionResizeMode(RCOL_SOURCE,QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(RCOL_KIND,QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(RCOL_FIT,QHeaderView.Fixed); self.table.setColumnWidth(RCOL_FIT,36)
        hh.setSectionResizeMode(RCOL_DEL,QHeaderView.Fixed); self.table.setColumnWidth(RCOL_DEL,30)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked|QAbstractItemView.EditKeyPressed)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True); lay.addWidget(self.table)
        self.table.itemChanged.connect(self._on_item_changed)

    def add_roi(self, roi):
        self._rois.append(roi); r=self.table.rowCount(); self.table.insertRow(r)
        btn_c=QPushButton(); btn_c.setFixedSize(22,18)
        btn_c.setStyleSheet(f"background:{roi.color};border:1px solid #666;")
        btn_c.clicked.connect(lambda _,ro=roi: self._pick(ro))
        self.table.setCellWidget(r,RCOL_COLOR,btn_c)
        self.table.setItem(r,RCOL_LABEL,QTableWidgetItem(roi.label))
        self.table.setItem(r,RCOL_SOURCE,QTableWidgetItem(roi.source))
        self.table.setItem(r,RCOL_KIND,QTableWidgetItem(roi.kind))
        btn_f=QPushButton("~"); btn_f.setFixedSize(30,18); btn_f.setToolTip("Linear fit")
        btn_f.setEnabled(roi.source=="ct")
        btn_f.clicked.connect(lambda _,ro=roi: self.fit_requested.emit(ro))
        self.table.setCellWidget(r,RCOL_FIT,btn_f)
        btn_d=QPushButton("✕"); btn_d.setFixedSize(22,18)
        btn_d.clicked.connect(lambda _,ro=roi: self._del(ro))
        self.table.setCellWidget(r,RCOL_DEL,btn_d)

    def _id_row(self, roi):
        for i,r in enumerate(self._rois):
            if r is roi: return i
        return -1

    def _on_item_changed(self, item):
        if item.column()==RCOL_LABEL:
            r=item.row()
            if 0<=r<len(self._rois):
                roi=self._rois[r]; roi.label=item.text(); self.label_changed.emit(roi,item.text())

    def _pick(self, roi):
        col=QColorDialog.getColor(QColor(roi.color),self,"ROI colour")
        if not col.isValid(): return
        h=col.name(); roi.color=h; r=self._id_row(roi)
        if r>=0:
            w=self.table.cellWidget(r,RCOL_COLOR)
            if w: w.setStyleSheet(f"background:{h};border:1px solid #666;")
        self.color_changed.emit(roi,h)

    def _del(self, roi):
        r=self._id_row(roi)
        if r<0: return
        self.table.removeRow(r); self._rois.pop(r); self.delete_clicked.emit(roi)

    def clear(self): self.table.setRowCount(0); self._rois.clear()


class ROIPanel(QWidget):
    color_changed=pyqtSignal(object,str); label_changed=pyqtSignal(object,str)
    delete_clicked=pyqtSignal(object); fit_requested=pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay=QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(2)
        lay.addWidget(QLabel("ROIs"))
        self.tabs=QTabWidget(); self.tabs.setDocumentMode(True)
        self._tables: dict[str,ROITable]={}
        for v in ALL_VIEWS:
            tbl=ROITable(); tbl.color_changed.connect(self.color_changed)
            tbl.label_changed.connect(self.label_changed); tbl.delete_clicked.connect(self.delete_clicked)
            tbl.fit_requested.connect(self.fit_requested)
            self._tables[v]=tbl; self.tabs.addTab(tbl,v)
        lay.addWidget(self.tabs,stretch=1)

    def set_active_view(self, view):
        idx={v:i for i,v in enumerate(ALL_VIEWS)}.get(view,0); self.tabs.setCurrentIndex(idx)

    def add_roi(self, roi): self._tables[roi.view].add_roi(roi)

    def clear(self):
        for tbl in self._tables.values(): tbl.clear()


class MaterialsTab(QWidget):
    show_material = pyqtSignal(object)
    hide_material = pyqtSignal(str)
    materials_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._materials: dict[str,Material]={}; self._folder=""
        self._build_ui()

    def _build_ui(self):
        lay=QVBoxLayout(self); lay.setContentsMargins(8,8,8,8); lay.setSpacing(6)
        row=QHBoxLayout()
        btn=QPushButton("Open Folder…"); btn.clicked.connect(self._on_browse); row.addWidget(btn)
        self.lbl_folder=QLabel("No folder selected"); self.lbl_folder.setStyleSheet("color:#888;")
        row.addWidget(self.lbl_folder,stretch=1); lay.addLayout(row)
        self._list=QListWidget()
        self._list.itemChanged.connect(self._on_item_changed)
        lay.addWidget(self._list,stretch=1)
        note=QLabel("All .tsv files in the selected folder are loaded.\n"
                    "Check a material to display its theoretical XAD point.\n"
                    "Format: header rows 'material <name>' and 'rho <value>',\n"
                    "then data rows: keV  ra/rho  cs/rho  pe/rho")
        note.setWordWrap(True); note.setStyleSheet("color:#666;font-size:11px;"); lay.addWidget(note)

    @property
    def materials(self):
        return self._materials

    def _on_browse(self):
        folder=QFileDialog.getExistingDirectory(self,"Select folder containing .tsv material files")
        if not folder: return
        self._folder=folder; self.lbl_folder.setText(os.path.basename(folder)); self._reload()

    def _reload(self):
        self._list.blockSignals(True); self._list.clear(); self._materials.clear()
        for m in load_materials_from_folder(self._folder):
            self._materials[m.name]=m
            item=QListWidgetItem(m.name)
            item.setFlags(item.flags()|Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked); self._list.addItem(item)
        self._list.blockSignals(False)
        self.materials_changed.emit(self._materials)

    def _on_item_changed(self, item):
        name=item.text()
        if item.checkState()==Qt.Checked:
            mat=self._materials.get(name)
            if mat: self.show_material.emit(mat)
        else:
            self.hide_material.emit(name)

    def recheck_all_visible(self):
        for i in range(self._list.count()):
            item=self._list.item(i)
            if item.checkState()==Qt.Checked:
                mat=self._materials.get(item.text())
                if mat: self.show_material.emit(mat)

    def uncheck(self, name):
        for i in range(self._list.count()):
            item=self._list.item(i)
            if item.text()==name:
                self._list.blockSignals(True); item.setCheckState(Qt.Unchecked)
                self._list.blockSignals(False); return


class MixturesTab(QWidget):
    show_mixture = pyqtSignal(str, str, str)
    hide_mixture = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._materials = {}
        self._default_pairs = [
            ("Soft Tissue — Iodine", "Soft Tissue", "Iodine"),
            ("Water — Iodine",       "Water",        "Iodine"),
            ("Water — Bone",         "Water",        "Bone"),
            ("Air — Water",          "Air",           "Water"),
        ]
        self._pairs = []
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(8,8,8,8); lay.setSpacing(6)

        add_row = QHBoxLayout()
        self._ed_name = QComboBox(); self._ed_name.setEditable(True)
        self._ed_name.setPlaceholderText("Line name")
        self._cbo_m1 = QComboBox(); self._cbo_m1.setMinimumWidth(80)
        self._cbo_m2 = QComboBox(); self._cbo_m2.setMinimumWidth(80)
        btn_add = QPushButton("Add"); btn_add.setFixedWidth(44)
        btn_add.clicked.connect(self._on_add_pair)
        add_row.addWidget(QLabel("Name:")); add_row.addWidget(self._ed_name, stretch=1)
        add_row.addWidget(QLabel("M1:")); add_row.addWidget(self._cbo_m1)
        add_row.addWidget(QLabel("M2:")); add_row.addWidget(self._cbo_m2)
        add_row.addWidget(btn_add)
        lay.addLayout(add_row)

        self._list = QListWidget()
        self._list.itemChanged.connect(self._on_item_changed)
        lay.addWidget(self._list, stretch=1)
        self._rebuild()

    def set_materials(self, materials_dict):
        self._materials = dict(materials_dict)
        names = sorted(self._materials.keys())
        for cbo in (self._cbo_m1, self._cbo_m2):
            cur = cbo.currentText(); cbo.blockSignals(True)
            cbo.clear(); cbo.addItems(names)
            idx = cbo.findText(cur)
            cbo.setCurrentIndex(idx if idx >= 0 else 0)
            cbo.blockSignals(False)
        self._rebuild()

    def _on_add_pair(self):
        name = self._ed_name.currentText().strip()
        m1   = self._cbo_m1.currentText()
        m2   = self._cbo_m2.currentText()
        if not name or not m1 or not m2 or m1 == m2: return
        if any(n == name for n,_,_ in self._pairs): return
        self._pairs.append((name, m1, m2))
        self._rebuild()

    def _rebuild(self):
        checked = {}
        for i in range(self._list.count()):
            it = self._list.item(i); checked[it.text()] = (it.checkState()==Qt.Checked)

        self._list.blockSignals(True); self._list.clear()

        for disp_name, m1, m2 in self._default_pairs + self._pairs:
            available = m1 in self._materials and m2 in self._materials
            label = disp_name if available else f"{disp_name}  ⚠ load {m1} + {m2}"
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if checked.get(disp_name, False) else Qt.Unchecked)
            item.setData(Qt.UserRole, (disp_name, m1, m2))
            if not available:
                item.setForeground(QColor("#666666"))
            self._list.addItem(item)

        self._list.blockSignals(False)

    def _on_item_changed(self, item):
        if not (item.flags() & Qt.ItemIsUserCheckable): return
        data = item.data(Qt.UserRole)
        if data is None: return
        canonical_name, m1, m2 = data
        if m1 not in self._materials or m2 not in self._materials: return
        if item.checkState() == Qt.Checked: self.show_mixture.emit(canonical_name, m1, m2)
        else: self.hide_mixture.emit(canonical_name)

    def recheck_all_visible(self):
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.checkState() == Qt.Checked:
                data = item.data(Qt.UserRole)
                if data is None: continue
                canonical_name, m1, m2 = data
                if m1 in self._materials and m2 in self._materials:
                    self.show_mixture.emit(canonical_name, m1, m2)

class FolderEntry:
    def __init__(self, name, folder):
        self.name=name; self.folder=folder; self.role="—"; self._dv=None
    def load(self):
        if self._dv is None: self._dv=read_dicom_folder(self.folder)
        return self._dv
    @property
    def n_slices(self): return self._dv.shape_hwn[2] if self._dv else "?"

COL_NAME,COL_FOLDER,COL_ROLE,COL_SLICES=0,1,2,3

class DataTab(QWidget):
    roles_changed=pyqtSignal(); ct_loaded=pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent); self.entries=[]; self._dv_ct=None; self._build_ui()

    def _build_ui(self):
        lay=QVBoxLayout(self); lay.setContentsMargins(8,8,8,8); lay.setSpacing(6)
        row=QHBoxLayout()
        b1=QPushButton("Add Folder…"); b1.clicked.connect(self._on_add); row.addWidget(b1)
        b1f=QPushButton("Add File…"); b1f.clicked.connect(self._on_add_file); row.addWidget(b1f)
        b2=QPushButton("Remove Selected"); b2.clicked.connect(self._on_remove); row.addWidget(b2)
        row.addStretch()
        b3=QPushButton("Load Conv. CT…"); b3.clicked.connect(self._on_load_ct); row.addWidget(b3)
        self.lbl_ct=QLabel("Conv. CT: —"); self.lbl_ct.setStyleSheet("color:#888;")
        row.addWidget(self.lbl_ct); lay.addLayout(row)
        self.table=QTableWidget(0,4)
        self.table.setHorizontalHeaderLabels(["Name","Folder","Role","Slices"])
        hh=self.table.horizontalHeader()
        hh.setSectionResizeMode(COL_NAME,QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(COL_FOLDER,QHeaderView.Stretch)
        hh.setSectionResizeMode(COL_ROLE,QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(COL_SLICES,QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False); lay.addWidget(self.table)
        note=QLabel("Add the two dual-energy DICOM folders. Tag one as High-keV and one as Low-keV.\n"
                    "Slices are z-matched automatically. Optionally load a conventional CT.")
        note.setWordWrap(True); note.setStyleSheet("color:#888;"); lay.addWidget(note)

    def _on_add(self):
        folder=QFileDialog.getExistingDirectory(self,"Select DICOM folder")
        if folder: self._add_path(folder)

    def _on_add_file(self):
        path,_=QFileDialog.getOpenFileName(self,"Select DICOM file","","DICOM (*.dcm);;All files (*)")
        if path: self._add_path(path)

    def _add_path(self, path):
        name=os.path.basename(path); entry=FolderEntry(name,path)
        try: entry.load()
        except Exception as exc: QMessageBox.critical(self,"Load error",str(exc)); return
        self.entries.append(entry); r=self.table.rowCount(); self.table.insertRow(r)
        self.table.setItem(r,COL_NAME,QTableWidgetItem(name))
        self.table.setItem(r,COL_FOLDER,QTableWidgetItem(path))
        self.table.setItem(r,COL_SLICES,QTableWidgetItem(str(entry.n_slices)))
        cb=QComboBox(); cb.addItems(["—","High","Low"])
        cb.currentTextChanged.connect(lambda txt,e=entry: self._on_role(e,txt))
        self.table.setCellWidget(r,COL_ROLE,cb)

    def _on_load_ct(self):
        folder=QFileDialog.getExistingDirectory(self,"Select conventional CT folder")
        if not folder:
            folder,_=QFileDialog.getOpenFileName(self,"Or select a single CT DICOM file","","DICOM (*.dcm);;All files (*)")
        if not folder: return
        try:
            self._dv_ct=read_dicom_folder(folder)
            self.lbl_ct.setText(f"Conv. CT: {os.path.basename(folder)}"); self.ct_loaded.emit(self._dv_ct)
        except Exception as exc: QMessageBox.critical(self,"Load error",str(exc))

    def _on_remove(self):
        rows=sorted({i.row() for i in self.table.selectedIndexes()},reverse=True)
        for r in rows: self.table.removeRow(r); self.entries.pop(r)
        self.roles_changed.emit()

    def _on_role(self, entry, text):
        if text in ("High","Low"):
            for r,other in enumerate(self.entries):
                if other is not entry and other.role==text:
                    other.role="—"; w=self.table.cellWidget(r,COL_ROLE)
                    if w: w.blockSignals(True); w.setCurrentText("—"); w.blockSignals(False)
        entry.role=text; self.roles_changed.emit()

    def get_hi_lo(self):
        hi=next((e for e in self.entries if e.role=="High"),None)
        lo=next((e for e in self.entries if e.role=="Low"),None)
        return hi,lo


# ── dark palette ──────────────────────────────────────────────────────────────
def _make_dark_palette():
    pal=QPalette()
    dark=QColor(45,45,45); darker=QColor(30,30,30); darkest=QColor(18,18,18)
    mid=QColor(60,60,60); light=QColor(80,80,80); bright=QColor(200,200,200)
    text=QColor(220,220,220); accent=QColor(42,130,218); acc_txt=QColor(255,255,255)
    roles=[(QPalette.Window,dark),(QPalette.WindowText,text),(QPalette.Base,darker),
           (QPalette.AlternateBase,darkest),(QPalette.ToolTipBase,dark),(QPalette.ToolTipText,text),
           (QPalette.Text,text),(QPalette.Button,mid),(QPalette.ButtonText,text),
           (QPalette.BrightText,bright),(QPalette.Link,accent),(QPalette.Highlight,accent),
           (QPalette.HighlightedText,acc_txt),(QPalette.Light,light),(QPalette.Midlight,mid),
           (QPalette.Dark,darkest),(QPalette.Mid,mid),(QPalette.Shadow,darkest)]
    for role,color in roles:
        pal.setColor(QPalette.Active,role,color); pal.setColor(QPalette.Inactive,role,color)
    for role,color in roles:
        pal.setColor(QPalette.Disabled,role,QColor(max(color.red()-40,0),max(color.green()-40,0),max(color.blue()-40,0)))
    return pal

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XAD van Cas")
        self._worker=None; self._build_menu(); self._build_ui(); self._fit_to_screen()

    def _fit_to_screen(self):
        screen=QApplication.primaryScreen().availableGeometry()
        self.resize(screen.width(),screen.height()); self.move(screen.topLeft())

    def _build_menu(self):
        mb=self.menuBar(); fm=mb.addMenu("File")
        a1=QAction("Save Figure…",self,shortcut="Ctrl+S"); a1.triggered.connect(self._save); fm.addAction(a1)

    def _build_ui(self):
        cw=QWidget(); self.setCentralWidget(cw)
        root=QHBoxLayout(cw); root.setContentsMargins(4,4,4,4); root.setSpacing(4)
        self.tabs=QTabWidget(); self.tabs.setDocumentMode(True)
        root.addWidget(self.tabs,stretch=1)

        vw=QWidget(); vl=QVBoxLayout(vw); vl.setContentsMargins(4,4,4,4); vl.setSpacing(4)
        self.tabs.addTab(vw,"XAD")

        top=QHBoxLayout(); top.setSpacing(8)
        self.btn_compute=QPushButton("Compute XAD graph")
        self.btn_compute.clicked.connect(self._on_compute); top.addWidget(self.btn_compute)
        top.addWidget(_vline()); top.addWidget(QLabel("Ref keV:"))
        self.spin_ref=QSpinBox(); self.spin_ref.setRange(10,500); self.spin_ref.setValue(80)
        top.addWidget(self.spin_ref)
        top.addWidget(_vline()); top.addWidget(QLabel("View:"))
        self._vbg=QButtonGroup(self); self._vbg.setExclusive(True)
        for v in ALL_VIEWS:
            btn=QPushButton(v); btn.setCheckable(True); btn.setFixedWidth(38)
            if v==VIEW_AX: btn.setChecked(True)
            self._vbg.addButton(btn); btn.clicked.connect(lambda chk,vv=v: self._on_view(vv))
            top.addWidget(btn)
        top.addWidget(_vline()); top.addWidget(QLabel("WL:"))
        self.spin_wc=QDoubleSpinBox(); self.spin_wc.setRange(-1000,2000)
        self.spin_wc.setValue(150); self.spin_wc.setFixedWidth(70); top.addWidget(self.spin_wc)
        top.addWidget(QLabel("WW:"))
        self.spin_ww=QDoubleSpinBox(); self.spin_ww.setRange(1,1500)
        self.spin_ww.setValue(600); self.spin_ww.setFixedWidth(70); top.addWidget(self.spin_ww)
        self.spin_wc.valueChanged.connect(self._on_wnd); self.spin_ww.valueChanged.connect(self._on_wnd)
        self.cbo_preset=QComboBox(); self.cbo_preset.addItem("Preset…")
        for n in WW_PRESETS: self.cbo_preset.addItem(n)
        self.cbo_preset.currentTextChanged.connect(self._on_preset); top.addWidget(self.cbo_preset)
        top.addWidget(_vline()); top.addWidget(QLabel("Blur:"))
        self.chk_gauss=QCheckBox("On"); self.chk_gauss.setChecked(True)
        self.chk_gauss.stateChanged.connect(self._on_gauss); top.addWidget(self.chk_gauss)
        top.addWidget(_vline()); top.addWidget(QLabel("XAD:"))
        self.chk_per_slice=QCheckBox("Per-slice"); self.chk_per_slice.setChecked(False)
        self.chk_per_slice.stateChanged.connect(self._on_per_slice); top.addWidget(self.chk_per_slice)
        top.addWidget(_vline()); top.addWidget(QLabel("Overlay:"))
        self.slider_opacity=QSlider(Qt.Horizontal); self.slider_opacity.setRange(0,100)
        self.slider_opacity.setValue(100); self.slider_opacity.setFixedWidth(90)
        self.lbl_opacity=QLabel("100%"); self.lbl_opacity.setFixedWidth(36)
        self.slider_opacity.valueChanged.connect(self._on_opacity)
        top.addWidget(self.slider_opacity); top.addWidget(self.lbl_opacity)
        top.addWidget(_vline()); top.addWidget(QLabel("Next colour:"))
        self.btn_nc=QPushButton(); self.btn_nc.setFixedSize(28,22)
        self._set_nc_btn(DEFAULT_COLORS[0]); self.btn_nc.clicked.connect(self._on_pick_nc)
        top.addWidget(self.btn_nc)
        top.addStretch()
        self.lbl_status=QLabel("No data."); self.lbl_status.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        top.addWidget(self.lbl_status); vl.addLayout(top)

        self.viewer=XADViewer(); self.viewer.roi_added.connect(self._on_roi_added)
        vl.addWidget(self.viewer,stretch=1)

        bot=QHBoxLayout(); bot.setSpacing(8); bot.addWidget(QLabel("Slice:"))
        self.slider=QSlider(Qt.Horizontal); self.slider.setMinimum(0); self.slider.setMaximum(0)
        self.slider.setMinimumWidth(300); bot.addWidget(self.slider)
        self.spin_sl=QSpinBox(); self.spin_sl.setMinimum(0); self.spin_sl.setMaximum(0)
        bot.addWidget(self.spin_sl); bot.addStretch(); vl.addLayout(bot)
        self._st=QTimer(); self._st.setSingleShot(True); self._st.setInterval(60)
        self._st.timeout.connect(self._apply_slice)
        self.slider.valueChanged.connect(self._on_slider); self.spin_sl.valueChanged.connect(self._on_spin)

        self.data_tab=DataTab(); self.data_tab.roles_changed.connect(self._on_roles_changed)
        self.data_tab.ct_loaded.connect(lambda dv: None)
        self.tabs.addTab(self.data_tab,"Data")

        self.mat_tab=MaterialsTab()
        self.mat_tab.show_material.connect(self._on_show_mat)
        self.mat_tab.hide_material.connect(self._on_hide_mat)
        self.tabs.addTab(self.mat_tab,"Materials")

        self.mix_tab=MixturesTab()
        self.mat_tab.materials_changed.connect(self.mix_tab.set_materials)
        self.mix_tab.show_mixture.connect(self._on_show_mixture)
        self.mix_tab.hide_mixture.connect(self._on_hide_mixture)
        self.tabs.addTab(self.mix_tab,"Lines/Mixtures")

        rp=QWidget(); rp.setFixedWidth(300)
        rl=QVBoxLayout(rp); rl.setContentsMargins(4,4,4,4); rl.setSpacing(6)
        grp=QGroupBox("Draw tool"); gl=QHBoxLayout(grp); gl.setSpacing(4)
        self._tbg=QButtonGroup(self); self._tbg.setExclusive(True)
        for lbl,kind in [("Polygon",ROI_POLYGON),("Ellipse",ROI_ELLIPSE),("Rectangle",ROI_RECTANGLE)]:
            rb=QRadioButton(lbl)
            if kind==ROI_POLYGON: rb.setChecked(True)
            rb.clicked.connect(lambda chk,k=kind: self._on_tool(k))
            self._tbg.addButton(rb); gl.addWidget(rb)
        rl.addWidget(grp)
        self.roi_panel=ROIPanel()
        self.roi_panel.color_changed.connect(lambda roi,c: self.viewer.update_roi_color(roi,c))
        self.roi_panel.delete_clicked.connect(self._on_roi_deleted)
        self.roi_panel.fit_requested.connect(self._on_fit_requested)
        self.roi_panel.label_changed.connect(self._on_roi_label_changed)
        rl.addWidget(self.roi_panel,stretch=1)
        self.fit_panel=FitResultsPanel()
        self.fit_panel.fit_computed.connect(self._on_fit_computed)
        rl.addWidget(self.fit_panel)
        btn_ca=QPushButton("Delete All"); btn_ca.clicked.connect(self._on_clear); rl.addWidget(btn_ca)
        root.addWidget(rp)

    def _on_tool(self, k): self.viewer.set_tool(k)

    def _on_view(self, v):
        self.viewer.set_view(v); self.roi_panel.set_active_view(v)
        n=self.viewer.n_slices()
        for w in (self.slider,self.spin_sl):
            w.blockSignals(True); w.setMaximum(max(0,n-1)); w.setValue(self.viewer._sl); w.blockSignals(False)

    def _on_wnd(self): self.viewer.set_window(self.spin_wc.value(),self.spin_ww.value())
    def _on_gauss(self): self.viewer.toggle_gauss(self.chk_gauss.isChecked())
    def _on_per_slice(self): self.viewer.set_per_slice(self.chk_per_slice.isChecked())

    def _on_opacity(self, val):
        self.lbl_opacity.setText(f"{val}%"); self.viewer.set_overlay_opacity(val/100.0)

    def _on_preset(self, text):
        if text not in WW_PRESETS: return
        wc,ww=WW_PRESETS[text]
        for w,v in ((self.spin_wc,wc),(self.spin_ww,ww)):
            w.blockSignals(True); w.setValue(v); w.blockSignals(False)
        self.viewer.set_window(wc,ww)
        self.cbo_preset.blockSignals(True); self.cbo_preset.setCurrentIndex(0); self.cbo_preset.blockSignals(False)

    def _on_pick_nc(self):
        col=QColorDialog.getColor(QColor(self.viewer._next_color),self,"Next ROI colour")
        if not col.isValid(): return
        h=col.name(); self.viewer.set_next_color(h); self._set_nc_btn(h)

    def _set_nc_btn(self, h): self.btn_nc.setStyleSheet(f"background:{h};border:1px solid #666;")

    def _on_roi_added(self, roi):
        self.roi_panel.add_roi(roi); self._set_nc_btn(self.viewer._next_color)

    def _on_roi_deleted(self, roi):
        self.viewer.remove_roi(roi); self.fit_panel.remove_roi(roi)

    def _on_roi_label_changed(self, roi, label):
        self.fit_panel.update_label(roi, label)

    def _on_fit_requested(self, roi):
        xs, ys = self.viewer._ct_roi_to_xad_pts(roi, self.viewer._sl)
        if len(xs) < 2:
            QMessageBox.information(self, "Fit", "Not enough points in ROI for a fit.")
            return
        self.fit_panel.update_fit(roi, xs, ys)

    def _on_fit_computed(self, roi, slope, intercept, x_lo, x_hi):
        self.viewer.draw_fit_line(roi, slope, intercept, x_lo, x_hi)

    def _on_clear(self): self.viewer.clear_rois(); self.roi_panel.clear(); self.fit_panel.clear()

    def _on_show_mat(self, mat):
        if self.viewer._histo.qimg is None:
            QMessageBox.information(self,"No XAD data",
                "Please compute XAD first, then select materials.")
            self.mat_tab.uncheck(mat.name); return
        self.viewer.show_material(mat)

    def _on_hide_mat(self, name): self.viewer.hide_material(name)

    def _on_show_mixture(self, name, m1_name, m2_name):
        if self.viewer._histo.qimg is None:
            QMessageBox.information(self,"No XAD data","Please compute XAD first."); return
        mats = self.mat_tab.materials
        m1 = mats.get(m1_name); m2 = mats.get(m2_name)
        if m1 is None or m2 is None:
            QMessageBox.information(self,"Missing materials",
                f"Could not find: {m1_name}, {m2_name}\nLoad them in the Materials tab first."); return
        color = "#6ec6ff" if "Iodine" in name else "#ffd166"
        self.viewer.show_mixture_line(name, m1, m2, color=color)

    def _on_hide_mixture(self, name): self.viewer.hide_mixture_line(name)

    def _save(self):
        path,_=QFileDialog.getSaveFileName(self,"Save figure","","PNG (*.png);;TIFF (*.tiff);;BMP (*.bmp)")
        if not path: return
        self.viewer.save_figure(path); self.lbl_status.setText(f"Saved: {os.path.basename(path)}")

    def _on_roles_changed(self):
        hi,lo=self.data_tab.get_hi_lo()
        self.lbl_status.setText(
            f"Ready — High: {hi.name}  Low: {lo.name}" if hi and lo
            else "Assign High/Low roles in data tab.")

    def _on_compute(self):
        hi,lo=self.data_tab.get_hi_lo()
        if not hi or not lo:
            QMessageBox.information(self,"Roles not set","Tag one folder as high and one as low."); return
        kh=self._kev(hi); kl=self._kev(lo); kr=self.spin_ref.value()
        self.btn_compute.setEnabled(False); self.lbl_status.setText("Computing XAD…")
        self._worker=ComputeWorker(hi.load(),lo.load(),kh,kl,kr)
        self._worker.done.connect(lambda *a,kh=kh,kl=kl,kr=kr: self._on_done(*a,kh=kh,kl=kl,kr=kr))
        self._worker.error.connect(self._on_error); self._worker.start()

    @staticmethod
    def _kev(e):
        m=re.search(r"(\d+)",os.path.basename(e.folder)); return int(m.group(1)) if m else 100

    def _on_done(self, pe, cs, lo, z_locs, dv_lo, kh, kl, kr):
        self.btn_compute.setEnabled(True); H,W,N=lo.shape
        self.lbl_status.setText(f"Ready — {N} slices  {H}x{W} px")
        for w in (self.slider,self.spin_sl):
            w.blockSignals(True); w.setMaximum(N-1); w.setValue(0); w.blockSignals(False)
        dv_ct=self.data_tab._dv_ct
        self.viewer.load_data(pe,cs,lo,z_locs,dv_lo,dv_ct,kh=kh,kl=kl,kr=kr)
        self.roi_panel.clear()
        self.mat_tab.recheck_all_visible()
        self.mix_tab.recheck_all_visible()

    def _on_error(self, msg):
        self.btn_compute.setEnabled(True); self.lbl_status.setText("Error.")
        QMessageBox.critical(self,"Error",msg)

    def _on_slider(self, v):
        self.spin_sl.blockSignals(True); self.spin_sl.setValue(v); self.spin_sl.blockSignals(False)
        self._st.start()

    def _on_spin(self, v):
        self.slider.blockSignals(True); self.slider.setValue(v); self.slider.blockSignals(False)
        self._st.start()

    def _apply_slice(self): self.viewer.set_slice(self.slider.value())


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling,True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,True)
    app=QApplication(sys.argv); app.setStyle("Windows")
    app.setPalette(_make_dark_palette())
    app.setStyleSheet("""
        QMainWindow, QWidget { font-size: 13px; }
        QPushButton { padding:3px 7px; background:#3c3c3c; border:1px solid #555; border-radius:3px; color:#ddd; }
        QPushButton:hover { background:#505050; }
        QPushButton:pressed { background:#282828; }
        QPushButton:checked { background:#2a82da; border:1px solid #1a6ab8; color:#fff; }
        QPushButton:disabled { background:#2a2a2a; color:#666; border:1px solid #3a3a3a; }
        QComboBox, QSpinBox, QDoubleSpinBox { min-height:22px; background:#2d2d2d;
            border:1px solid #555; border-radius:3px; color:#ddd; }
        QComboBox::drop-down { border:none; }
        QComboBox QAbstractItemView { background:#2d2d2d; color:#ddd; selection-background-color:#2a82da; }
        QTabBar::tab { padding:5px 12px; background:#333; color:#aaa; border:1px solid #444; border-bottom:none; }
        QTabBar::tab:selected { background:#1e1e1e; color:#eee; }
        QTabBar::tab:hover { background:#3d3d3d; }
        QTableWidget { gridline-color:#3a3a3a; }
        QHeaderView::section { background:#3a3a3a; color:#ccc; border:1px solid #555; padding:3px; }
        QListWidget { background:#2d2d2d; border:1px solid #444; }
        QListWidget::item { padding:3px 4px; color:#ddd; }
        QListWidget::item:selected { background:#2a82da; }
        QListWidget::item:hover { background:#3a3a3a; }
        QScrollBar:vertical { background:#2d2d2d; width:10px; }
        QScrollBar::handle:vertical { background:#555; border-radius:4px; }
        QScrollBar:horizontal { background:#2d2d2d; height:10px; }
        QScrollBar::handle:horizontal { background:#555; border-radius:4px; }
        QSlider::groove:horizontal { height:4px; background:#444; border-radius:2px; }
        QSlider::handle:horizontal { background:#2a82da; width:14px; height:14px; margin:-5px 0; border-radius:7px; }
        QGroupBox { color:#aaa; border:1px solid #444; border-radius:4px; margin-top:6px; }
        QGroupBox::title { subcontrol-origin:margin; subcontrol-position:top left; padding:0 4px; left:8px; }
        QRadioButton { spacing:4px; color:#ddd; }
        QCheckBox { spacing:4px; color:#ddd; }
        QLabel { color:#ccc; }
        QToolTip { background:#3c3c3c; color:#eee; border:1px solid #666; }
        QMenuBar { background:#2d2d2d; color:#ccc; }
        QMenuBar::item:selected { background:#2a82da; }
        QMenu { background:#2d2d2d; color:#ccc; border:1px solid #555; }
        QMenu::item:selected { background:#2a82da; }
        QMessageBox { background:#2d2d2d; }
        QFrame[frameShape="5"] { color:#555; }
    """)
    win=MainWindow(); win.show(); sys.exit(app.exec_())


if __name__ == "__main__":
    main()