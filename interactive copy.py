import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from matplotlib.patches import Polygon as MplPolygon
from mpl_toolkits.axes_grid1 import make_axes_locatable 
from itertools import cycle
import xad_models as XM
import pydicom as dcm
import os
from scipy.ndimage import gaussian_filter

""" de berekeningen zijn een copy paste van https://github.com/asch99/XAD/tree/main, het gros van het script doet de mapping vanuit XAD -> CT en vice versa, 
rayleigh en photoelectrisch effect zijn samen gecombineerd in de g_pe parameter """

class XADInteractive:
    def __init__(self, HUpe, HUcs, HUlo, keVref):
        # voor de polygons moet dit
        self.HUpe_flat = HUpe.flatten()
        self.HUcs_flat = HUcs.flatten()
        self.HUlo = HUlo
        self.shape = HUlo.shape
        
        plt.style.use('dark_background')
        self.fig, (self.ax_img, self.ax_xad) = plt.subplots(1, 2, figsize=(16, 8))


        histo, xedges, yedges = np.histogram2d(
            self.HUcs_flat, self.HUpe_flat, bins=400, range=[[1.1*self.HUcs_flat.min(), 1.1*self.HUcs_flat.max()], [1.1*self.HUpe_flat.min(), 1.1*self.HUpe_flat.max()]] # wat padding aan de zijkant
        )
        
        # gauss filteren voor betere visualisatie
        histo = gaussian_filter(histo, sigma=1.2)
        histo = np.log10(1 + histo).T 
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

        self.ax_xad.set_facecolor('white')
        self.xad_bg = self.ax_xad.pcolormesh(
            X, Y, histo, cmap='magma', 
            vmin=0, vmax=np.max(histo)*1, zorder=0
        )

        self.ax_xad.set_xlabel("HUcs")
        self.ax_xad.set_ylabel("HUpe")
        self.ax_xad.set_title("XAD", fontsize=14, pad=15)
        divider = make_axes_locatable(self.ax_xad)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        plt.colorbar(self.xad_bg, cax=cax).set_label(r'$\log_{10}$(count+1)') #dichtheid aangeven

        self.ax_img.imshow(self.HUlo, cmap='gray', vmin=-150, vmax=250)
        self.ax_img.set_title("CT Slice", fontsize=14, pad=15)

        pix_x, pix_y = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))
        self.pix_coords = np.vstack((pix_x.flatten(), pix_y.flatten())).T
        self.xad_coords = np.vstack((self.HUcs_flat, self.HUpe_flat)).T

        self.colors = cycle(plt.cm.tab10.colors)
        self.selections = []
        self.line_props = dict(color='red', linestyle='-', linewidth=1, alpha=1)

        self.lasso_img = PolygonSelector(self.ax_img, onselect=self.on_select_img, 
                                         props=self.line_props, useblit=True)
        self.lasso_xad = PolygonSelector(self.ax_xad, onselect=self.on_select_xad, 
                                         props=self.line_props, useblit=True)
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.tight_layout()

    def _get_mask_indices(self, path, coords):
        extents = path.get_extents()
        in_bbox = (coords[:, 0] >= extents.x0) & (coords[:, 0] <= extents.x1) & \
                  (coords[:, 1] >= extents.y0) & (coords[:, 1] <= extents.y1)
        
        indices = np.where(in_bbox)[0]
        if len(indices) == 0: return indices
        return indices[path.contains_points(coords[indices])]

    def on_select_img(self, verts):
        if len(verts) < 3: return
        color = next(self.colors)
        path = Path(verts)
        ind = self._get_mask_indices(path, self.pix_coords)

        poly = MplPolygon(verts, facecolor='none', edgecolor=color, linewidth=2)
        self.ax_img.add_patch(poly)
        highlight = self.ax_xad.scatter(self.HUcs_flat[ind], self.HUpe_flat[ind], s=5, color=color, alpha=0.6)
        
        self.selections.append({'poly': poly, 'scat': highlight})

        self.lasso_img.disconnect_events()
        self.lasso_img = PolygonSelector(self.ax_img, onselect=self.on_select_img, 
                                         props=self.line_props, useblit=True)
        self.fig.canvas.draw_idle()

    def on_select_xad(self, verts):
        if len(verts) < 3: return
        color = next(self.colors)
        path = Path(verts)
        ind = self._get_mask_indices(path, self.xad_coords)

        poly = MplPolygon(verts, facecolor='none', edgecolor=color, linewidth=2)
        self.ax_xad.add_patch(poly)
        highlight = self.ax_img.scatter(self.pix_coords[ind, 0], self.pix_coords[ind, 1], 
                                        s=2, color=color, alpha=0.6)
        
        self.selections.append({'poly': poly, 'scat': highlight})

        self.lasso_xad.disconnect_events()
        self.lasso_xad = PolygonSelector(self.ax_xad, onselect=self.on_select_xad, 
                                         props=self.line_props, useblit=True)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == 'c':
            for item in self.selections:
                item['poly'].remove()
                item['scat'].remove()
            self.selections = []
            self.fig.canvas.draw_idle()


def get_HUpecs(keVHUhi, keVHUlo, keVref, scaling):
    khi, hhi = keVHUhi
    klo, hlo = keVHUlo
    muhi = XM.MU('Water', khi) * (hhi/1000. + 1.)
    mulo = XM.MU('Water', klo) * (hlo/1000. + 1.)

    if isinstance(scaling, (float, int)): # als 'scaling' een integer :
        g_pe = [np.power(1.*k/keVref, scaling) for k in [khi, klo]]
        g_cs = [XM.kleinNishina(k)/XM.kleinNishina(keVref) for k in [khi, klo]]
    else: # als 'scaling' geen integer :
        g_pe = [(XM.dPE(scaling,k)+XM.dRA(scaling,k))/(XM.dPE(scaling,keVref)+XM.dRA(scaling,keVref)) for k in [khi, klo]]
        g_cs = [XM.dCS(scaling,k)/XM.dCS(scaling,keVref) for k in [khi, klo]]
        
    cs = (muhi - mulo * g_pe[0]/g_pe[1]) / g_cs[0] / (1. - g_cs[1]/g_cs[0] * g_pe[0]/g_pe[1])
    pe = (mulo - cs * g_cs[1]) / g_pe[1]
    
    HUcs = 1000 * (cs - XM.dCS('Water', keVref)) / XM.MU('Water', keVref)
    HUpe = 1000 * (pe - (XM.dPE('Water', keVref) + XM.dRA('Water', keVref))) / XM.MU('Water', keVref)
    return HUpe, HUcs

def read_HUhilo(root, keVhi, keVlo):
    def _get_hu(keV):
        ds = dcm.dcmread(os.path.join(root, f"{keV}.dcm"))
        return ds.RescaleIntercept + ds.RescaleSlope * ds.pixel_array.astype(float)
    return _get_hu(keVhi), _get_hu(keVlo)

def main():
    # Setup parameters
    root = "images/CT7500" # mapje naar de beelden 
    keVhi, keVlo = 160, 60 # dit moet je veranderen als de namen van de beelden anders zijn dan {energie in kev}.dcm" 
    keVref = 80 # werd aangegeven deze rond de rond de 80 te houden, nog even uitzoeken hoe groot de invloed hiervan is
    scaling = "Water" # voor philips scanners was dit het beste 

    try:
        HUhi, HUlo = read_HUhilo(root, keVhi, keVlo)
        HUpe, HUcs = get_HUpecs((keVhi, HUhi), (keVlo, HUlo), keVref, scaling)
        

        app = XADInteractive(HUpe, HUcs, HUlo, keVref) # die derde parameter geeft aan welk beeld je wil zien
        plt.show()
    except Exception as e:
        print(f"helaas: {e}. bestandje niet gevonnden {root}")

if __name__ == "__main__":
    main()