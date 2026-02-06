#!/usr/bin/env python
"""
Arnold Schilham, July 2024, UMC Utrecht, the Netherlands

Demonstration of XAD technique for Spectral CT Images to accompany publication
"Material Analysis with Spectral Computed Tomography: X-Ray Attenuation Decomposition Imaging"
by A Schilham, R W van Hamersvelt, P A de Jong, and T Leiner

This example reads in a high keV and a low keV Virtual Mono-energetic dataset.
That data will be used to calculate in each pixel the pe+ra = Photo-Electric+Rayleigh Scatter 
component and the cs = Compton Scatter component of the x-ray attenuation in HUpe and HUcs,
respectively. With those components, an XAD graph will be constructed.

  1. Read a high and low keV images and extract the HU data
  2. Calculate the attenuation components pe+ra and cs in HUpe and HUcs
  3. Construct the XAD graph
  
Changelog:
  20250219: updated for pydicom 3.0.1
  20240710: version to accompany publication
"""
import pydicom as dcm   # read DICOM images
import numpy as np    # access to fast math
import xad_models as XM  # previously derived model
import matplotlib.pyplot as plt # make graphs
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os.path

def read_HUhilo(root, keVhi, keVlo):
    """
    Read a high and low keV CT data set and extract HU values
    Assume image name = root/keV.dcm
    """
    
    def _get_hu(keV):
        # first read the high and low energy data sets
        hu = dcm.dcmread(os.path.join(root, "{}.dcm".format(keV)))
        
        # read HU scale correction
        slope, intercept = hu.RescaleSlope, hu.RescaleIntercept
        
        # drop everything, just keep the pixeldata
        hu = hu.pixel_array.astype(float)
        
        # apply HU scale correction
        hu = intercept+ slope*hu
        return hu
    
    return _get_hu(keVhi), _get_hu(keVlo)


def show_images(hulolab, huhilab):
    """
    Show two images next to each other with a colorbar
    """
    # define a shared window/level for both images
    win,lev = 200,0
    vmin = lev-win/2
    vmax = lev+win/2

    # define 3 subfigures
    fig = plt.figure()
    dx = 0.01
    dy = 0.01
    bx = 0.15
    hy = .232
    by = .550
    axes = [fig.add_axes([       0, dy, (1-bx)/2-dx, 1-dy]), 
            fig.add_axes([(1-bx)/2, dy, (1-bx)/2-dx, 1-dy]),
            fig.add_axes([  (1-bx), hy,      (bx/5),   by])]  # Left, bottom, width, height

    # plot HU of lo energy
    axes[0].imshow(hulolab[0], cmap='gray', vmin=vmin, vmax=vmax); # plot grayscale
    axes[0].set_title(hulolab[1])
    axes[0].axis('off')
    
    # plot HU of hi energy
    ims=axes[1].imshow(huhilab[0], cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title(huhilab[1])
    axes[1].axis('off')
    
    # plot the colorbar
    cbar = plt.colorbar(ims, cax=axes[2])
    cbar.set_label(r'HU')

def get_HUpecs(keVHUhi, keVHUlo, keVref, scaling):
    """
    scaling: either a number to indicate the power coefficient scaling of the PE with energy,
             or a string to indicate the material to use for energy scaling of that material
    keVref: any reference keV value, typically 65-75keV 
    
    Calculate PE and SC components:
      solve mu(E) = a_pe*f_pe(E)+a_cs*f_cs(E) = mu_w(E)*( HU(E)/ 1000 + 1 )
    
    We do not want to be troubled with the exact multiplication constant for each material, so 
    we are going to calculate a_pe*f_pe(E_ref) and a_cs*f_cs(E_ref) with E_ref a reference energy
    
    Used model:
      f_pe ~ E^(-3)
      f_cs = Klein-Nishina
      and here we use pe = pe+ra
    
    mu(Elo) = f_pe(Elo)*pe(Elo)+f_cs(Elo)*cs(Elo)
    mu(Ehi) = f_pe(Ehi)*pe(Ehi)+f_cs(Ehi)*cs(Ehi)
    
    Substitute 
      g_pe(E,Eref) = f_pe(E)/f_pe(Eref)
      g_cs(E,Eref) = f_cs(E)/f_cs(Eref)
      
    mu(Elo) = g_pe(Elo,Eref)*pe(Eref)+g_cs(Elo,Eref)*cs(Eref)
    mu(Ehi) = g_pe(Ehi,Eref)*pe(Eref)+g_cs(Ehi,Eref)*cs(Eref)

    Solve cs(Eref), pe(Eref) from these two equations:
     Eref you choose.
     Elo and Ehi are given.
     mu(Elo) and mu(Ehi) are (calculated directly from) the low and high monoE CT images.
     f_pe(E) and f_cs(E) are the energy functions of the Alvarez-Macovsci model or 
       generic material scaling.
     Only unknowns are cs(Eref) and pe(Eref).

    Note1: For the AM model, the correct power for f_pe for a specific scanner setup can 
      be obtained by analysing a series of virtual monoE reconstructions for a range of 
      keV values.

    Note2: An alternative approach is to use published data for the energy dependence of a fixed 
      material for the f_pe and f_cs (generic material scaling). For some brands of CT scanners 
      (Siemens and GE), the Alvarez-Macovsci scaling works well, while for others brands (Philips) 
      a scaling with water seems a better match.
    """
    keVhi,HUhi = keVHUhi
    keVlo,HUlo = keVHUlo

    # calculate mu values for hi and lo
    muhi = XM.MU('Water', keVhi)*(HUhi/1000. +1.)
    mulo = XM.MU('Water', keVlo)*(HUlo/1000. +1.)

    # calculate relative functions for pe and cs
    if isinstance(scaling, (float, int)):
        # use Alvarez-Macovsci like model
        g_pe = [np.power(1.*k/keVref, scaling) for k in [keVhi, keVlo] ]
        g_cs = [XM.kleinNishina(k)/XM.kleinNishina(keVref) for k in [keVhi, keVlo] ]
    elif isinstance(scaling, str):
        # use energy scaling from a fixed material
        g_pe = [(XM.dPE(scaling,k)+XM.dRA(scaling,k))/(XM.dPE(scaling,keVref)+XM.dRA(scaling,keVref)) for k in [keVhi, keVlo] ]
        g_cs = [XM.dCS(scaling,k)/XM.dCS(scaling,keVref) for k in [keVhi, keVlo] ]
        
    # solve linear equations to calculate the cs and pe attenuation components (at keVref)
    cs = (muhi-mulo*g_pe[0]/g_pe[1])/g_cs[0]/(1.-g_cs[1]/g_cs[0]*g_pe[0]/g_pe[1])
    pe = (mulo-cs*g_cs[1])/g_pe[1]
    
    # turn cs and pe into HUpe and HUcs
    HUcs = 1000*(cs-XM.dCS('Water', keVref))/XM.MU('Water', keVref)
    HUpe = 1000*(pe-(XM.dPE('Water', keVref)+XM.dRA('Water', keVref)))/XM.MU('Water', keVref)

    return HUpe,HUcs
    

def XAD(HUpe, HUcs):
    """
    Make a 2D histogram of the (HUpe,HUcs) values.
    """
    # plot in a 2D histogram
    xedges = np.arange(max(-1000., HUcs.min()), max(850, HUcs.max()), HUcs.max()/400.) 
    yedges = np.arange(max(-1000., HUpe.min()), HUpe.max(), HUpe.max()/400.) 
    
    histo, xedges, yedges = np.histogram2d(HUcs.flatten(), HUpe.flatten(), bins=(xedges, yedges)) # variable binsize
    
    histo = histo.T  # Let each row list bins with common y range.
    histo = np.log10(1.+histo) # log axis
    X, Y = np.meshgrid(xedges, yedges) # pcolormesh can display actual edges
    
    vmin,vmax = np.min(histo), np.max(histo)
    
    fig, ax = plt.subplots()
    ims = ax.pcolormesh(X, Y, histo, vmin=vmin, vmax=.66*vmax, cmap='gray')
    ax.set_xlabel("HUcs")
    ax.set_ylabel("HUpe")
    ax.set_xlim(xedges[0], xedges[-1])
    ax.set_ylim(yedges[0], yedges[-1])

    # create an axes on the right side of ax for a colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(ims, cax=cax)
    cbar.set_label(r'$\log_{10}$(count+1)')
    plt.tight_layout()
    return ax

def add_materials(ax, keVref):
    """
    plot the known materials in the XAD plot
    """
    mats = ['Water', 'PMMA', 'Teflon']
    
    pe_w = XM.dPE('Water',keVref)+XM.dRA('Water',keVref)
    cs_w = XM.dCS('Water',keVref)
    mu_w = pe_w+cs_w
    for mat in mats:
        pe = XM.dPE(mat,keVref)+XM.dRA(mat,keVref)
        cs = XM.dCS(mat,keVref)
        
        HUpe = 1000*(pe-pe_w)/mu_w
        HUcs = 1000*(cs-cs_w)/mu_w
        
        ax.plot(HUcs,HUpe, 'o', markerfacecolor='none', markeredgewidth=2, label=mat)
        
    ax.legend(loc='upper left')
    
def main():
    """
    """
    # 1. Read a high and low keV images and extract the HU data
    root = "images/CT7500" # Philips images
    keVhi, keVlo = 160, 60 # input energies

    HUhi, HUlo = read_HUhilo(root, keVhi, keVlo) # read CT image of high and low kV and extract HU values
    show_images((HUhi, 'HUhi'), (HUlo,'HUlo') )  # Show the low and high kVp images
    
    # 2. Calculate the attenuation components pe+ra and cs in HUpe and HUcs
    keVref = 80 # reference energy, take any value
    #scaling = -2.64 # use Alvarez Macovsci scaling for Siemens
    scaling = "Water" # use material scaling for Philips
    
    HUpe, HUcs = get_HUpecs((keVhi, HUhi), (keVlo, HUlo), keVref, scaling) # Calculate the attenuation components in HUpe and HUcs
    show_images((HUpe,'HUpe'), (HUcs, 'HUcs'))  # Show the attenuation component images

    ax = XAD(HUpe,HUcs) # make XAD graph
    
    add_materials(ax, keVref)

    plt.show()
    print("Done.")
        
if __name__ == "__main__":
    main()
