"""
Demonstration of XAD technique for Spectral CT Images to accompany publication
"Material Analysis with Spectral Computed Tomography: X-Ray Attenuation Decomposition Imaging"
Arnold Schilham, July 2024, UMC Utrecht, the Netherlands

This file defines the helper functions for different models to calculate the energy scalings of
the x-ray attenuation components.
  
Changelog:
  20240710: version to accompany publication
"""
import numpy as np  # access to fast math

# Read attenuation vs energy for interesting stuff; original data from NIST
NIST = {}
density = {}
materials = ['Water', 'PMMA', 'Teflon']
for mat in materials:
    with open('data/NIST/{}.tsv'.format(mat), 'r') as fin:
        data = fin.read().splitlines() # read whole file and split on '\n'
    # material water
    # density 1.0
    # keV pe/rho ra/rho cs/rho
    # 40 .....
    material = data[0].split('\t')[1]
    NIST[material] = {}
    density[material] = float( data[1].split('\t')[1] )
    #hdrs = data[1]
    for line in data[3:]:
        vals = line.split('\t')
        NIST[material][int(vals[0])] = {'ra': float(vals[1]), 'cs': float(vals[2]),'pe': float(vals[3])}
        

# define some helpers for later
def dPE(material, keV):
    # return the photo-electric part of the mass attenuation of given material at given energy
    return density[material]*NIST[material][keV]['pe']

def dRA(material, keV):
    # return the rayleigh scatter part of the mass attenuation of given material at given energy
    return density[material]*NIST[material][keV]['ra']

def dCS(material, keV):
    # return the compton scatter part of the mass attenuation of given material at given energy
    return density[material]*NIST[material][keV]['cs']

def MU(material, keV):
    # return the total mass attenuation of given material at given energy
    stuff = NIST[material][keV]
    
    return density[material]*(stuff['pe']+stuff['ra']+stuff['cs'])

def kleinNishina(keV):
    # Calculate the Klein-Nishina coefficient for given energy in keV

    a = keV/510.998928 # alpha, rest-mass electron
    
    g = 1.+2.*a
    f1 = (1.+a)/(a*a)
    f2 = 2.*(1.+a)/g
    f3 = 1./a*np.log(g)
    f4 = (1.+3.*a)/(g*g)
        
    return f1*( f2-f3 )+f3/2.-f4

