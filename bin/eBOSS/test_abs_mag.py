import time
t0 = time.time()
from scipy.interpolate import interp1d
from scipy.integrate import quad
import numpy as np

import magnitude_library
aa = magnitude_library.Photo()
print aa.normDict

import astropy.cosmology as co
import astropy.units as uu

bb = co.Planck13

def get_abs_mags(redshift = 0.1):
	distMod = bb.distmod( redshift ).value #.to(uu.cm).value)**2.
	wl = np.arange(3600,10000,2)/(1+redshift)
	fl = 16 * 10**(-17) * np.ones_like(wl) # erg/cm2/s/A
	#fl = fl_obs * dl2 # erg/s/A
	spec = magnitude_library.interp1d(wl, fl)
	mags, arr = aa.computeMagnitudes(spec, distMod) 
	print redshift, distMod, mags#,  mags - distMod 
	return mags, distMod, time.time()-t0

outs = np.array([get_mags(z) for z in np.arange(0.01, 2.2, 0.1)])

import sys
sys.exit()

#b={
#"u": 1.4 * 10**(-10), 
#"g": 0.9 * 10**(-10), 
#"r": 1.2 * 10**(-10), 
#"i": 1.8 * 10**(-10), 
#"z": 7.4 * 10**(-10)} 

#mf00={
#"u": 24.63,
#"g": 25.11,
#"r": 24.80,
#"i": 24.36,
#"z": 22.83}

#mf010b={
#"u": 22.12,
#"g": 22.60,
#"r": 22.29,
#"i": 21.85,
#"z": 20.32}

#np.arcsinh()

#mag = -(2.5/ln(10))*[asinh((f/f0)/2b)+ln(b)]
#error(mag) = 2.5 / ln(10) * error(counts)/exptime * 1/2b * 


#mag = -(2.5/ln(10))*[asinh((f/f0)/2b)+ln(b)]
#print outs