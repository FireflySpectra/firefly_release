import numpy as np
import warnings
import math
import os
import scipy.interpolate as interpolate
from astropy.io import fits
import time

from firefly_fitter import fitter, sigmaclip
from firefly_library import normalise_spec 
#from firefly_instrument import *

# Calzetti curves, and other general attenuation curves are computed
# here, along with (in dust_calzetti) applying to spectra directly.
def find_nearest(array,value,output):
	idx = (np.abs(np.array(array)-np.array(value))).argmin()
	return output[idx]

def curve_smoother(x, y, smoothing_length):
	"""
	Smoothes a curve y = f(x) with a running median over a given smoothing length.

	Returns the smoothed array.
	
	Used internally in function determine_attenuation

	:param x: x
	:param y: y
	:param smoothing_length: smoothing length in the same unit than x.
	"""
	y_out = np.zeros(len(y))
	for w in range(len(x)):
		check_index = (x < x[w]+smoothing_length)&(x>x[w]-smoothing_length)
		y_out[w] = np.median(y[check_index])
	return y_out

def reddening_ccm(wave, ebv=None, a_v=None, r_v=3.1, model='ccm89'):
    """
	Not used in FIREFLY
	Determines a CCM reddening curve.

    Parameters
    ----------
    wave: ~numpy.ndarray
        wavelength in Angstroms
    flux: ~numpy.ndarray
    ebv: float
        E(B-V) differential extinction; specify either this or a_v.
    a_v: float
        A(V) extinction; specify either this or ebv.
    r_v: float, optional
        defaults to standard Milky Way average of 3.1
    model: {'ccm89', 'gcc09'}, optional
        * 'ccm89' is the default Cardelli, Clayton, & Mathis (1989) [1]_, but
          does include the O'Donnell (1994) parameters to match IDL astrolib.
        * 'gcc09' is Gordon, Cartledge, & Clayton (2009) [2]_. This paper has
          incorrect parameters for the 2175A bump; not yet corrected here.

    Returns
    -------
    reddening_curve: ~numpy.ndarray
        Multiply to deredden flux, divide to redden.

    Notes
    -----
    Cardelli, Clayton, & Mathis (1989) [1]_ parameterization is used for all
    models. The default parameter values are from CCM except in the optical
    range, where the updated parameters of O'Donnell (1994) [3]_ are used
    (matching the Goddard IDL astrolib routine CCM_UNRED).

    The function is works between 910 A and 3.3 microns, although note the
    default ccm89 model is scientifically valid only at >1250 A.

    Model gcc09 uses the updated UV coefficients of Gordon, Cartledge, & Clayton
    (2009) [2]_, and is valid from 910 A to 3030 A. This function will use CCM89
    at longer wavelengths if GCC09 is selected, but note that the two do not
    connect perfectly smoothly. There is a small discontinuity at 3030 A. Note
    that GCC09 equations 14 and 15 apply to all x>5.9 (the GCC09 paper
    mistakenly states they do not apply at x>8; K. Gordon, priv. comm.).

    References
    ----------
    [1] Cardelli, J. A., Clayton, G. C., & Mathis, J. S. 1989, ApJ, 345, 245
    [2] Gordon, K. D., Cartledge, S., & Clayton, G. C. 2009, ApJ, 705, 1320
    [3] O'Donnell, J. E. 1994, ApJ, 422, 158O

    """

    from scipy.interpolate import interp1d

    model = model.lower()
    if model not in ['ccm89','gcc09']:
        raise ValueError('model must be ccm89 or gcc09')
    if (a_v is None) and (ebv is None):
        raise ValueError('Must specify either a_v or ebv')
    if (a_v is not None) and (ebv is not None):
        raise ValueError('Cannot specify both a_v and ebv')
    if a_v is not None:
        ebv = a_v / r_v

    if model == 'gcc09':
        raise ValueError('TEMPORARY: gcc09 currently does 2175A bump '+
            'incorrectly')

    x = 1e4 / wave      # inverse microns

    if any(x < 0.3) or any(x > 11):
        raise ValueError('ccm_dered valid only for wavelengths from 910 A to '+
            '3.3 microns')
    if any(x > 8) and (model == 'ccm89'):
        warnings.warn('CCM89 should not be used below 1250 A.')
#    if any(x < 3.3) and any(x > 3.3) and (model == 'gcc09'):
#        warnings.warn('GCC09 has a discontinuity at 3030 A.')

    a = np.zeros(x.size)
    b = np.zeros(x.size)

    # NIR
    valid = (0.3 <= x) & (x < 1.1)
    a[valid] = 0.574 * x[valid]**1.61
    b[valid] = -0.527 * x[valid]**1.61

    # optical, using O'Donnell (1994) values
    valid = (1.1 <= x) & (x < 3.3)
    y = x[valid] - 1.82
    coef_a = np.array([-0.505, 1.647, -0.827, -1.718, 1.137, 0.701, -0.609,
        0.104, 1.])
    coef_b = np.array([3.347, -10.805, 5.491, 11.102, -7.985, -3.989, 2.908,
        1.952, 0.])
    a[valid] = np.polyval(coef_a,y)
    b[valid] = np.polyval(coef_b,y)

    # UV
    valid = (3.3 <= x) & (x < 8)
    y = x[valid]
    f_a = np.zeros(y.size)
    f_b = np.zeros(y.size)
    select = (y >= 5.9)
    yselect = y[select] - 5.9

    f_a[select] = -0.04473 * yselect**2 - 0.009779 * yselect**3
    f_b[select] = 0.2130 * yselect**2 + 0.1207 * yselect**3
    a[valid] = 1.752 - 0.316*y - (0.104 / ((y-4.67)**2 + 0.341)) + f_a
    b[valid] = -3.090 + 1.825*y + (1.206 / ((y-4.62)**2 + 0.263)) + f_b

    # far-UV CCM89 extrapolation
    valid = (8 <= x) & (x < 11)
    y = x[valid] - 8.
    coef_a = np.array([-0.070, 0.137, -0.628, -1.073])
    coef_b = np.array([0.374, -0.420, 4.257, 13.670])
    a[valid] = np.polyval(coef_a,y)
    b[valid] = np.polyval(coef_b,y)

    # Overwrite UV with GCC09 model if applicable. Not an extrapolation.
    if model == 'gcc09':
        valid = (3.3 <= x) & (x < 11)
        y = x[valid]
        f_a = np.zeros(y.size)
        f_b = np.zeros(y.size)
        select = (5.9 <= y)
        yselect = y[select] - 5.9
        f_a[select] = -0.110 * yselect**2 - 0.0099 * yselect**3
        f_b[select] = 0.537 * yselect**2 + 0.0530 * yselect**3
        a[valid] = 1.896 - 0.372*y - (0.0108 / ((y-4.57)**2 + 0.0422)) + f_a
        b[valid] = -3.503 + 2.057*y + (0.718 / ((y-4.59)**2 + 0.0530*3.1)) + f_b

    a_v = ebv * r_v
    a_lambda = a_v * (a + b/r_v)
    reddening_curve = 10**(0.4 * a_lambda)

    return reddening_curve
#    return a_lambda / a_v  #debug


def reddening_fm(wave, ebv=None, a_v=None, r_v=3.1, model='f99'):
    """Determines a Fitzpatrick & Massa reddening curve.

    Parameters
    ----------
    wave: ~numpy.ndarray
        wavelength in Angstroms
    ebv: float
        E(B-V) differential extinction; specify either this or a_v.
    a_v: float
        A(V) extinction; specify either this or ebv.
    r_v: float, optional
        defaults to standard Milky Way average of 3.1
    model: {'f99', 'fm07'}, optional
        * 'f99' is the default Fitzpatrick (1999) [1]_
        * 'fm07' is Fitzpatrick & Massa (2007) [2]_. Currently not R dependent.

    Returns
    -------
    reddening_curve: ~numpy.ndarray
        Multiply to deredden flux, divide to redden.

    Notes
    -----
    Uses Fitzpatrick (1999) [1]_ by default, which relies on the UV
    parametrization of Fitzpatrick & Massa (1990) [2]_ and spline fitting in the
    optical and IR. This function is defined from 910 A to 6 microns, but note
    the claimed validity goes down only to 1150 A. The optical spline points are
    not taken from F99 Table 4, but rather updated versions from E. Fitzpatrick
    (this matches the Goddard IDL astrolib routine FM_UNRED).

    The fm07 model uses the Fitzpatrick & Massa (2007) [3]_ parametrization,
    which has a slightly different functional form. That paper claims it
    preferable, although it is unclear if signficantly (Gordon et al. 2009)
    [4]_. It is not the literature standard, so not default here.

    References
    ----------
    [1] Fitzpatrick, E. L. 1999, PASP, 111, 63
    [2] Fitpatrick, E. L. & Massa, D. 1990, ApJS, 72, 163
    [3] Fitpatrick, E. L. & Massa, D. 2007, ApJ, 663, 320
    [4] Gordon, K. D., Cartledge, S., & Clayton, G. C. 2009, ApJ, 705, 1320

    """

    from scipy.interpolate import interp1d

    model = model.lower()
    if model not in ['f99','fm07']:
        raise ValueError('model must be f99 or fm07')
    if (a_v is None) and (ebv is None):
        raise ValueError('Must specify either a_v or ebv')
    if (a_v is not None) and (ebv is not None):
        raise ValueError('Cannot specify both a_v and ebv')
    if a_v is not None:
        ebv = a_v / r_v

    if model == 'fm07':
        raise ValueError('TEMPORARY: fm07 currently not properly R dependent')

    x = 1e4 / wave      # inverse microns
    k = np.zeros(x.size)

    if any(x < 0.167) or any(x > 11):
        raise ValueError('fm_dered valid only for wavelengths from 910 A to '+
            '6 microns')

    # UV region
    uvsplit = 10000. / 2700.  # Turn 2700A split into inverse microns.
    uv_region = (x >= uvsplit)
    y = x[uv_region]
    k_uv = np.zeros(y.size)

    # Fitzpatrick (1999) model
    if model == 'f99':
        x0, gamma = 4.596, 0.99
        c3, c4 = 3.23, 0.41
        c2 = -0.824 + 4.717 / r_v
        c1 = 2.030 - 3.007 * c2
        D = y**2 / ((y**2-x0**2)**2 + y**2 * gamma**2)
        F = np.zeros(y.size)
        valid = (y >= 5.9)
        F[valid] = 0.5392 * (y[valid]-5.9)**2 + 0.05644 * (y[valid]-5.9)**3
        k_uv = c1 + c2*y + c3*D + c4*F
    # Fitzpatrick & Massa (2007) model
    if model == 'fm07':
        x0, gamma = 4.592, 0.922
        c1, c2, c3, c4, c5 = -0.175, 0.807, 2.991, 0.319, 6.097
        D = y**2 / ((y**2-x0**2)**2 + y**2 * gamma**2)
        valid = (y <= c5)
        k_uv[valid] = c1 + c2*y[valid] + c3*D[valid]
        valid = (y > c5)
        k_uv[valid] = c1 + c2*y[valid] + c3*D[valid] + c4*(y[valid]-c5)**2

    k[uv_region] = k_uv

    # Calculate values for UV spline points to anchor OIR fit
    x_uv_spline = 10000. / np.array([2700., 2600.])
    D = x_uv_spline**2 / ((x_uv_spline**2-x0**2)**2 + x_uv_spline**2 * gamma**2)
    k_uv_spline = c1 + c2*x_uv_spline +c3*D

    # Optical / IR
    OIR_region = (x < uvsplit)
    y = x[OIR_region]
    k_OIR = np.zeros(y.size)

    # Fitzpatrick (1999) model
    if model == 'f99':
        # The OIR anchors are up from IDL astrolib, not F99.
        anchors_extinction = np.array([0, 0.26469*r_v/3.1, 0.82925*r_v/3.1, # IR
            -0.422809 + 1.00270*r_v + 2.13572e-04*r_v**2, # optical
            -5.13540e-02 + 1.00216*r_v - 7.35778e-05*r_v**2,
            0.700127 + 1.00184*r_v - 3.32598e-05*r_v**2,
            (1.19456 + 1.01707*r_v - 5.46959e-03*r_v**2 + 7.97809e-04*r_v**3 +
                -4.45636e-05*r_v**4)])
        anchors_k = np.append(anchors_extinction-r_v, k_uv_spline)
        # Note that interp1d requires that the input abscissa is monotonically
        # _increasing_. This is opposite the usual ordering of a spectrum, but
        # fortunately the _output_ abscissa does not have the same requirement.
        anchors_x = 1e4 / np.array([26500., 12200., 6000., 5470., 4670., 4110.])
        anchors_x = np.append(0., anchors_x)  # For well-behaved spline.
        anchors_x = np.append(anchors_x, x_uv_spline)
        OIR_spline = interp1d(anchors_x, anchors_k, kind='cubic')
        k_OIR = OIR_spline(y)
    # Fitzpatrick & Massa (2007) model
    if model == 'fm07':
        anchors_k_opt = np.array([0., 1.322, 2.055])
        IR_wave = np.array([float('inf'), 4., 2., 1.333, 1.])
        anchors_k_IR = (-0.83 + 0.63*r_v) * IR_wave**-1.84 - r_v
        anchors_k = np.append(anchors_k_IR, anchors_k_opt)
        anchors_k = np.append(anchors_k, k_uv_spline)
        anchors_x = np.array([0., 0.25, 0.50, 0.75, 1.])  # IR
        opt_x = 1e4 / np.array([5530., 4000., 3300.])  # optical
        anchors_x = np.append(anchors_x, opt_x)
        anchors_x = np.append(anchors_x, x_uv_spline)
        OIR_spline = interp1d(anchors_x, anchors_k, kind='cubic')
        k_OIR = OIR_spline(y)

    k[OIR_region] = k_OIR

    reddening_curve = 10**(0.4 * ebv * (k+r_v))

    return reddening_curve
#    return (k+r_v) / r_v # debug

def dust_calzetti_py(ebv,lam):
	"""
	Returns the Calzetti extinction for a given E(B-V) and a wavelength array
	"""
	l = lam/10000.
	k = np.zeros_like(lam)
	s1 = (l >= 0.63) & (l<= 2.2)
	k[s1] = 2.659*(-1.857+1.040/l[s1])+4.05
	s2 = (l < 0.63)
	k[s2] = 2.659*(-2.156+1.509/l[s2]-0.198/l[s2]**2+0.011/l[s2]**3)+4.05
	#s3 = (l > 2.2)
	#k[s1] = 0.0
	#output = []
	#for i in lam:
		#l = i / 10000.0 #converting from angstrom to micrometers
		#if (l >= 0.63 and l<= 2.2):
			#k= 2.659*(-1.857+1.040/l[s1])+4.05
		#if (l < 0.63):
			#k= (2.659*(-2.156+1.509/l-0.198/l**2+0.011/l**3)+4.05)
		#if (l > 2.2):
			#k= 0.0
		#output.append(10**(-0.4 * ebv * k))
	output = 10**(-0.4 * ebv * k)	
	return output


def dust_allen_py(ebv,lam):

	''' Calculates the attenuation for the Milky Way (MW) as found in Allen (1976).'''
	from scipy.interpolate import interp1d
	wave = [1000,1110,1250,1430,1670,2000,2220,2500,2860,3330,3650,4000,4400,5000,5530,6700,9000,10000,20000,100000]
	allen_k = [4.20,3.70,3.30,3.00,2.70,2.80,2.90,2.30,1.97,1.69,1.58,1.45,1.32,1.13,1.00,0.74,0.46,0.38,0.11,0.00]
	allen_k = np.array(allen_k)*3.1

	total = interp1d(wave, allen_k, kind='cubic')
	wavelength_vector = np.arange(1000,10000,100)
	fitted_function = total(wavelength_vector)

	output = []
	for l in range(len(lam)):
		k = find_nearest(wavelength_vector,lam[l],fitted_function)
		output.append(10**(-0.4*ebv*k))
	return output

def dust_prevot_py(ebv,lam):

	''' Calculates the attenuation for the Small Magellanic Cloud (SMC) as found in Prevot (1984).'''
	from scipy.interpolate import interp1d
	wave = [1275,1330,1385,1435,1490,1545,1595,1647,1700,1755,1810,1860,1910,2000,2115,2220,2335,2445,2550,2665,2778,\
	2890,2995,3105,3704,4255,5291,10000]
	prevot_k = [13.54,12.52,11.51,10.80,9.84,9.28,9.06,8.49,8.01,7.71,7.17,6.90,6.76,6.38,5.85,5.30,4.53,4.24,3.91,3.49,\
	3.15,3.00,2.65,2.29,1.81,1.00,0.74,0.00]
	prevot_k = np.array(prevot_k)*2.72

	total = interp1d(wave, prevot_k, kind='linear')
	wavelength_vector = np.arange(1275,10000,100)
	fitted_function = total(wavelength_vector)

	output = []
	for l in range(len(lam)):
		k = find_nearest(wavelength_vector,lam[l],fitted_function)
		output.append(10**(-0.4*ebv*k))
	return output


def get_SFD_dust(long,lat,dustmap='ebv',interpolate=True):
    """
    Gets map values from Schlegel, Finkbeiner, and Davis 1998 extinction maps.
    
    `dustmap` can either be a filename (if '%s' appears in the string, it will be
    replaced with 'ngp' or 'sgp'), or one of:
    
    * 'i100' 
        100-micron map in MJy/Sr
    * 'x'
        X-map, temperature-correction factor
    * 't'
        Temperature map in degrees Kelvin for n=2 emissivity
    * 'ebv'
        E(B-V) in magnitudes
    * 'mask'
        Mask values 
        
    For these forms, the files are assumed to lie in the current directory.
    
    Input coordinates are in degrees of galactic latiude and logitude - they can
    be scalars or arrays.
    
    if `interpolate` is an integer, it can be used to specify the order of the
    interpolating polynomial
    
    .. todo::
        Check mask for SMC/LMC/M31, E(B-V)=0.075 mag for the LMC, 0.037 mag for
        the SMC, and 0.062 for M31. Also auto-download dust maps. Also add
        tests. Also allow for other bands.
    
    """
    from numpy import sin,cos,round,isscalar,array,ndarray,ones_like
    #from pyfits import open
    
    if type(dustmap) is not str:
        raise ValueError('dustmap is not a string')
    dml=dustmap.lower()
    if dml == 'ebv' or dml == 'eb-v' or dml == 'e(b-v)' :
        dustmapfn=os.environ['STELLARPOPMODELS_DIR']+'/data/SFD_dust_4096_%s.fits'
    elif dml == 'i100':
        dustmapfn=os.environ['STELLARPOPMODELS_DIR']+'/data/SFD_i100_4096_%s.fits'
    elif dml == 'x':
        dustmapfn=os.environ['STELLARPOPMODELS_DIR']+'/data/SFD_xmap_%s.fits'
    elif dml == 't':
        dustmapfn=os.environ['STELLARPOPMODELS_DIR']+'/data/SFD_temp_%s.fits'
    elif dml == 'mask':
        dustmapfn=os.environ['STELLARPOPMODELS_DIR']+'/data/SFD_mask_4096_%s.fits'
    else:
        dustmapfn=dustmap
    
    if isscalar(long):
        l=array([long])*math.pi/180
    else:
        l=array(long)*math.pi/180
    if isscalar(lat):
        b=array([lat])*math.pi/180
    else:
        b=array(lat)*math.pi/180
        
    if not len(l)==len(b):
        raise ValueError('input coordinate arrays are of different length')
    
    
    
    if '%s' not in dustmapfn:
        f=fits.open(dustmapfn)
        try:
            mapds=[f[0].data]
        finally:
            f.close()
        assert mapds[-1].shape[0] == mapds[-1].shape[1],'map dimensions not equal - incorrect map file?'
        
        polename=dustmapfn.split('.')[0].split('_')[-1].lower()
        if polename=='ngp':
            n=[1]
            if sum(b > 0) > 0:
                b=b
        elif polename=='sgp':
            n=[-1]
            if sum(b < 0) > 0:
                b=b
        else:
            raise ValueError("couldn't determine South/North from filename - should have 'sgp' or 'ngp in it somewhere")
        masks = [ones_like(b).astype(bool)]
    else: #need to do things seperately for north and south files
        nmask = b >= 0
        smask = ~nmask
        
        masks = [nmask,smask]
        ns = [1,-1]
        
        mapds=[]
        f=fits.open(dustmapfn%'ngp')
        try:
            mapds.append(f[0].data)
        finally:
            f.close()
        assert mapds[-1].shape[0] == mapds[-1].shape[1],'map dimensions not equal - incorrect map file?'
        f=fits.open(dustmapfn%'sgp')
        try:
            mapds.append(f[0].data)
        finally:
            f.close()
        assert mapds[-1].shape[0] == mapds[-1].shape[1],'map dimensions not equal - incorrect map file?'
    
    retvals=[]
    for n,mapd,m in zip(ns,mapds,masks):
        #project from galactic longitude/latitude to lambert pixels (see SFD98)
        npix=mapd.shape[0]
        
        x=npix/2*cos(l[m])*(1-n*sin(b[m]))**0.5+npix/2-0.5
        y=-npix/2*n*sin(l[m])*(1-n*sin(b[m]))**0.5+npix/2-0.5
        #now remap indecies - numpy arrays have y and x convention switched from SFD98 appendix
        x,y=y,x
        
        if interpolate:
            from scipy.ndimage import map_coordinates
            if type(interpolate) is int:
                retvals.append(map_coordinates(mapd,[x,y],order=interpolate))
            else:
                retvals.append(map_coordinates(mapd,[x,y]))
        else:
            x=round(x).astype(int)
            y=round(y).astype(int)
            retvals.append(mapd[x,y])
            
            
    
        
    if isscalar(long) or isscalar(lat):
        for r in retvals:
            if len(r)>0:
                return r[0]
        assert False,'None of the return value arrays were populated - incorrect inputs?'
    else:
        #now recombine the possibly two arrays from above into one that looks like  the original
        retval=ndarray(l.shape)
        for m,val in zip(masks,retvals):
            retval[m] = val
        return retval
        
    
def eq2gal(ra,dec):
	"""
	Convert Equatorial coordinates to Galactic Coordinates in the epch J2000.

	Keywords arguments:
	ra  -- Right Ascension (in radians)
	dec -- Declination (in radians)

	Return a tuple (l, b):
	l -- Galactic longitude (in radians)
	b -- Galactic latitude (in radians)
	"""
	# RA(radians),Dec(radians),distance(kpc) of Galactic center in J2000
	Galactic_Center_Equatorial=(math.radians(266.40510), math.radians(-28.936175), 8.33)

	# RA(radians),Dec(radians) of Galactic Northpole in J2000
	Galactic_Northpole_Equatorial=(math.radians(192.859508), math.radians(27.128336))

	alpha = Galactic_Northpole_Equatorial[0]
	delta = Galactic_Northpole_Equatorial[1]
	la = math.radians(33.0)

	b = math.asin(math.sin(dec) * math.sin(delta) +
					math.cos(dec) * math.cos(delta) * math.cos(ra - alpha))

	l = math.atan2(math.sin(dec) * math.cos(delta) - 
					math.cos(dec) * math.sin(delta) * math.cos(ra - alpha), 
					math.cos(dec) * math.sin(ra - alpha)
					) + la

	l = l if l >= 0 else (l + math.pi * 2.0)

	l = l % (2.0 * math.pi)


	return l*180.0/math.pi, b*180.0/math.pi

def get_dust_radec(ra,dec,dustmap,interpolate=True):
	"""
	Gets the value of dust from MW at ra and dec.
	"""
	#from .coords import equatorial_to_galactic
	l,b = eq2gal(math.radians(ra),math.radians(dec))
	return get_SFD_dust(l,b,dustmap,interpolate)

def unred(wave, ebv, R_V=3.1, LMC2=False, AVGLMC=False):
	"""
	 Deredden a flux vector using the Fitzpatrick (1999) parameterization
 
	 Parameters
	 ----------
	 wave :   array
			  Wavelength in Angstrom
	 flux :   array
			  Calibrated flux vector, same number of elements as wave.
	 ebv  :   float, optional
			  Color excess E(B-V). If a negative ebv is supplied,
			  then fluxes will be reddened rather than dereddened.
			  The default is 3.1.
	 AVGLMC : boolean
			  If True, then the default fit parameters c1,c2,c3,c4,gamma,x0 
			  are set to the average values determined for reddening in the 
			  general Large Magellanic Cloud (LMC) field by
			  Misselt et al. (1999, ApJ, 515, 128). The default is
			  False.
	 LMC2 :   boolean
			  If True, the fit parameters are set to the values determined
			  for the LMC2 field (including 30 Dor) by Misselt et al.
			  Note that neither `AVGLMC` nor `LMC2` will alter the default value 
			  of R_V, which is poorly known for the LMC.
   
	 Returns
	 -------             
	 new_flux : array 
				Dereddened flux vector, same units and number of elements
				as input flux.
 
	 Notes
	 -----

	 .. note:: This function was ported from the IDL Astronomy User's Library.

	 :IDL - Documentation:
 
	  PURPOSE:
	   Deredden a flux vector using the Fitzpatrick (1999) parameterization
	  EXPLANATION:
	   The R-dependent Galactic extinction curve is that of Fitzpatrick & Massa 
	   (Fitzpatrick, 1999, PASP, 111, 63; astro-ph/9809387 ).    
	   Parameterization is valid from the IR to the far-UV (3.5 microns to 0.1 
	   microns).    UV extinction curve is extrapolated down to 912 Angstroms.

	  CALLING SEQUENCE:
		FM_UNRED, wave, flux, ebv, [ funred, R_V = , /LMC2, /AVGLMC, ExtCurve= 
						  gamma =, x0=, c1=, c2=, c3=, c4= ]
	  INPUT:
		 WAVE - wavelength vector (Angstroms)
		 FLUX - calibrated flux vector, same number of elements as WAVE
				  If only 3 parameters are supplied, then this vector will
				  updated on output to contain the dereddened flux.
		 EBV  - color excess E(B-V), scalar.  If a negative EBV is supplied,
				  then fluxes will be reddened rather than dereddened.

	  OUTPUT:
		 FUNRED - unreddened flux vector, same units and number of elements
				  as FLUX

	  OPTIONAL INPUT KEYWORDS
		  R_V - scalar specifying the ratio of total to selective extinction
				   R(V) = A(V) / E(B - V).    If not specified, then R = 3.1
				   Extreme values of R(V) range from 2.3 to 5.3

	   /AVGLMC - if set, then the default fit parameters c1,c2,c3,c4,gamma,x0 
				 are set to the average values determined for reddening in the 
				 general Large Magellanic Cloud (LMC) field by Misselt et al. 
				 (1999, ApJ, 515, 128)
		/LMC2 - if set, then the fit parameters are set to the values determined
				 for the LMC2 field (including 30 Dor) by Misselt et al.
				 Note that neither /AVGLMC or /LMC2 will alter the default value 
				 of R_V which is poorly known for the LMC. 
			
		 The following five input keyword parameters allow the user to customize
		 the adopted extinction curve.    For example, see Clayton et al. (2003,
		 ApJ, 588, 871) for examples of these parameters in different interstellar
		 environments.

		 x0 - Centroid of 2200 A bump in microns (default = 4.596)
		 gamma - Width of 2200 A bump in microns (default  =0.99)
		 c3 - Strength of the 2200 A bump (default = 3.23)
		 c4 - FUV curvature (default = 0.41)
		 c2 - Slope of the linear UV extinction component 
			  (default = -0.824 + 4.717/R)
		 c1 - Intercept of the linear UV extinction component 
			  (default = 2.030 - 3.007*c2
	"""

	x = 10000./ wave # Convert to inverse microns 
	curve = x*0.

	# Set some standard values:
	x0 = 4.596
	gamma =  0.99
	c3 =  3.23      
	c4 =  0.41    
	c2 = -0.824 + 4.717/R_V
	c1 =  2.030 - 3.007*c2

	if LMC2:
		x0    =  4.626
		gamma =  1.05   
		c4   =  0.42   
		c3    =  1.92      
		c2    = 1.31
		c1    =  -2.16
	elif AVGLMC:   
		x0 = 4.596  
		gamma = 0.91
		c4   =  0.64  
		c3    =  2.73      
		c2    = 1.11
		c1    =  -1.28

	# Compute UV portion of A(lambda)/E(B-V) curve using FM fitting function and 
	# R-dependent coefficients
	xcutuv = np.array([10000.0/2700.0])
	xspluv = 10000.0/np.array([2700.0,2600.0])

	iuv = np.where(x >= xcutuv)[0]
	N_UV = len(iuv)
	iopir = np.where(x < xcutuv)[0]
	Nopir = len(iopir)
	if (N_UV > 0): xuv = np.concatenate((xspluv,x[iuv]))
	else:  xuv = xspluv

	yuv = c1  + c2*xuv
	yuv = yuv + c3*xuv**2/((xuv**2-x0**2)**2 +(xuv*gamma)**2)
	yuv = yuv + c4*(0.5392*(np.maximum(xuv,5.9)-5.9)**2+0.05644*(np.maximum(xuv,5.9)-5.9)**3)
	yuv = yuv + R_V
	yspluv  = yuv[0:2]  # save spline points
 
	if (N_UV > 0): curve[iuv] = yuv[2::] # remove spline points

	# Compute optical portion of A(lambda)/E(B-V) curve
	# using cubic spline anchored in UV, optical, and IR
	xsplopir = np.concatenate(([0],10000.0/np.array([26500.0,12200.0,6000.0,5470.0,4670.0,4110.0])))
	ysplir   = np.array([0.0,0.26469,0.82925])*R_V/3.1 
	ysplop   = np.array((np.polyval([-4.22809e-01, 1.00270, 2.13572e-04][::-1],R_V ), 
			np.polyval([-5.13540e-02, 1.00216, -7.35778e-05][::-1],R_V ), 
			np.polyval([ 7.00127e-01, 1.00184, -3.32598e-05][::-1],R_V ), 
			np.polyval([ 1.19456, 1.01707, -5.46959e-03, 7.97809e-04, -4.45636e-05][::-1],R_V ) ))
	ysplopir = np.concatenate((ysplir,ysplop))

	if (Nopir > 0): 
	  tck = interpolate.splrep(np.concatenate((xsplopir,xspluv)),np.concatenate((ysplopir,yspluv)),s=0)
	  curve[iopir] = interpolate.splev(x[iopir], tck)

	#Now apply extinction correction to input flux vector
	curve *= ebv

	return 10.**(0.4*curve)


def hpf(flux, windowsize=20, w_start=20):
	"""
	What does this one do ? High pass filtering ?
	"""
	D = np.size(flux)
	#w_start = w_start
	# Rita's typical inputs for SDSS:
	# w = 10 # 10
	# windowsize = 20 # 20

	# My MaNGA inputs:
	# if w == 0 and windowsize == 0:
	#     w = 40
	#     windowsize = 0
	if w_start == 0 and windowsize == 0:
		w_start = int(D/100.0)
		windowsize = 0.0

	h           = np.fft.fft(flux)
	h_filtered  = np.zeros(D,dtype=complex)
	window      = np.zeros(D)
	unwindow    = np.zeros(D)
	dw          = int(windowsize)
	dw_float    = float(windowsize)
	window[0]   = 1 # keep F0 for normalisation
	unwindow[0] = 1

	if windowsize > 0:
		for i in range(dw):
			window[w_start+i] = (i+1.0)/dw_float
			window[D-1-(w_start+dw-i)] = (dw_float-i)/dw_float

		window[w_start+dw:D-(w_start+dw)] = 1
	else:
		window[w_start:D-w_start] = 1


	unwindow        = 1 - window
	unwindow[0]     = 1

	h_filtered      = h * window
	un_h_filtered   = h*unwindow

	res     = np.real(np.fft.ifft(h_filtered))
	unres   = np.real(np.fft.ifft(un_h_filtered)) 
	res_out = (1.0+(res-np.median(res))/unres) * np.median(res) 

	return res_out 


def determine_attenuation(wave,data_flux,error_flux,model_flux,SPM,age,metal):
	"""
	Determines the dust attenuation to be applied to the models based on the data.
	 * 1. high pass filters the data and the models : makes hpf_model and hpf_data
	 * 2. normalises the hpf_models to the median hpf_data
	 * 3. fits the hpf models to data : chi2 maps
	 * 
	:param wave: wavelength
	:param data_flux: data flux
	:param error_flux: error flux
	:param model_flux: model flux
	:param SPM: SPM StellarPopulationModel object
	:param age: age
	:param metal: metallicity
	"""
	# 1. high pass filters the data and the models
	t_i = time.time()
	print('start dust determination', t_i)
	smoothing_length = SPM.dust_smoothing_length
	print('smoothing done', time.time()-t_i)
	hpf_data    = hpf(data_flux)
	print('hpf done', time.time()-t_i)
	hpf_models  = np.zeros(np.shape(model_flux))
	for m in range(len(model_flux)):
		hpf_models[m] = hpf(model_flux[m])
	print('propagated to models done', time.time()-t_i)

	zero_dat = np.where( (np.isnan(hpf_data)) & (np.isinf(hpf_data)) )
	hpf_data[zero_dat] = 0.0
	for m in range(len(model_flux)):
		hpf_models[m,zero_dat] = 0.0
	print('nans=>0', time.time()-t_i)
	hpf_error    = np.zeros(len(error_flux))
	hpf_error[:] = np.median(error_flux)/np.median(data_flux) * np.median(hpf_data)
	hpf_error[zero_dat] = np.max(hpf_error)*999999.9
	# 2. normalises the hpf_models to the median hpf_data
	hpf_models,mass_factors = normalise_spec(hpf_data,hpf_models)
	print('normalization', time.time()-t_i)
	# 3. fits the hpf models to data : chi2 maps
	hpf_weights,hpf_chis,hpf_branch = fitter(wave,hpf_data,hpf_error, hpf_models , SPM )
	print('fitting EBV', time.time()-t_i)
	# 4. use best fit to determine the attenuation curve : fine_attenuation
	best_fit_index  = [np.argmin(hpf_chis)]
	best_fit_hpf    = np.dot(hpf_weights[best_fit_index],hpf_models)[0]
	best_fit        = np.dot(hpf_weights[best_fit_index],model_flux)[0]
	fine_attenuation= (data_flux / best_fit) - (hpf_data/best_fit_hpf) + 1
	bad_atten = np.isnan(fine_attenuation) | np.isinf(fine_attenuation)
	fine_attenuation[bad_atten] = 1.0
	hpf_error[bad_atten] = np.max(hpf_error)*9999999999.9 
	fine_attenuation= fine_attenuation / np.median(fine_attenuation)
	print('finalize EBV', time.time()-t_i)
	# 5. propagates the hpf to the age and metallicity estimates
	av_age_hpf      = np.dot(hpf_weights,age)
	av_metal_hpf    = np.dot(hpf_weights,metal)
	print('get age, metal', time.time()-t_i)
	# 6. smoothes the attenuation curve obtained
	smooth_attenuation = curve_smoother(wave,fine_attenuation,smoothing_length)
	print('smoothes distribution', time.time()-t_i)

	# Fit a dust attenuation law to the best fit attenuation.
	if SPM.dust_law == 'calzetti':
		"""
		Assume E(B-V) distributed 0 to max_ebv.
		Uses the attenuation curves of Calzetti (2000) for starburst galaxies.
		"""
		num_laws = SPM.num_dust_vals
		ebv_arr     = np.arange(num_laws)/(SPM.max_ebv*num_laws*1.0)
		chi_dust    = np.zeros(num_laws)
		for ei,e in enumerate(ebv_arr):
			laws = np.array(dust_calzetti_py(e,wave))
			laws = laws/np.median(laws)
			chi_dust_arr = (smooth_attenuation-laws)**2
			chi_clipped_arr     = sigmaclip(chi_dust_arr, low=3.0, high=3.0)
			chi_clip_sq         = np.square(chi_clipped_arr[0])
			chi_dust[ei]        = np.sum(chi_clip_sq)

		dust_fit 			= ebv_arr[np.argmin(chi_dust)]
		#laws                = np.array(dust_calzetti_py(dust_fit,wave))
		#laws 				= laws/np.median(laws)
		#chi_dust_arr        = (smooth_attenuation-laws)**2
		#chi_clipped_arr     = sigmaclip(chi_dust_arr, low=3.0, high=3.0)
		#chi_clip_sq         = np.square(chi_clipped_arr[0])
		#clipped_arr         = np.where((chi_dust_arr > chi_clipped_arr[1]) & (chi_dust_arr < chi_clipped_arr[2]))[0]
		#	
		#for m in range(min([100,np.size(hpf_chis)])):
		#	sort_ind = np.argsort(hpf_chis)
		#	attenuation= data_flux/(np.dot(hpf_weights[sort_ind[m]],model_flux))
		#	attenuation= attenuation/np.median(attenuation)

	if SPM.dust_law == 'allen':
		"""
		Assume E(B-V) distributed 0 to max_ebv.
		Uses the attenuation curves of Allen (1976) of the Milky Way.
		"""
		num_laws = SPM.num_dust_vals
		ebv_arr     = np.arange(num_laws)/(SPM.max_ebv*num_laws*1.0)
		chi_dust    = np.zeros(num_laws)
		for ei,e in enumerate(ebv_arr):
			laws = np.array(dust_allen_py(e,wave))
			laws = laws/np.median(laws)
			chi_dust_arr = (smooth_attenuation-laws)**2
			chi_clipped_arr     = sigmaclip(chi_dust_arr, low=3.0, high=3.0)
			chi_clip_sq         = np.square(chi_clipped_arr[0])
			chi_dust[ei]        = np.sum(chi_clip_sq)

		dust_fit = ebv_arr[np.argmin(chi_dust)]

	if SPM.dust_law == 'prevot':
		"""
		Assume E(B-V) distributed 0 to max_ebv.
		Uses the attenuation curves of Prevot (1984) and Bouchert et al. (1985) for the Small Magellanic Cloud (SMC).
		"""
		num_laws = SPM.num_dust_vals
		ebv_arr     = np.arange(num_laws)/(SPM.max_ebv*num_laws*1.0)
		chi_dust    = np.zeros(num_laws)
		for ei,e in enumerate(ebv_arr):
			laws = np.array(dust_prevot_py(e,wave))
			laws = laws/np.median(laws)

			chi_dust_arr = (smooth_attenuation-laws)**2
			chi_clipped_arr     = sigmaclip(chi_dust_arr, low=3.0, high=3.0)
			chi_clip_sq         = np.square(chi_clipped_arr[0])
			chi_dust[ei]        = np.sum(chi_clip_sq)

		dust_fit = ebv_arr[np.argmin(chi_dust)]

	print('fits attenuation', time.time()-t_i)
	# print "Best E(B-V) = "+str(dust_fit)
	return dust_fit,smooth_attenuation

