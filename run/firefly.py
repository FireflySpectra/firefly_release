
"""
.. moduleauthor:: Daniel Thomas <daniel.thomas__at__port.ac.uk>
.. contributions:: Johan Comparat <johan.comparat__at__gmail.com>
.. contributions:: Violeta Gonzalez-Perez <violegp__at__gmail.com>
.. contributions:: Justus Neumann <jusneuma.astro__at__gmail.com>

Firefly is initiated with this script. 
All input data and parmeters are now specified in this one file.

"""

import sys, os

sys.path.append(os.path.join(os.getcwd(), "python"))
os.environ["FF_DIR"] = os.getcwd()
os.environ["STELLARPOPMODELS_DIR"] = os.path.join(os.environ["FF_DIR"], "stellar_population_models")

import numpy as np
from astropy.io import fits
import astropy.cosmology as co
import firefly_setup as fs
import firefly_models as fm
import time

t0=time.time()
cosmo = co.Planck15

#input file with path to read in wavelength, flux and flux error arrays
#the example is for an ascii file with extension 'dat'
input_file='example_data/spec-0266-51602-0001.dat'
data = np.loadtxt(input_file, unpack=True)
suffix = ""

#redshift
redshift = 0.021275453

wavelength = data[0,:]
flux = data[1,:]
error = data[2,:]
restframe_wavelength = wavelength/(1+redshift)

# RA and DEC
ra=145.89219 ; dec=0.059372

#velocity dispersion in km/s
vdisp = 135.89957

#instrumental resolution
r_instrument = np.zeros(len(wavelength))
for wi,w in enumerate(wavelength):
	r_instrument[wi] = 2000

# masking emission lines
# defines size in \AA of mask in pixels
# set to value>0 for masking (20 recommended), otherwise 0
N_angstrom_masked=0
# set emission lines to be masked, comment-out lines that should not be masked
emlines = [
						'He-II',# 'He-II:  3202.15A, 4685.74'
						'Ne-V', #  'Ne-V:   3345.81, 3425.81'
					    'O-II',#  'O-II:   3726.03, 3728.73'
					    'Ne-III',# 'Ne-III: 3868.69, 3967.40'
					    'H-ζ',#   'H-ζ:     3889.05'
					    'H-ε', #  'H-ε:     3970.07'
					    'H-δ',#   'H-δ:     4101.73'
					    'H-γ',#   'H-γ:     4340.46'
					    'O-III',# 'O-III:  4363.15, 4958.83, 5006.77'
					    'Ar-IV',# 'Ar-IV:  4711.30, 4740.10'
					    'H-β',#   'H-β:     4861.32'
					    'N-I',#   'H-I:    5197.90, 5200.39'
					    'He-I',#  'He-I:   5875.60'
					    'O-I',#   'O-I:    6300.20, 6363.67'
					    'N-II',#  'N-II:   6547.96, 6583.34'
					    'H-α',#   'H-α:     6562.80'
					    'S-II',#  'S-II:   6716.31, 6730.68'
					    'Ar-III',#'Ar-III: 7135.67'
						]

# choose model: 'm11', 'MaStar'
model_key='MaStar'

#model flavour
# m11: 'MILES', 'STELIB', 'ELODIE', 'MARCS (kr IMF only)'
# MaStar: 'gold'
model_lib=['gold']

# choose IMF: 'kr' (Kroupa), 'ss' (Salpeter)
imfs=['kr']

# minimum age and metallicity of models to be used 
# choose age in Gyr or 'AoU' for the age of the Universe
age_limits = [0,'AoU']
Z_limits = [-3.,3.]

#specify whether data in air or vaccum
data_wave_medium='vacuum'

#specify whether you would like to fit in air or vacuum wavelengths
fit_wave_medium='vacuum'

#Firefly assumes flux units of erg/s/A/cm^2.
#Choose factor in case flux is scaled
#(e.g. flux_units=10**(-17) for SDSS)
flux_units=10**(-17)

#specify whether write results
write_results=True

# set whether to correct for Milky Way reddening
milky_way_reddening=True

# set parameters for dust determination: 'on', 'hpf_only' (i.e. E(B-V)=0)
hpf_mode = 'on' 

# 'calzetti', 'allen', 'prevot' 
dust_law = 'calzetti'

# Only change the following parameters, if you know what you are doing.
max_ebv = 1.5                   
num_dust_vals = 200             
dust_smoothing_length = 200 
max_iterations = 10
pdf_sampling = 300 

print('')
print('Starting firefly ...')

age_min = age_limits[0]
if type(age_limits[1])==str:
	if age_limits[1]=='AoU':
		age_max = cosmo.age(redshift).value
	elif age_limits[1]!='AoU':
		print('Unrecognised maximum age limit. Try again.')
		sys.exit()
else:
	age_max = age_limits[1]
Z_min=Z_limits[0]
Z_max=Z_limits[1]

#set output folder and output filename in firefly directory
#and write output file
outputFolder = os.path.join( os.environ['FF_DIR'], 'output')
output_file = os.path.join( outputFolder , 'spFly-' + os.path.basename( input_file )[0:-4] ) + ".fits"

if os.path.isfile(output_file):
	print()
	print('Warning: This object has already been processed, the file will be over-witten.')
	answer = input('** Do you want to continue? (Y/N)')
	if (answer=='N' or answer=='n'):
		sys.exit()
	os.remove(output_file)
if os.path.isdir(outputFolder)==False:
	os.mkdir(outputFolder)
print()
print( 'Output file: ', output_file                 )
print()

prihdr = fm.pyfits.Header()
prihdr['FILE']          = os.path.basename(output_file)
prihdr['MODELS']	= model_key
prihdr['FITTER']	= "FIREFLY"	
prihdr['AGEMIN']	= str(age_min)		
prihdr['AGEMAX']	= str(age_max)
prihdr['ZMIN']	        = str(Z_min)
prihdr['ZMAX']	        = str(Z_max)
prihdr['redshift']	= redshift
prihdr['HIERARCH age_universe']	= np.round(cosmo.age(redshift).value,3)
prihdu = fm.pyfits.PrimaryHDU(header=prihdr)
tables = [prihdu]

#define input object to pass data on to firefly modules and initiate run
spec=fs.firefly_setup(input_file,milky_way_reddening=milky_way_reddening, \
                                  N_angstrom_masked=N_angstrom_masked,\
                                  hpf_mode=hpf_mode,data_wave_medium = data_wave_medium)

spec.openSingleSpectrum(wavelength, flux, error, redshift, ra, dec, vdisp, emlines, r_instrument)

did_not_converge = 0.
try :
	#prepare model templates
	model = fm.StellarPopulationModel(spec, output_file, cosmo, models = model_key, \
                                           model_libs = model_lib, imfs = imfs, \
                                           age_limits = [age_min,age_max], downgrade_models = True, \
                                           data_wave_medium = data_wave_medium, fit_wave_medium=fit_wave_medium,Z_limits = Z_limits, \
                                           suffix=suffix, use_downgraded_models = False, \
                                           dust_law=dust_law, max_ebv=max_ebv, num_dust_vals=num_dust_vals, \
                                           dust_smoothing_length=dust_smoothing_length,max_iterations=max_iterations, \
                                           pdf_sampling=pdf_sampling, flux_units=flux_units)


	#initiate fit
	model.fit_models_to_data()
	tables.append( model.tbhdu )
except (ValueError):
	tables.append( model.create_dummy_hdu() )
	did_not_converge +=1
	print('did not converge')
if did_not_converge < 1 :
	complete_hdus = fm.pyfits.HDUList(tables)
	if os.path.isfile(output_file):
		os.remove(output_file)		
	complete_hdus.writeto(output_file)

print()
print ("Done... total time:", int(time.time()-t0) ,"seconds.")
print()
