
"""
.. moduleauthor:: Daniel Thomas <daniel.thomas__at__port.ac.uk>
.. contributions:: Johan Comparat <johan.comparat__at__gmail.com>
.. contributions:: Violeta Gonzalez-Perez <violegp__at__gmail.com>

Firefly is initiated with this script. 
All input data and parmeters are now specified in this one file.

"""

import numpy as np
import sys, os
from os.path import join

os.environ["FF_DIR"] = os.getcwd()
os.environ["PYTHONPATH"] = os.path.join(os.getcwd(), "python")
os.environ["STELLARPOPMODELS_DIR"] = os.path.join(os.getcwd(), "stellar_population_models")

import time
import firefly_setup as setup
import firefly_models as spm
import astropy.cosmology as co

t0=time.time()

#redshift
redshift = 1.33

#input file with path to read in wavelength, flux and flux error arrays
#the example is for an ascii file with extension 'ascii'
input_file='example_data/CDFS022490.ascii'
data = np.loadtxt(input_file, unpack=True)
lamb = data[0,:]
wavelength = data[0,:][np.where(lamb>3600*(1+redshift))]
#wavelength = data[0,:][np.where(np.logical_and(lamb>=min(wave_model)*(1+redshift),lamb<=max(wave_model)*(1+redshift)))]
flux = data[1,:][np.where(lamb>3600*(1+redshift))]
error = flux*0.1
restframe_wavelength = wavelength/(1+redshift)

# RA and DEC
ra=53.048 ; dec=-27.72

#velocity dispersion
vdisp = 220.

#instrumental resolution
r_instrument = np.zeros(len(wavelength))
for wi,w in enumerate(wavelength):
	r_instrument[wi] = 600

# masking emission lines
# N_angstrom_masked set to 20 in _init_ function
N_angstrom_masked=20
lines_mask = ((restframe_wavelength > 3728 - N_angstrom_masked) & (restframe_wavelength < 3728 + N_angstrom_masked)) | ((restframe_wavelength > 5007 - N_angstrom_masked) & (restframe_wavelength < 5007 + N_angstrom_masked)) | ((restframe_wavelength > 4861 - N_angstrom_masked) & (restframe_wavelength < 4861 + N_angstrom_masked)) | ((restframe_wavelength > 6564 - N_angstrom_masked) & (restframe_wavelength < 6564 + N_angstrom_masked)) 

#key which models and minimum age and metallicity of models to be used 
models_key='m11'
cosmo = co.Planck15
ageMin = 0. ; ageMax = cosmo.age(redshift).value
ZMin = 0.001 ; ZMax = 10.

#model flavour
model_libs=['MILES']

#model imf
imfs=['kr']

#specify whether data in air or vaccum
data_wave_medium='air'

#Firefly assumes flux units of erg/s/A/cm^2.
#Choose factor in case flux is scaled
#(e.g. flux_units=10**(-17) for SDSS)
flux_units=1

#specify whether models should be downgraded
#to the instrumental resolution and galaxy velocity dispersion
downgrade_models=True

#specify whether write results
write_results=True

#set output folder and output filename in firefly directory
#and write output file
outputFolder = join( os.environ['FF_DIR'], 'output')
output_file = join( outputFolder , 'spFly-' + os.path.basename( input_file )[0:-6] ) + ".fits"
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
prihdr = spm.pyfits.Header()
prihdr['FILE']          = os.path.basename(output_file)
prihdr['MODELS']	= models_key
prihdr['FITTER']	= "FIREFLY"	
prihdr['AGEMIN']	= str(ageMin)		
prihdr['AGEMAX']	= str(ageMax)
prihdr['ZMIN']	        = str(ZMin)
prihdr['ZMAX']	        = str(ZMax)
prihdr['redshift']	= redshift
prihdr['HIERARCH age_universe']	= np.round(cosmo.age(redshift).value,3)
prihdu = spm.pyfits.PrimaryHDU(header=prihdr)
tables = [prihdu]

#define input object to pass data on to firefly modules and initiate run
spec=setup.firefly_setup(input_file,N_angstrom_masked=N_angstrom_masked)
spec.openSingleSpectrum(wavelength, flux, error, redshift, ra, dec, vdisp, lines_mask, r_instrument)

did_not_converge = 0.
try :
	#prepare model templates
	model = spm.StellarPopulationModel(spec, output_file, cosmo, models = models_key, model_libs = model_libs, imfs = imfs, age_limits = [ageMin,ageMax], downgrade_models = downgrade_models, data_wave_medium = data_wave_medium, Z_limits = [ZMin,ZMax], use_downgraded_models = False, write_results = write_results, flux_units=flux_units)
	#initiate fit
	model.fit_models_to_data()
	tables.append( model.tbhdu )
except (ValueError):
	tables.append( model.create_dummy_hdu() )
	did_not_converge +=1
	print('did not converge')
if did_not_converge < 1 :
	complete_hdus = spm.pyfits.HDUList(tables)
	if os.path.isfile(output_file):
		os.remove(output_file)		
	complete_hdus.writeto(output_file)

print()
print ("Done... total time:", int(time.time()-t0) ,"seconds.")
print()
