
from os.path import join
import os
import glob
import numpy as np
import pyfits
import astropy.cosmology as co
cosmo = co.Planck13

import GalaxySpectrumFIREFLY as gs
import StellarPopulationModel as spm
from firefly_dust import get_dust_radec

import time
import datetime

date_stamp = datetime.datetime.today().strftime('%d%b%y')

# example of how to run a GC spectrum in firefly (FF)

# setting up directories and file to be tested
gcs_directory = './example_data/spectra/'
output_directory = '../output/'

if os.path.isdir(output_directory)==False:
    os.mkdir(output_directory) 

gcs_file = 'NGC7099_2016-10-01.fits'

# select the M11-library and IMF type for the models to be used
models_used = 'ELODIE'
imf_used = 'cha'

# the resolution of the M11-MARCS and M11-ELODIE models are higher than the GC data, they require to be convolved
if models_used == 'MARCS' or models_used == 'ELODIE':
    convolve_models = True

# the resolution of the M11-MILES and M11-STELIB models are lower than the GC data, so no convolution required
# FF will convolve the GC data to match the one of the models - although it is not recommended to ever downgrade observed data
if models_used == 'MILES' or models_used == 'STELIB':
    convolve_models = False

# setting up the routines for FF
spec_gcs = gs.GalaxySpectrumFIREFLY(gcs_directory+gcs_file, milky_way_reddening=True)
spec_gcs.openGCsUsher()

# chosen name for the output file to be unique every time FF is run
outFile = output_directory+gcs_file+'_'+imf_used+'_'+date_stamp

# FF set-up and ready to run
model_gcs = spm.StellarPopulationModel(spec_gcs, outFile, cosmo, models = 'm11', model_libs = [models_used], imfs = [imf_used], age_limits = [0,15], Z_limits = [-3.,5.], wave_limits = [3350.,9000.], 
	     suffix="_SPM-"+models_used+".fits", downgrade_models = convolve_models, data_wave_medium = 'vacuum', use_downgraded_models = False, write_results = True)
model_gcs.fit_models_to_data()

print('     > done')

exit()

