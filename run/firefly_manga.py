
"""
Created on Tue May 12 11:00:35 2020

.. moduleauthor:: Justus Neumann <jusneuma.astro__at__gmail.com>
.. contributions:: Daniel Goddard <daniel.goddard__at__port.ac.uk>

General purpose:
................

Reads in a MaNGA datacube and analyses each spectrum from Voronoi binned spectra.
The bin number is optional and should only be given if only a single bin is meant to be fitted.

"""

import sys, os

sys.path.append(os.path.join(os.getcwd(), "python"))
os.environ["FF_DIR"] = os.getcwd()
os.environ["STELLARPOPMODELS_DIR"] = os.path.join(os.environ["FF_DIR"], "stellar_population_models")

import numpy as np
import astropy.io.fits as fits
import astropy.cosmology as co
import firefly_setup as fs
import firefly_models as fm
import time

from multiprocessing import Pool
import multiprocessing

t0=time.time()
cosmo = co.Planck15

# paths to input files
logcube_dir = 'example_data/manga/'
maps_dir = 'example_data/manga/'
output_dir = 'output/manga/'
dap_file = 'example_data/manga/dapall-v2_4_3-2.2.1.fits'
suffix = ''

# define plate number, IFU number, bin number
plate = 8080
ifu = 12701
bin_number = 0 #'all' if entire IFU, otherwise bin number

# masking emission lines
# defines size of mask in pixels
# set to value>0 for masking (20 recommended), otherwise 0
N_angstrom_masked=20 
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

# choose model: 'm11', 'MaStar')
model_key = 'MaStar'

# model flavour
# m11: 'MILES', 'STELIB', 'ELODIE', 'MARCS'
# MaStar: 'Th-MaStar', 'E-MaStar'
model_lib = ['E-MaStar']

# choose IMF: 'kr' (Kroupa), 'ss' (Salpeter)
imfs = ['kr']

# minimum age and metallicity of models to be used 
# choose age in Gyr or 'AoU' for the age of the Universe
age_limits = [0,'AoU']
Z_limits = [-3.,3.]

#specify data medium: 'air', 'vaccum'
data_wave_medium = 'vacuum'
#Firefly assumes flux units of erg/s/A/cm^2.
#Choose factor in case flux is scaled
#(e.g. flux_units=10**(-17) for MaNGA)
flux_units=10**(-17)

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

# END Configuration
# ------------------------------

def f(i):
    galaxy_bin_number  = i
    print('Fitting bin number {}'.format(i))
    output_file = direc+'/spFly-'+str(plate)+'-'+str(ifu)+'-bin'+str(i)+'.fits'
    spec = fs.firefly_setup(maps, milky_way_reddening=milky_way_reddening, \
                                  N_angstrom_masked=N_angstrom_masked,\
                                  hpf_mode=hpf_mode)
    spec.openMANGASpectrum(logcube, dap_file, galaxy_bin_number, plate, ifu, emlines)
    
    age_min = age_limits[0]
    if type(age_limits[1])==str:
        if age_limits[1]=='AoU':
            age_max = cosmo.age(spec.redshift).value
        elif age_limits[1]!='AoU':
            print('Unrecognised maximum age limit. Try again.')
            sys.exit()
    else:
        age_max = age_limits[1]
    
    #prepare model templates
    model = fm.StellarPopulationModel(spec, output_file, cosmo, models = model_key, \
                                           model_libs = model_lib, imfs = imfs, \
                                           age_limits = [age_min,age_max], downgrade_models = True, \
                                           data_wave_medium = data_wave_medium, Z_limits = Z_limits, \
                                           suffix=suffix, use_downgraded_models = False, \
                                           dust_law=dust_law, max_ebv=max_ebv, num_dust_vals=num_dust_vals, \
                                           dust_smoothing_length=dust_smoothing_length,max_iterations=max_iterations, \
                                           pdf_sampling=pdf_sampling, flux_units=flux_units)
    #initiate fit
    model.fit_models_to_data()

def f_mp(unique_bin_number):       #multiprocessing function
    if __name__ == '__main__':
        pool = Pool(processes=multiprocessing.cpu_count()) #Define number of cores to use
        pool.map(f, unique_bin_number)
        pool.close()
        pool.join()

# ------------------------------
# START Running firefly
		
print('')
print('Starting firefly ...')
print('Plate = {}, IFU = {}'.format(plate,ifu))

# Set MAPS and LOGCUBE paths.
logcube = os.path.join(logcube_dir,'manga-'+str(plate)+'-'+str(ifu)+'-LOGCUBE-VOR10-GAU-MILESHC.fits.gz')
maps = os.path.join(maps_dir,'manga-'+str(plate)+'-'+str(ifu)+'-MAPS-VOR10-GAU-MILESHC.fits.gz')

# Create output path if it doesn't exist.
direc = os.path.join(output_dir,str(plate),str(ifu))
if not os.path.exists(direc):
    os.makedirs(direc)

# Read in MAPS file as this contains part of the information.
maps_header = fits.open(maps)
unique_bin_number = list(np.unique(maps_header['BINID'].data)[1:])
print('Number of bins = {}'.format(len(unique_bin_number)))

if bin_number=='all':
  N = f_mp(unique_bin_number)
else:
  f(bin_number)

print('Time to complete: ', (time.time())-t0)
