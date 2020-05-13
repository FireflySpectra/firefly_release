
"""
Created on Tue May 12 11:00:35 2020

.. moduleauthor:: Justus Neumann <jusneuma.astro__at__gmail.com>
.. contributions:: Daniel Goddard <daniel.goddard__at__port.ac.uk>

General purpose:
................

Reads in a MaNGA datacube and analyses each spectrum from Voronoi binned spectra.
Run this script by calling > python firefly_manga.py <txt_file>.
The text file should specify in one line, space-separated: <plate number> <ifu number> <bin number>.
The bin number is optional and should only be given if only a single bin is meant to be fitted.

"""

import os
import sys
import numpy as np
import astropy.io.fits as fits
import astropy.cosmology as co
import time
from multiprocessing import Pool
import multiprocessing

import firefly_setup as fs
import firefly_models as fm

# ------------------------------
# START Configuration

logcube_directory = '/home/jneumann/firefly/MaNGA/MPL-9/'
maps_directory = '/home/jneumann/firefly/MaNGA/MPL-9/'
output_directory = '/home/jneumann/firefly/MaNGA/MPL-9/output/debug/'
dap_path = 'home/jneumann/firefly/MaNGA/MPL-9/dapall-v2_7_1-2.4.1.fits'
suffix = '-SPM-MaStar.fits'

cosmo = co.Planck15
mpl = 'MPL9'                     # 'MPL5','MPL6','MPL7','MPL8','MPL9'
models = 'm11'                     # 'bc03', 'm09', 'm11', 'MaStar'
model_libs = ['MILES']               # 'MILES', 'STELIB', 'ELODIE', 'MARCS', for MaStar models use 'Th-MaStar' or 'E-MaStar'
imfs = ['kr']                   # 'kr', 'ss', 'cha'; cha currently not supported for MaStar models
age_limits = [0,'AoU']          # chose age in Gyr or 'AoU' for the age of the universe
Z_limits = [-3.,3.]
data_wave_medium = 'vacuum'     # 'vacuum', 'air'
downgrade_models = True
use_downgrade_models = False
milky_way_reddening=True
hpf_mode = 'on'                 # 'on', 'hpf_only' (i.e. E(B-V)=0)
dust_law = 'calzetti'            # 'calzetti', 'allen', 'prevot' 
N_angstrom_masked=20                 # Number of Angstrom masked around emission lines
flux_units=10**(-17)

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
    outFile = direc+'/manga-'+str(plate)+'-'+str(ifu)+'-bin'+str(i)
    spec_MaNGA = fs.firefly_setup(maps, milky_way_reddening=milky_way_reddening, \
                                  N_angstrom_masked=N_angstrom_masked,\
                                  hpf_mode=hpf_mode)
    spec_MaNGA.openObservedMANGASpectrum(mpl, logcube, dap_path, galaxy_bin_number, plate, ifu)
    
    age_min = age_limits[0]
    if type(age_limits[1])==str:
        if age_limits[1]=='AoU':
            age_max = cosmo.age(spec_MaNGA.redshift).value
        elif age_limits[1]!='AoU':
            print('Unrecognised maximum age limit. Try again.')
            sys.exit()
    else:
        age_max = age_limits[1]
    
    model_sdss = fm.StellarPopulationModel(spec_MaNGA, outFile, cosmo, models = models, \
                                           model_libs = model_libs, imfs = imfs, \
                                           age_limits = [age_min,age_max], downgrade_models = downgrade_models, \
                                           data_wave_medium = data_wave_medium, Z_limits = Z_limits, \
                                           suffix=suffix, use_downgraded_models = use_downgrade_models, \
                                           dust_law=dust_law, max_ebv=max_ebv, num_dust_vals=num_dust_vals, \
                                           dust_smoothing_length=dust_smoothing_length,max_iterations=max_iterations, \
                                           pdf_sampling=pdf_sampling, flux_units=flux_units)
    model_sdss.fit_models_to_data()

def f_mp(unique_bin_number):       #multiprocessing function
    if __name__ == '__main__':
        pool = Pool(processes=multiprocessing.cpu_count()) #Define number of cores to use
        pool.map(f, unique_bin_number)
        pool.close()
        pool.join()

# ------------------------------
# START Running firefly
		
start = time.time()
print('')
print('Starting firefly ...')
dat = np.loadtxt(sys.argv[1])
plate = int(dat[0])
ifu = int(dat[1])
print('Plate = {}, IFU = {}'.format(plate,ifu))

# Set MAPS and LOGCUBE paths.
logcube = os.path.join(logcube_directory,'manga-'+str(plate)+'-'+str(ifu)+'-LOGCUBE-VOR10-MILESHC-MASTARHC.fits.gz')
maps = os.path.join(maps_directory,'manga-'+str(plate)+'-'+str(ifu)+'-MAPS-VOR10-MILESHC-MASTARHC.fits.gz')

# Create output path if it doesn't exist.
direc = os.path.join(output_directory,str(plate),str(ifu))
if not os.path.exists(direc):
    os.makedirs(direc)

# Read in MAPS file as this contains part of the information.
maps_header = fits.open(maps)
unique_bin_number = list(np.unique(maps_header['BINID'].data)[1:])
print('Number of bins = {}'.format(len(unique_bin_number)))

if len(dat)==3:
	bin_number = int(dat[2])
	f(bin_number)
else:
	N = f_mp(unique_bin_number)

print('Time to complete: ', (time.time())-start)