"""
.. moduleauthor:: Daniel Goddard <daniel.goddard__at__port.ac.uk>

General purpose:
................

Reads in a MaNGA datacube and analyses each spectrum from Voronoi binned spectra.

*Imports*::

	from os.path import join
	import os
	import glob
	import numpy as np
	import pyfits
	import astropy.cosmology as co
	import matplotlib.pyplot as plt
	import GalaxySpectrumFIREFLY as gs
	import StellarPopulationModel as spm
	from firefly_dust import get_dust_radec

"""
#! /usr/bin/env python
from os.path import join
import os
import glob
import numpy as np
import pyfits
import astropy.cosmology as co
cosmo = co.Planck15
import matplotlib.pyplot as plt
import GalaxySpectrumFIREFLY as gs
import StellarPopulationModel as spm
from firefly_dust import get_dust_radec

# Example of how to run a MaNGA spectrum.
maps_directory = '../data//manga-7495-12704-MAPS-VOR10-GAU-MILESHC.fits'
logcube_directory = '../data/manga-7495-12704-LOGCUBE-VOR10-GAU-MILESHC.fits'
drp_path = '../data/drpall-v2_0_1.fits'

# Read in MAPS file as this contains part of the information.
splitting = maps_directory.replace('/Users/Daniel/Downloads/', '').split('-')
maps_header = pyfits.open(maps_directory)
unique_bin_number = np.unique(maps_header['BINID'].data)[1:]

for i in range(len(unique_bin_number)):
	galaxy_bin_number  = unique_bin_number[i]
	spec_MaNGA = gs.GalaxySpectrumFIREFLY(maps_directory, milky_way_reddening=True)
	spec_MaNGA.openObservedMANGASpectrum('MPL5', logcube_directory, drp_path, galaxy_bin_number, 7495, 12704)
	outFile = "/Users/Daniel/Documents/NEW_7495-12704"
	model_sdss = spm.StellarPopulationModel(spec_MaNGA, outFile, cosmo, models = 'm11', model_libs = ['MILES'], imfs = ['kr'], age_limits = [6,10], downgrade_models = True, data_wave_medium = 'vacuum', Z_limits = [-3.,1.],suffix="-SPM-MILES.fits", use_downgraded_models = False)
	model_sdss.fit_models_to_data()

	stop


