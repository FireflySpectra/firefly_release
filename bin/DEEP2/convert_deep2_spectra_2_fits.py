#! /usr/bin/env python
import sys
from os.path import join
import os
import time
import numpy as np
import glob
import astropy.cosmology as co
cosmo = co.Planck13

# for one galaxy spectrum
import GalaxySpectrumFIREFLY as gs
import StellarPopulationModel as spm

import astropy.io.fits as fits

catalog=fits.open(join(os.environ['DEEP2_DIR'], "catalogs", "zcat.deep2.dr4.v4.LFcatalogTC.Planck15.fits"))[1].data

outputFolder = join( os.environ['DEEP2_DIR'], 'spec')

def convert_spec_2_fits(catalog_entry, output_file, mask, objno):
	path_to_spectrum = glob.glob(join(os.environ['DEEP2_DIR'], 'spectra', mask, '*', '*' + objno + '*_fc_tc.dat'))
	print path_to_spectrum
	
	if len(path_to_spectrum)>=1:
		spec=gs.GalaxySpectrumFIREFLY("-", milky_way_reddening=True)
		spec.openObservedDEEP2pectrum(catalog_entry)

		prihdr = fits.Header()
		prihdr['FILE']          = os.path.basename(output_file)
		prihdr['MASK']          = catalog_entry['MASK'] 
		prihdr['OBJNO']         = catalog_entry['OBJNO']   
		prihdr['RA']	  	  	= catalog_entry['RA']
		prihdr['DEC']	    	= catalog_entry['DEC']
		prihdr['redshift']	    = catalog_entry['ZBEST']
		prihdu = fits.PrimaryHDU(header=prihdr)

		waveCol = fits.Column(name="wavelength",format="D", unit="Angstrom", array= spec.wavelength)
		dataCol = fits.Column(name="flux",format="D", unit="1e-17erg/s/cm2/Angstrom", array= spec.flux)
		errorCol = fits.Column(name="flux_error",format="D", unit="1e-17erg/s/cm2/Angstrom", array= spec.error)
		
		cols = fits.ColDefs([ waveCol, dataCol, errorCol]) 
		tbhdu = fits.BinTableHDU.from_columns(cols)


		complete_hdus = fits.HDUList([prihdu, tbhdu])
		if os.path.isfile(output_file):
			os.remove(output_file)
		complete_hdus.writeto(output_file)
	

print len(catalog), "N lines in the catalog"
for catalog_entry in catalog:
	mask=str(catalog_entry['MASK'])
	objno=str(catalog_entry['OBJNO'])
	output_file = join(outputFolder, 'deep2-'+mask+'-'+objno +".fits")
	#if os.path.isfile(output_file):
	#	print "pass", output_file
	#else:
	convert_spec_2_fits(catalog_entry, output_file, mask, objno)
		
