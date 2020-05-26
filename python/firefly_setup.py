"""
.. moduleauthor:: Johan Comparat <johan.comparat__at__gmail.com>
.. contributions:: Violeta Gonzalez-Perez <violegp__at__gmail.com>
..                 Sofia Meneses-Goytia <s.menesesgoytia__at__gmail.com>
..                 Justus Neumann <jusneuma.astro__at__gmail.com>
*General purpose*:
The class firefly_setup is dedicated to handling spectra to be fed to FIREFLY for fitting its stellar population
*Imports*::
	import numpy as np
	import astropy.io.fits as pyfits
	import glob
	from firefly_dust import get_dust_radec
"""

import numpy as np
import astropy.io.fits as pyfits
import glob
import sys,os
from firefly_dust import get_dust_radec

import astropy.cosmology as cc
cosmo = cc.Planck15
import astropy.units as uu
import cmath
import astropy.constants as const

class firefly_setup:
	"""
	Loads the environnement to transform observed spectra into the input for FIREFLY. 
	
	Currently SDSS spectra, speclite format is handled as well as stacks from the VVDS and the DEEP2 galaxy surveys.
	:param path_to_spectrum: path to the spectrum
	:param milky_way_reddening: True if you want to correct from the Milky way redenning using the Schlegel 98 dust maps.
	:param hpf_mode: models the dust attenuation observed in the spectrum using high pass filter.
	:param survey: name of the survey
	:param N_angstrom_masked: number ofangstrom masked around emission lines to look only at the continuum spectrum
	
	In this aims, it stores the following data in the object :
		* hdu list from the spec lite
		* SED data : wavelength (in angstrom), flux, error on the flux (in 10^{-17} erg/cm2/s/Angstrom, like the SDSS spectra)
		* Metadata :
			* ra : in degrees J2000
			* dec : in degrees J2000
			* redshift : best fit
			* vdisp : velocity dispersion in km/s
			* r_instrument : resolution of the instrument at each wavelength observed
			* trust_flag : 1 or True if trusted 
			* bad_flags : ones as long as the wavelength array, filters the pixels with bad data
			* objid : object id optional : set to 0
		
	"""
	def __init__(self,path_to_spectrum, milky_way_reddening=True , hpf_mode = 'on', N_angstrom_masked = 20.):
		self.path_to_spectrum=path_to_spectrum
		self.milky_way_reddening = milky_way_reddening
		self.hpf_mode = hpf_mode
		self.N_angstrom_masked = N_angstrom_masked
	
	def openSingleSpectrum(self, wavelength, flux, error, redshift, ra, dec, vdisp, lines_mask, r_instrument):
        
		assert len(wavelength)==len(flux)==len(error)==len(lines_mask)==len(r_instrument),\
			"The arrays wavelength, flux, error, lines_mask and r_instrument must have identical lengths."
		
		self.wavelength=wavelength
		self.flux=flux
		self.error=error
		self.redshift=redshift
		self.ra=ra
		self.dec=dec
		self.vdisp=vdisp
		self.lines_mask=lines_mask
		self.r_instrument=r_instrument
		
		self.DL = cosmo.luminosity_distance(self.redshift).to(uu.cm)
		self.bad_flags = np.ones(len(self.wavelength))
		self.bad_flags = self.bad_flags[(self.lines_mask==False)]
		self.restframe_wavelength = self.wavelength/(1+self.redshift)
		self.trust_flag = 1
		self.objid = 0
		
		self.restframe_wavelength = self.restframe_wavelength[(self.lines_mask==False)] 
		self.wavelength = self.wavelength[(self.lines_mask==False)] 
		self.flux = self.flux[(self.lines_mask==False)]
		self.error = self.error[(self.lines_mask==False)]
		self.r_instrument = self.r_instrument[(self.lines_mask==False)]
		
		# removes the bad data from the spectrum 
		self.bad_data = np.isnan(self.flux) | np.isinf(self.flux) | (self.flux <= 0.0) | np.isnan(self.error) | np.isinf(self.error)
		self.flux[self.bad_data]     = 0.0
		self.error[self.bad_data]     = np.max(self.flux) * 99999999999.9
		self.bad_flags[self.bad_data] = 0
		        
		if self.milky_way_reddening:
		# gets the amount of MW reddening on the models
			self.ebv_mw = get_dust_radec(self.ra,self.dec,'ebv')
		else:
			self.ebv_mw = 0.0
		#print(self.ebv_mw)	


	def openMANGASpectrum(self, path_to_logcube, path_to_dapall, bin_number, plate_number, ifu_number):
		"""Loads an observed MaNGA spectrum in.
		:param data_release: Must specify which data release of MaNGA you are using, as file structure has changed.
		:param path_to_logcube: Must specify the path to logcube (if using MPL5 or higher). Set to 0 otherwise.		
		"""
		
		# Read in MAPS file as this contains part of the information.
		maps_header = pyfits.open(self.path_to_spectrum)
		bin_identification = maps_header['BINID'].data
		where = np.where(bin_number == bin_identification[0,:,:]) #use 1st channel of bin_identification
		x_position, y_position = where[0][0], where[1][0]
		
		# Get S/N, right ascension and declination.
		signal, ra, dec = maps_header['BIN_SNR'].data[x_position,y_position], maps_header[0].header['OBJRA'],maps_header[0].header['OBJDEC']
		velocity_dispersion = maps_header['STELLAR_SIGMA'].data 		# DO NOT USE VELOCITY DISPERSION CORRECTION!
		vdisp = velocity_dispersion[x_position,y_position]
		
		# Open LOGCUBE to get the flux, wavelength, and error
		header = pyfits.open(path_to_logcube)
		wavelength, flux, emline, bit_mask, inverse_variance = header['WAVE'].data, header['FLUX'].data, header['EMLINE'].data, header['MASK'].data, header['IVAR'].data
		self.wavelength = wavelength
		correct_flux = flux[:,x_position,y_position]
		correct_flux_emline = emline[:, x_position, y_position]
		output_flux = correct_flux - correct_flux_emline
		correct_inverse_variance = inverse_variance[:, x_position, y_position]
		
		self.error = np.sqrt(1.0/(correct_inverse_variance))
		self.bad_flags = np.ones(len(output_flux))
		self.flux = output_flux
		self.vdisp = vdisp
		#self.lines_mask=lines_mask

		mask = bit_mask[:,x_position,y_position]# | self.lines_mask
		self.wavelength = self.wavelength[(mask==False)] 
		self.flux = self.flux[(mask==False)] 
		self.error = self.error[(mask==False)]
		self.bad_flags = self.bad_flags[(mask==False)]
		
		dap_all = pyfits.open(path_to_dapall)
		get = np.where(dap_all[1].data['PLATEIFU']==str(plate_number)+'-'+str(ifu_number))
		c = const.c.value/1000
		# Use redshift as measured from the stellar kinematics by the DAP.
		redshift = dap_all[1].data['STELLAR_Z'][get][0]
		# If redshift measurement failed, use redshift estimate from NSA or ancillary programs.
		if redshift<0:
			redshift = dap_all[1].data['Z'][get][0]
			
		sys_vel = maps_header[0].header['SCINPVEL']
		bin_vel = maps_header['STELLAR_VEL'].data[x_position,y_position]	
	
		if np.abs(redshift*c-sys_vel)>1:
			print('The are problems with the redshift estimate.')
			print('c*STELLAR_Z = '+str(redshift*c)+', sys_vel = '+str(sys_vel))
		
		if redshift<0:
			print('There are problems with the redshift.')
			print('z = {}'.format(redshift))
		
		redshift_corr = (sys_vel+bin_vel)/c
		self.redshift = redshift
		self.restframe_wavelength = self.wavelength / (1.0+redshift_corr)
					
		# Get Trust flag, object_id, xpos, ypos and instrumental resolution.
		self.trust_flag, self.objid, self.r_instrument = True, 0, np.loadtxt(os.path.join(os.environ['FF_DIR'],'data/MaNGA_spectral_resolution.txt'))
		self.r_instrument = self.r_instrument[0:self.r_instrument.shape[0]//2]
		self.r_instrument = self.r_instrument[(mask==False)]
		self.xpos, self.ypos = ra, dec
		
		# gets the amount of MW reddening on the models
		if self.milky_way_reddening :
			self.ebv_mw = get_dust_radec(ra, dec, 'ebv')
		else:
			self.ebv_mw = 0.0
