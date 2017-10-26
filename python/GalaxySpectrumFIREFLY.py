"""
.. moduleauthor:: Johan Comparat <johan.comparat__at__gmail.com>
.. contributions:: Violeta Gonzalez-Perez <violegp__at__gmail.com>
..                 Sofia Meneses-Goytia <s.menesesgoytia__at__gmail.com>

*General purpose*:

The class GalaxySpectrumFIREFLY is dedicated to handling spectra to be fed to FIREFLY for fitting its stellar population

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
cosmo = cc.Planck13
import astropy.units as uu
import cmath

class GalaxySpectrumFIREFLY:
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

	def openObservedSDSSSpectrum(self, survey='sdssMain'):
		"""
		It reads an SDSS spectrum and provides the input for the firefly fitting routine.

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
		self.hdulist = pyfits.open(self.path_to_spectrum)
		self.ra = self.hdulist[0].header['RA']
		self.dec = self.hdulist[0].header['DEC']

		self.wavelength = 10**self.hdulist[1].data['loglam']
		self.flux = self.hdulist[1].data['flux']
		self.error = self.hdulist[1].data['ivar']**(-0.5)
		self.bad_flags = np.ones(len(self.wavelength))
		if survey=='sdssMain':
			self.redshift = self.hdulist[2].data['Z'][0] 
		if survey=='sdss3':
			self.redshift = self.hdulist[2].data['Z_NOQSO'][0] 
		if survey=='sdss4':
			self.redshift = self.hdulist[2].data['Z_NOQSO'][0] 
			
		self.vdisp = self.hdulist[2].data['VDISP'][0]
		self.restframe_wavelength = self.wavelength / (1.0+self.redshift)

		self.trust_flag = 1
		self.objid = 0

		# masking emission lines
		lines_mask = ((self.restframe_wavelength > 3728 - self.N_angstrom_masked) & (self.restframe_wavelength < 3728 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 5007 - self.N_angstrom_masked) & (self.restframe_wavelength < 5007 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 4861 - self.N_angstrom_masked) & (self.restframe_wavelength < 4861 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 6564 - self.N_angstrom_masked) & (self.restframe_wavelength < 6564 + self.N_angstrom_masked)) 

		self.restframe_wavelength = self.restframe_wavelength[(lines_mask==False)] 
		self.wavelength = self.wavelength[(lines_mask==False)] 
		self.flux = self.flux[(lines_mask==False)] 
		self.error = self.error[(lines_mask==False)] 
		self.bad_flags = self.bad_flags[(lines_mask==False)] 		
		
		bad_data = np.isnan(self.flux) | np.isinf(self.flux) | (self.flux <= 0.0) | np.isnan(self.error) | np.isinf(self.error)
		# removes the bad data from the spectrum 
		self.flux[bad_data] 	= 0.0
		self.error[bad_data] 	= np.max(self.flux) * 99999999999.9
		self.bad_flags[bad_data] = 0

		self.r_instrument = np.zeros(len(self.wavelength))
		for wi,w in enumerate(self.wavelength):
			if w<6000:
				self.r_instrument[wi] = (2270.0-1560.0)/(6000.0-3700.0)*w + 420.0 
			else:
				self.r_instrument[wi] = (2650.0-1850.0)/(9000.0-6000.0)*w + 250.0 


		if self.milky_way_reddening :
			# gets the amount of MW reddening on the models
			self.ebv_mw = get_dust_radec(self.ra,self.dec,'ebv')
		else:
			self.ebv_mw = 0.0



	def openObservedDEEP2pectrum(self, catalog_entry, survey='deep2_ascii'):
		"""
		It reads a DEEP2 spectrum and provides the input for the firefly fitting routine.
		"""

		self.ebv_mw = 0.0

		if survey=='deep2_ascii':
			mask=str(catalog_entry['MASK'])
			objno=str(catalog_entry['OBJNO'])

			path_to_spectrum = glob.glob(os.path.join(os.environ['DEEP2_DIR'], 'spectra', mask, '*', '*' + objno + '*_fc_tc.dat'))[0]
		
			wl, fl, flErr= np.loadtxt(path_to_spectrum, unpack=True)

			self.wavelength = wl	
			self.flux, self.error= fl * 1e17, flErr * 1e17
			
			self.ra = catalog_entry['RA']
			self.dec = catalog_entry['DEC']
			self.redshift = catalog_entry['ZBEST']

			if self.milky_way_reddening :
				self.ebv_mw = catalog_entry['SFD_EBV']
		
		if survey=='deep2_fits':
			path_to_spectrum = catalog_entry
			self.hdulist = pyfits.open(self.path_to_spectrum)

			self.ra = self.hdulist[0].header['RA']
			self.dec = self.hdulist[0].header['DEC']
			self.redshift = self.hdulist[0].header['REDSHIFT']
			
			self.wavelength = self.hdulist[1].data['wavelength']
			self.flux = self.hdulist[1].data['flux']
			self.error = self.hdulist[1].data['flux_error']

		self.bad_flags = np.ones(len(self.wavelength))

		self.restframe_wavelength = self.wavelength / (1.0+self.redshift)

		self.vdisp = 60. #catalog_entry['VDISP']
		self.trust_flag = 1
		self.objid = 0

		# masking emission lines
		lines_mask = ((self.restframe_wavelength > 3728 - self.N_angstrom_masked) & (self.restframe_wavelength < 3728 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 5007 - self.N_angstrom_masked) & (self.restframe_wavelength < 5007 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 4861 - self.N_angstrom_masked) & (self.restframe_wavelength < 4861 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 6564 - self.N_angstrom_masked) & (self.restframe_wavelength < 6564 + self.N_angstrom_masked)) 

		self.restframe_wavelength = self.restframe_wavelength[(lines_mask==False)] 
		self.wavelength = self.wavelength[(lines_mask==False)] 
		self.flux = self.flux[(lines_mask==False)] 
		self.error = self.error[(lines_mask==False)] 
		self.bad_flags = self.bad_flags[(lines_mask==False)] 		
		
		bad_data = np.isnan(self.flux) | np.isinf(self.flux) | (self.flux <= 0.0) | np.isnan(self.error) | np.isinf(self.error)
		# removes the bad data from the spectrum 
		self.flux[bad_data] 	= 0.0
		self.error[bad_data] 	= np.max(self.flux) * 99999999999.9
		self.bad_flags[bad_data] = 0

		self.r_instrument = np.zeros(len(self.wavelength))
		for wi,w in enumerate(self.wavelength):
			self.r_instrument[wi] = 6000.


	def openObservedMANGASpectrum(self, data_release, path_to_logcube, path_to_drpall, bin_number, plate_number, ifu_number):
		"""Loads an observed MaNGA spectrum in.
		:param data_release: Must specify which data release of MaNGA you are using, as file structure has changed.
		:param data_release: Must specify the path to logcube (if using MPL5 or higher). Set to 0 otherwise.		
		"""
		if data_release == 'MPL5':

			# Read in MAPS file as this contains part of the information.
			maps_header = pyfits.open(self.path_to_spectrum)
			bin_identification = maps_header['BINID'].data
			where = np.where(bin_number == bin_identification)
			x_position, y_position = where[0][0], where[1][0]

			# Get S/N, right ascension and declination.
			signal, ra, dec = maps_header['BIN_SNR'].data[x_position,y_position], maps_header[0].header['OBJRA'],maps_header[0].header['OBJDEC']

			# Correct sigma for instrumental resolution
			velocity_dispersion_wrong = maps_header['STELLAR_SIGMA'].data
			velocity_dispersion_correction = maps_header['STELLAR_SIGMACORR'].data
			if velocity_dispersion_wrong[x_position,y_position] > velocity_dispersion_correction[x_position,y_position]:
				correction = np.sqrt((velocity_dispersion_wrong[x_position,y_position])**2-(velocity_dispersion_correction[x_position,y_position])**2)
				vdisp = correction
			else:
				correction = cmath.sqrt((velocity_dispersion_wrong[x_position,y_position])**2-(velocity_dispersion_correction[x_position,y_position])**2)
				vdisp = velocity_dispersion_wrong[x_position,y_position]

			# Open LOGCUBE to get the flux, wavelength, and error
			header = pyfits.open(path_to_logcube)
			wavelength, flux, emline, emline_base, bit_mask, inverse_variance = header['WAVE'].data, header['FLUX'].data, header['EMLINE'].data, header['EMLINE_BASE'].data,header['MASK'].data, header['IVAR'].data
			self.wavelength = wavelength

			correct_flux = flux[:,x_position,y_position]
			correct_flux_emline = emline[:, x_position, y_position]
			output_flux = correct_flux - correct_flux_emline
			correct_inverse_variance = inverse_variance[:, x_position, y_position]
					
			self.error = np.sqrt(1.0/(correct_inverse_variance))
			self.bad_flags = np.ones(len(output_flux))
			self.flux = output_flux
			self.vdisp = vdisp

			# Open drp all file to get the correct redshift of the galaxy.
			dap_all = pyfits.open(path_to_drpall)
			main_header, manga_plate, manga_ifu, manga_redshift = dap_all[1].data, [], [], []
			for q in range(len(main_header)):
				galaxy = main_header[q]
				plate, ifu, z = galaxy['plate'], galaxy['ifudsgn'],galaxy['nsa_z']	
				manga_plate.append(plate)
				manga_ifu.append(ifu)
				manga_redshift.append(z)

			manga_plate, manga_ifu, manga_redshift = np.array(manga_plate,dtype=int), np.array(manga_ifu,dtype=int), np.array(manga_redshift)
			where = np.where((manga_plate == int(plate_number)))
			ifu_sorted = manga_ifu[where]
			redshift_sorted = manga_redshift[where]
			where_1 = np.where(int(ifu_number) == ifu_sorted)
			redshift = redshift_sorted[where_1][0]
			self.restframe_wavelength = wavelength / (1.0+redshift)
			self.redshift = redshift

			# Get Trust flag, object_id, xpos, ypos and instrumental resolution.
			self.trust_flag, self.objid, self.r_instrument = True, 0, np.loadtxt('../bin_MaNGA/MaNGA_spectral_resolution.txt')
			self.xpos, self.ypos = ra, dec 
			if self.milky_way_reddening :
				# gets the amount of MW reddening on the models
				self.ebv_mw = get_dust_radec(ra, dec, 'ebv')
			else:
				self.ebv_mw = 0.0

	def openEllipticalsSMG(self):
		hdulist = pyfits.open(self.path_to_spectrum)
		hdulist.info()

		ra = hdulist[0].header['RA']
		dec = hdulist[0].header['DEC']
		redshift = hdulist[0].header['REDSHIFT']

		naxis1 = hdulist[0].header['NAXIS1']
		cdelt1 = hdulist[0].header['CDELT1']
		crval1 = hdulist[0].header['CRVAL1']
		crval2 = hdulist[0].header['CRVAL2']
		restframe_wavelength = np.arange(crval1,crval2,cdelt1)
		wavelength = restframe_wavelength * (1. + redshift)

		meanWL = (wavelength[1:]+wavelength[:-1])/2.
		deltaWL = hdulist[0].header['FWHM']
		resolution = np.ones_like(wavelength)*np.mean(meanWL / deltaWL)
		vdisp  = hdulist[0].header['VELDISP']

		flux = hdulist[0].data
		error = np.zeros(len(flux))
		bad_flags = np.ones(len(restframe_wavelength))

		bad_data = np.isnan(flux) | np.isinf(flux) | (flux <= 0.0) | np.isnan(error) | np.isinf(error)
		# removes the bad data from the spectrum 
		flux[bad_data] = 0.0
		error[bad_data] = np.max(flux) * 99999999999.9
		bad_flags[bad_data] = 0
		
		#import matplotlib.pyplot as plt
		#plt.plot(wavelength, flux, 'r')
		#plt.show()
		#stop

		self.xpos, self.ypos, self.redshift, self.restframe_wavelength, self.wavelength = ra, dec, redshift, restframe_wavelength, wavelength
		self.flux, self.error, self.bad_flags, self.r_instrument, self.vdisp = flux, error, bad_flags, resolution, vdisp
		self.trust_flag, self.objid = True, ''

		if self.milky_way_reddening :
			# gets the amount of MW reddening on the models
			self.ebv_mw = get_dust_radec(ra,dec,'ebv')
		else:
			self.ebv_mw = 0.0
			
	def openGCsUsher(self):
			hdulist = pyfits.open(self.path_to_spectrum)
			hdulist.info()

			ra = hdulist[0].header['RA']
			dec = hdulist[0].header['DEC']
			redshift = hdulist[0].header['REDSHIFT']
	
			naxis1 = hdulist[0].header['NAXIS2']
			cdelt1 = hdulist[0].header['CDELT1']
			crval1 = hdulist[0].header['CRVAL1']
			crval2 = hdulist[0].header['CRVAL2']
			restframe_wavelength = np.arange(crval1,crval2,cdelt1)
			wavelength = restframe_wavelength * (1. + redshift)
			
			meanWL = (wavelength[1:]+wavelength[:-1])/2.
			deltaWL = hdulist[0].header['FWHM']
			resolution = np.ones_like(wavelength)*np.mean(meanWL / deltaWL)
			vdisp  = hdulist[0].header['VELDISP']

			flux = hdulist[0].data
			flux = flux.flatten()
			error = hdulist[1].data
			error = error.flatten()
			bad_flags = np.ones(len(restframe_wavelength))
	
			bad_data = np.isnan(flux) | np.isinf(flux) | (flux <= 0.0) | np.isnan(error) | np.isinf(error) 
			# removes the bad data from the spectrum 
			flux[bad_data] = 0.0
			error[bad_data] = np.max(flux) * 99999999999.9
			bad_flags[bad_data] = 0

			self.xpos, self.ypos, self.redshift, self.restframe_wavelength, self.wavelength = ra, dec, redshift, restframe_wavelength, wavelength
			self.flux, self.error, self.bad_flags, self.r_instrument, self.vdisp = flux, error, bad_flags, resolution, vdisp
			self.trust_flag, self.objid = True, ''

			if self.milky_way_reddening :
			# gets the amount of MW reddening on the models
				self.ebv_mw = get_dust_radec(ra,dec,'ebv')
			else:
				self.ebv_mw = 0.0

#--------------------Under development

	def openILLUSTRISsimulatedSpectrum(self, fractional_error=0.1 ):
		"""
		Reads the spectra from the Illustris simulation and converts it to the inputs needed by firefly.
		"""
		self.ra=0.
		self.dec=0.
		self.redshift = 0.01
		
		hdu=pyfits.open(self.path_to_spectrum) 
		spec_data = hdu[7].data
		area = 4. * np.pi * cosmo.luminosity_distance(self.redshift).to(u.cm)**2.
		
		self.restframe_wavelength = spec_data['lambda']*u.m.to(u.AA)
		self.wavelength = self.restframe_wavelength * (1+ self.redshift)

		self.flux = spec_data['L_lambda']*u.W.to(u.erg/u.s)/u.m.to(u.AA)/area.value * 1e17 # 1e-17 erg/cm2/s/A
		self.error = self.flux * fractional_error
		self.bad_flags = np.ones(len(self.restframe_wavelength))
		
		self.vdisp = 70.
		self.trust_flag = 1
		self.objid = 0
		
		lines_mask = ((self.restframe_wavelength > 3728 - self.N_angstrom_masked) & (self.restframe_wavelength < 3728 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 5007 - self.N_angstrom_masked) & (self.restframe_wavelength < 5007 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 4861 - self.N_angstrom_masked) & (self.restframe_wavelength < 4861 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 6564 - self.N_angstrom_masked) & (self.restframe_wavelength < 6564 + self.N_angstrom_masked)) 

		self.restframe_wavelength = self.restframe_wavelength[(lines_mask==False)] 
		self.wavelength = self.wavelength[(lines_mask==False)] 
		self.flux = self.flux[(lines_mask==False)] 
		self.error = self.error[(lines_mask==False)] 
		self.bad_flags = self.bad_flags[(lines_mask==False)] 		
		
		bad_data = np.isnan(self.flux) | np.isinf(self.flux) | (self.flux <= 0.0) | np.isnan(self.error) | np.isinf(self.error)
		# removes the bad data from the spectrum 
		self.flux[bad_data] 	= 0.0
		self.error[bad_data] 	= np.max(self.flux) * 99999999999.9
		self.bad_flags[bad_data] = 0

		self.r_instrument = np.zeros(len(self.wavelength))
		for wi,w in enumerate(self.wavelength):
			if w<6000:
				self.r_instrument[wi] = (2270.0-1560.0)/(6000.0-3700.0)*w + 420.0 
			else:
				self.r_instrument[wi] = (2650.0-1850.0)/(9000.0-6000.0)*w + 250.0 

		self.ebv_mw = 0.0

	def openGAMAsimulatedSpectrum(self, error_multiplicative_factor = 1.):
		"""
		Opens the smulated data set
		filename = os.path.join(os.environ['DATA_DIR'], "spm", "GAMAmock/gal_0000_GAMA_M10_z0.15.dat")
		"""
		data = np.loadtxt(self.path_to_spectrum, unpack=True, skiprows=1)
		f=open(self.path_to_spectrum, 'r')
		self.redshift = float(f.readline())
		f.close()
		DL = cosmo.luminosity_distance(self.redshift).to(uu.cm)
		m_w_2_erg_p_s = uu.W.to(uu.erg/uu.s) * 10.**(17.)
		self.ra=0.
		self.dec=0.
		self.wavelength = data[0] * (1+ self.redshift)
		self.flux = data[1] * m_w_2_erg_p_s / DL.value**2
		self.error = data[2] * m_w_2_erg_p_s / DL.value**2 * error_multiplicative_factor
		self.bad_flags = np.ones(len(self.wavelength))
		
		self.vdisp = 70.
		self.restframe_wavelength = data[0]

		self.trust_flag = 1
		self.objid = 0

		# masking emission lines
		lines_mask = ((self.restframe_wavelength > 3728 - self.N_angstrom_masked) & (self.restframe_wavelength < 3728 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 5007 - self.N_angstrom_masked) & (self.restframe_wavelength < 5007 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 4861 - self.N_angstrom_masked) & (self.restframe_wavelength < 4861 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 6564 - self.N_angstrom_masked) & (self.restframe_wavelength < 6564 + self.N_angstrom_masked)) 

		self.restframe_wavelength = self.restframe_wavelength[(lines_mask==False)] 
		self.wavelength = self.wavelength[(lines_mask==False)] 
		self.flux = self.flux[(lines_mask==False)] 
		self.error = self.error[(lines_mask==False)] 
		self.bad_flags = self.bad_flags[(lines_mask==False)] 		
		
		bad_data = np.isnan(self.flux) | np.isinf(self.flux) | (self.flux <= 0.0) | np.isnan(self.error) | np.isinf(self.error)
		# removes the bad data from the spectrum 
		self.flux[bad_data] 	= 0.0
		self.error[bad_data] 	= np.max(self.flux) * 99999999999.9
		self.bad_flags[bad_data] = 0

		self.r_instrument = np.zeros(len(self.wavelength))
		for wi,w in enumerate(self.wavelength):
			if w<6000:
				self.r_instrument[wi] = (2270.0-1560.0)/(6000.0-3700.0)*w + 420.0 
			else:
				self.r_instrument[wi] = (2650.0-1850.0)/(9000.0-6000.0)*w + 250.0 

		self.ebv_mw = 0.0


	def openObservedStack(self, fluxKeyword='medianWeightedStack'):
		"""
		It reads an Stack spectrum from the LF analysis and provides the input for the firefly fitting routine.
		:param fluxKeyword: parameter to choose the mean or the median stack 'meanWeightedStack', 'medianWeightedStack'
		"""
		self.hdulist = pyfits.open(self.path_to_spectrum)
		self.ra = 0. #self.hdulist[0].header['RA']
		self.dec = 0. #self.hdulist[0].header['DEC']
		self.redshift = float(os.path.basename(self.path_to_spectrum).split('-')[-1].split('_')[0][1:])
		self.restframe_wavelength = self.hdulist[1].data['wavelength']
		self.wavelength = self.restframe_wavelength * (1. + self.redshift)
		
		meanWL = (self.wavelength[1:]+self.wavelength[:-1])/2.
		deltaWL = self.wavelength[1:]-self.wavelength[:-1]
		resolution = np.ones_like(self.wavelength)*np.mean(meanWL / deltaWL)
		
		#self.flux = self.hdulist[1].data['meanWeightedStack'] * 1e17
		self.flux = self.hdulist[1].data[fluxKeyword] * 1e17
		self.error = self.hdulist[1].data['jackknifStackErrors'] * 1e17
		self.bad_flags = np.ones(len(self.restframe_wavelength))
		Nstacked = float(self.path_to_spectrum.split('-')[-1].split('_')[3])

		lines_mask = ((self.restframe_wavelength > 3728 - self.N_angstrom_masked) & (self.restframe_wavelength < 3728 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 5007 - self.N_angstrom_masked) & (self.restframe_wavelength < 5007 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 4861 - self.N_angstrom_masked) & (self.restframe_wavelength < 4861 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 6564 - self.N_angstrom_masked) & (self.restframe_wavelength < 6564 + self.N_angstrom_masked)) | ( self.hdulist[1].data['NspectraPerPixel'] < Nstacked * 0.8 ) | (self.flux==-9999.99)

		self.restframe_wavelength = self.restframe_wavelength[(lines_mask==False)] 
		self.wavelength = self.wavelength[(lines_mask==False)] 
		self.flux = self.flux[(lines_mask==False)] 
		self.error = self.error[(lines_mask==False)] 
		self.bad_flags = self.bad_flags[(lines_mask==False)] 
		self.r_instrument = resolution[(lines_mask==False)] 

		bad_data 	= np.isnan(self.flux) | np.isinf(self.flux) | (self.flux <= 0.0) | np.isnan(self.error) | np.isinf(self.error) 
		# removes the bad data from the spectrum 
		self.flux[bad_data] 	= 0.0
		self.error[bad_data] 	= np.max(self.flux) * 99999999999.9
		self.bad_flags[bad_data] = 0
		
		self.vdisp = 70. # km/s

		self.trust_flag = 1
		self.objid = 0

		if self.milky_way_reddening :
			# gets the amount of MW reddening on the models
			self.ebv_mw = get_dust_radec(self.ra,self.dec,'ebv')
		else:
			self.ebv_mw = 0.0

		#print "there are", len(self.wavelength),"data points at redshift",self.redshift," between", np.min(self.wavelength[bad_data==False]), np.max(self.wavelength[bad_data==False]), "Angstrom.", np.min(self.restframe_wavelength[bad_data==False]), np.max(self.restframe_wavelength[bad_data==False]), "Angstrom in the rest frame."

	def openObservedStackTutorial(self):
		"""
		It reads an Stack spectrum from the LF analysis and provides the input for the firefly fitting routine.
		:param path_to_spectrum:
		:param sdss_dir: directory with the observed spectra
		:param milky_way_reddening: True or False if you want to correct the redenning of the Milky way.
		:param hpf_mode: 'on' high pass filters the data to correct from dust in the galaxy.

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
		self.hdulist = pyfits.open(self.path_to_spectrum)
		self.ra = 0. #self.hdulist[0].header['RA']
		self.dec = 0. #self.hdulist[0].header['DEC']
		self.redshift = float(os.path.basename(self.path_to_spectrum).split('-')[-1].split('_')[0][1:])
		self.restframe_wavelength = self.hdulist[1].data['WAVE'][0]
		self.wavelength = self.restframe_wavelength * (1. + self.redshift)
		
		meanWL = (self.wavelength[1:]+self.wavelength[:-1])/2.
		deltaWL = self.wavelength[1:]-self.wavelength[:-1]
		resolution = np.ones_like(self.wavelength)*np.mean(meanWL / deltaWL)
		# units of 1e-17 f lambda
		self.flux = self.hdulist[1].data['FLUXMEDIAN'][0]# * 1e-17
		self.error = self.hdulist[1].data['FLUXMEDIAN_ERR'][0]# * 1e-17
		self.bad_flags = np.ones(len(self.restframe_wavelength))
		Nstacked = float(self.path_to_spectrum.split('-')[-1].split('_')[3])
		
		lines_mask = ((self.restframe_wavelength > 3728 - self.N_angstrom_masked) & (self.restframe_wavelength < 3728 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 5007 - self.N_angstrom_masked) & (self.restframe_wavelength < 5007 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 4861 - self.N_angstrom_masked) & (self.restframe_wavelength < 4861 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 6564 - self.N_angstrom_masked) & (self.restframe_wavelength < 6564 + self.N_angstrom_masked)) 

		self.restframe_wavelength = self.restframe_wavelength[(lines_mask==False)] 
		self.wavelength = self.wavelength[(lines_mask==False)] 
		self.flux = self.flux[(lines_mask==False)] 
		self.error = self.error[(lines_mask==False)] 
		self.bad_flags = self.bad_flags[(lines_mask==False)] 
		self.r_instrument = resolution[(lines_mask==False)] 

		bad_data 	= np.isnan(self.flux) | np.isinf(self.flux) | (self.flux <= 0.0) | np.isnan(self.error) | np.isinf(self.error) 
		# removes the bad data from the spectrum 
		self.flux[bad_data] 	= 0.0
		self.error[bad_data] 	= np.max(self.flux) * 99999999999.9
		self.bad_flags[bad_data] = 0
		
		self.vdisp = 70. # km/s

		self.trust_flag = 1
		self.objid = 0

		if self.milky_way_reddening :
			# gets the amount of MW reddening on the models
			self.ebv_mw = get_dust_radec(self.ra,self.dec,'ebv')
		else:
			self.ebv_mw = 0.0

		# # print"there are", len(self.wavelength),"data points at redshift",self.redshift," between", np.min(self.wavelength[bad_data==False]), np.max(self.wavelength[bad_data==False]), "Angstrom.", np.min(self.restframe_wavelength[bad_data==False]), np.max(self.restframe_wavelength[bad_data==False]), "Angstrom in the rest frame."
		
	def openStackEBOSS(self, redshift = 0.85, fluxKeyword='medianWeightedStack'):
		self.hdulist = pyfits.open(self.path_to_spectrum)
		self.ra = 0. #self.hdulist[0].header['RA']
		self.dec = 0. #self.hdulist[0].header['DEC']
		self.redshift = redshift
		self.restframe_wavelength = self.hdulist[1].data['wavelength']
		self.wavelength = self.restframe_wavelength * (1. + self.redshift)
		
		meanWL = (self.wavelength[1:]+self.wavelength[:-1])/2.
		deltaWL = self.wavelength[1:]-self.wavelength[:-1]
		resolution = np.ones_like(self.wavelength)*np.mean(meanWL / deltaWL)
		
		self.flux = self.hdulist[1].data[fluxKeyword] #* 10**(-17)
		self.error = self.hdulist[1].data['jackknifStackErrors'] #* 10**(-17)
		self.bad_flags = np.ones(len(self.restframe_wavelength))
		
		lines_mask = ((self.restframe_wavelength > 3728 - self.N_angstrom_masked) & (self.restframe_wavelength < 3728 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 5007 - self.N_angstrom_masked) & (self.restframe_wavelength < 5007 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 4861 - self.N_angstrom_masked) & (self.restframe_wavelength < 4861 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 6564 - self.N_angstrom_masked) & (self.restframe_wavelength < 6564 + self.N_angstrom_masked)) 

		self.restframe_wavelength = self.restframe_wavelength[(lines_mask==False)] 
		self.wavelength = self.wavelength[(lines_mask==False)] 
		self.flux = self.flux[(lines_mask==False)] 
		self.error = self.error[(lines_mask==False)] 
		self.bad_flags = self.bad_flags[(lines_mask==False)] 
		self.r_instrument = resolution[(lines_mask==False)] 

		bad_data 	= np.isnan(self.flux) | np.isinf(self.flux) | (self.flux <= 0.0) | np.isnan(self.error) | np.isinf(self.error) 
		# removes the bad data from the spectrum 
		self.flux[bad_data] 	= 0.0
		self.error[bad_data] 	= np.max(self.flux) * 99999999999.9
		self.bad_flags[bad_data] = 0
		
		self.vdisp = 70. # km/s

		self.trust_flag = 1
		self.objid = 0

		if self.milky_way_reddening :
			# gets the amount of MW reddening on the models
			self.ebv_mw = get_dust_radec(self.ra,self.dec,'ebv')
		else:
			self.ebv_mw = 0.0

		# # print "there are", len(self.wavelength),"data points at redshift",self.redshift," between", np.min(self.wavelength[bad_data==False]), np.max(self.wavelength[bad_data==False]), "Angstrom.", np.min(self.restframe_wavelength[bad_data==False]), np.max(self.restframe_wavelength[bad_data==False]), "Angstrom in the rest frame."

	def openObservedVVDSpectrum(self, catalog_entry, survey='vvds'):
		"""
		It reads a VVDS spectrum and provides the input for the firefly fitting routine.
		"""
		self.hdulist = pyfits.open(glob.glob(os.path.join(os.environ['VVDS_DIR'], 'spectra',"sc_*" + str(catalog_entry['NUM']) + "*atm_clean.fits"))[0])
		wl=self.hdulist[0].header['CRVAL1'] + self.hdulist[0].header['CDELT1'] * np.arange(2,self.hdulist[0].header['NAXIS1']+2)
		fl=self.hdulist[0].data[0]
		
		correctionAperture = 1. / catalog_entry['fo']
		
		noiseFileName=glob.glob(glob.glob(os.path.join(os.environ['VVDS_DIR'], 'spectra', "sc_*"+str(catalog_entry['NUM'])+"*noise.fits"))[0])[0]
		noiseHDU=pyfits.open(noiseFileName)
		flErr=noiseHDU[0].data[0]
		
		self.wavelength,self.flux,self.error=wl, fl*correctionAperture * 1e17, flErr*correctionAperture * 1e17
		
		self.ra = catalog_entry['ALPHA']
		self.dec = catalog_entry['DELTA']

		self.bad_flags = np.ones(len(self.wavelength))
		self.redshift = catalog_entry['Z']
			
		self.vdisp = 2000. #catalog_entry['VDISP']
		self.restframe_wavelength = self.wavelength / (1.0+self.redshift)

		self.trust_flag = 1
		self.objid = 0

		# masking emission lines
		lines_mask = ((self.restframe_wavelength > 3728 - self.N_angstrom_masked) & (self.restframe_wavelength < 3728 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 5007 - self.N_angstrom_masked) & (self.restframe_wavelength < 5007 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 4861 - self.N_angstrom_masked) & (self.restframe_wavelength < 4861 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 6564 - self.N_angstrom_masked) & (self.restframe_wavelength < 6564 + self.N_angstrom_masked)) 

		self.restframe_wavelength = self.restframe_wavelength[(lines_mask==False)] 
		self.wavelength = self.wavelength[(lines_mask==False)] 
		self.flux = self.flux[(lines_mask==False)] 
		self.error = self.error[(lines_mask==False)] 
		self.bad_flags = self.bad_flags[(lines_mask==False)] 		
		
		bad_data = np.isnan(self.flux) | np.isinf(self.flux) | (self.flux <= 0.0) | np.isnan(self.error) | np.isinf(self.error)
		# removes the bad data from the spectrum 
		self.flux[bad_data] 	= 0.0
		self.error[bad_data] 	= np.max(self.flux) * 99999999999.9
		self.bad_flags[bad_data] = 0

		self.r_instrument = np.zeros(len(self.wavelength))
		for wi,w in enumerate(self.wavelength):
			self.r_instrument[wi] = 220.

		if self.milky_way_reddening :
			self.ebv_mw = catalog_entry['EBV_MW']
		else:
			self.ebv_mw = 0.0


	def openObservedVIPERSpectrum(self, catalog_entry, survey='vipers'):
		"""
		It reads a VVDS spectrum and provides the input for the firefly fitting routine.
		"""
		self.field='W'+catalog_entry['id_IAU'][7]
		specFileName=os.path.join(os.environ['VIPERS_DIR'], 'spectra',"VIPERS_"+ self.field+ "_PDR2_SPECTRA_1D",catalog_entry['id_IAU'][:6]+"_"+catalog_entry['id_IAU'][7:]+".fits")
		
		self.hdulist = pyfits.open(specFileName)
		
		wlA=self.hdulist[1].data['WAVES']
		flA=self.hdulist[1].data['FLUXES']
		flErrA=self.hdulist[1].data['NOISE']
		mask=self.hdulist[1].data['MASK']
		wl, fl, flErr= wlA[(mask==0)], flA[(mask==0)], flErrA[(mask==0)]

		correctionAperture = 1. / catalog_entry['fo']
		
		self.wavelength,self.flux,self.error=wl, fl*correctionAperture * 1e17, flErr*correctionAperture * 1e17
		
		self.ra = catalog_entry['ALPHA']
		self.dec = catalog_entry['DELTA']

		self.bad_flags = np.ones(len(self.wavelength))
		self.redshift = catalog_entry['zspec']
			
		self.vdisp = 2000. #catalog_entry['VDISP']
		self.restframe_wavelength = self.wavelength / (1.0+self.redshift)

		self.trust_flag = 1
		self.objid = 0

		# masking emission lines
		lines_mask = ((self.restframe_wavelength > 3728 - self.N_angstrom_masked) & (self.restframe_wavelength < 3728 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 5007 - self.N_angstrom_masked) & (self.restframe_wavelength < 5007 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 4861 - self.N_angstrom_masked) & (self.restframe_wavelength < 4861 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 6564 - self.N_angstrom_masked) & (self.restframe_wavelength < 6564 + self.N_angstrom_masked)) 

		self.restframe_wavelength = self.restframe_wavelength[(lines_mask==False)] 
		self.wavelength = self.wavelength[(lines_mask==False)] 
		self.flux = self.flux[(lines_mask==False)] 
		self.error = self.error[(lines_mask==False)] 
		self.bad_flags = self.bad_flags[(lines_mask==False)] 
		
		bad_data = np.isnan(self.flux) | np.isinf(self.flux) | (self.flux <= 0.0) | np.isnan(self.error) | np.isinf(self.error)

		self.flux[bad_data] 	= 0.0
		self.error[bad_data] 	= np.max(self.flux) * 99999999999.9
		self.bad_flags[bad_data] = 0

		self.r_instrument = np.zeros(len(self.wavelength))
		for wi,w in enumerate(self.wavelength):
			self.r_instrument[wi] = 220.

		if self.milky_way_reddening :
			self.ebv_mw = catalog_entry['EBV']
		else:
			self.ebv_mw = 0.0


	def openObservedMuseSpectrum(self, catalog):
		"""Loads an observed MUSE spectrum in counts.
		:param catalog: name of the catalog with redshifts.
		"""
		self.wavelength, flA, flErrA = np.loadtxt(self.path_to_spectrum, unpack=True)
		self.flux, self.error = flA*1e-3, flErrA*1e-3 # units of 1e-17
		self.bad_flags = np.ones(len(self.wavelength))
		bad_data = np.isnan(self.flux) | np.isinf(self.flux) | (self.flux <= 0.0) | np.isnan(self.error) | np.isinf(self.error)
		# removes the bad data from the spectrum 
		self.flux[bad_data] 	= 0.0
		self.error[bad_data] 	= np.max(self.flux) * 99999999999.9
		self.bad_flags[bad_data] = 0
		
		self.redshift = catalog['FINAL_Z']
		self.vdisp = 100 # catalog['VDISP']
		self.restframe_wavelength = self.wavelength / (1.0+self.redshift)

		# masking emission lines
		lines_mask = ((self.restframe_wavelength > 3728 - self.N_angstrom_masked) & (self.restframe_wavelength < 3728 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 5007 - self.N_angstrom_masked) & (self.restframe_wavelength < 5007 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 4861 - self.N_angstrom_masked) & (self.restframe_wavelength < 4861 + self.N_angstrom_masked)) | ((self.restframe_wavelength > 6564 - self.N_angstrom_masked) & (self.restframe_wavelength < 6564 + self.N_angstrom_masked)) 

		self.restframe_wavelength = self.restframe_wavelength[(lines_mask==False)] 
		self.wavelength = self.wavelength[(lines_mask==False)] 
		self.flux = self.flux[(lines_mask==False)] 
		self.error = self.error[(lines_mask==False)] 
		self.bad_flags = self.bad_flags[(lines_mask==False)] 	
		
		self.r_instrument = np.zeros(len(self.wavelength))
		for wi,w in enumerate(self.wavelength):
			if w<6000:
				self.r_instrument[wi] = (2270.0-1560.0)/(6000.0-3700.0)*w + 420.0 
			else:
				self.r_instrument[wi] = (2650.0-1850.0)/(9000.0-6000.0)*w + 250.0 


		self.trust_flag = 1
		self.objid = 0

		if self.milky_way_reddening :
			# gets the amount of MW reddening on the models
			self.ebv_mw = get_dust_radec(catalog['ALPHA'], catalog['DELTA'], 'ebv')
		else:
			self.ebv_mw = 0.

	def openSAMIgalaxy(self, xposition, yposition):

		from astropy.io import fits
		header = fits.open(self.path_to_spectrum)
		print header['PRIMARY'].header
		stop

		# Open up the header to get key information to construct wavelength vector.
		crval3, cdelt, naxis, crpix = header['PRIMARY'].header['CRVAL3'], header['PRIMARY'].header['CDELT3'],header['PRIMARY'].header['NAXIS3'],header['PRIMARY'].header['CRPIX3']
		crval = crval3-(crpix*cdelt)
		wavelength = np.arange(naxis)*cdelt+crval

		# Now open up the flux and variance for an arbitrary spaxel. Use the variance to get error spectrum.
		flux, variance = header['PRIMARY'].data[:,xposition,yposition], header['VARIANCE'].data[:,xposition,yposition]
		error = np.sqrt(variance)

		# Redshift, sigma etc.
		redshift, ra, dec = 0.050,214.36654,0.48281
		restframe_wavelength = wavelength/(1+redshift)
		vdisp = 100

		bad_flags = np.ones(len(restframe_wavelength))
		bad_data = np.isnan(flux) | np.isinf(flux) | (flux <= 0.0) | np.isnan(error) | np.isinf(error)
		flux[bad_data] = 0.0
		error[bad_data] = np.max(flux) * 99999999999.9
		bad_flags[bad_data] = 0
		resolution = np.ones_like(wavelength)*1700

		self.xpos, self.ypos, self.redshift, self.restframe_wavelength, self.wavelength = ra, dec, redshift, restframe_wavelength, wavelength
		self.flux, self.error, self.bad_flags, self.r_instrument, self.vdisp = flux, error, bad_flags, resolution, vdisp
		self.trust_flag, self.objid = True, ''
		#import matplotlib.pyplot as plt
		#plt.plot(restframe_wavelength, flux, 'r')
		#plt.show()
		#stop

		if self.milky_way_reddening :
			# gets the amount of MW reddening on the models
			self.ebv_mw = get_dust_radec(ra,dec,'ebv')
		else:
			self.ebv_mw = 0.0
