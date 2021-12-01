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
	import os
	from firefly_dust import get_dust_radec	
	import astropy.cosmology as cc
	import astropy.units as uu
	import astropy.constants as const
"""

import numpy as np
import astropy.io.fits as pyfits
import os
from firefly_dust import get_dust_radec

import astropy.cosmology as cc
cosmo = cc.Planck15
import astropy.units as uu
import astropy.constants as const

class firefly_setup:
	"""
	Loads the environnement to transform observed spectra into the input for FIREFLY. 
	
	Currently SDSS spectra, speclite format is handled as well as input data from the MaNGA survey.
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

	def mask_emissionlines(self, element_emission_lines):

		"""
		Firefly needs to mask emission lines of elements as this can affect the fitting.
		"""
		#Dictionary of corrosponding elements to their emission lines
		emission_dict = {'He-II' : (3202.15, 4685.74),
						 'Ne-V'  : (3345.81, 3425.81),
						 'O-II'  : (3726.03, 3728.73),
						 'Ne-III': (3868.69, 3967.40),
						 'H-ζ'   : 3889.05,
						 'H-ε'   : 3970.07,
						 'H-δ'   : 4101.73,
						 'H-γ'   : 4340.46,
						 'O-III' : (4363.15, 4958.83, 5006.77),
						 'Ar-IV' : (4711.30, 4740.10),
						 'H-β'   : 4861.32,
						 'N-I'   : (5197.90, 5200.39),
						 'He-I'  : 5875.60,
						 'O-I'   : (6300.20, 6363.67),
						 'N-II'  : (6547.96, 6583.34),
						 'H-α'   : 6562.80,
						 'S-II'  : (6716.31, 6730.68),
						 'Ar-III': 7135.67}

		#Create an array full of booleans equal to False, same size as the restframe_wavelength
		self.lines_mask = np.zeros_like(self.restframe_wavelength,dtype=bool)

		#Loop through the input of the emission lines list
		for i in range(len(element_emission_lines)):

			#Check if the value is in the dictionary
			if element_emission_lines[i] in emission_dict:

				ele_line = element_emission_lines[i]
				line = emission_dict[ele_line]

				#Check if it contains a tuple (some elements have more then one emission line)
				if type(line) == tuple:

					#Find the number of emission lines for this value
					n_lines = len(line)

					#Loop through and mask them
					for n in range(n_lines):

						n_line = line[n]

						#Creates the boolean array
						temp_lines_mask = ((self.restframe_wavelength > n_line - self.N_angstrom_masked) & (self.restframe_wavelength < n_line + self.N_angstrom_masked))
						#Adds the boolean array to the exisiting one to save it
						self.lines_mask = (temp_lines_mask | self.lines_mask)
						
				else:
					temp_lines_mask = ((self.restframe_wavelength > line - self.N_angstrom_masked) & (self.restframe_wavelength < line + self.N_angstrom_masked))
					self.lines_mask = (temp_lines_mask | self.lines_mask)

			else:
				print(element_emission_lines[i])
				raise KeyError
	
	def openSingleSpectrum(self, wavelength, flux, error, redshift, ra, dec, vdisp, emlines, r_instrument):
        
		assert len(wavelength)==len(flux)==len(error)==len(r_instrument),\
			"The arrays wavelength, flux, error, and r_instrument must have identical lengths."
		
		self.wavelength=wavelength
		self.flux=flux
		self.error=error
		self.redshift=redshift
		self.ra=ra
		self.dec=dec
		self.vdisp=vdisp
		self.r_instrument=r_instrument
		
		self.DL = cosmo.luminosity_distance(self.redshift).to(uu.cm)
		self.bad_flags = np.ones(len(self.wavelength))
		self.restframe_wavelength = self.wavelength/(1+self.redshift)
		self.trust_flag = 1
		self.objid = 0
		
		# removes the bad data from the spectrum 
		self.bad_data = np.isnan(self.flux) | np.isinf(self.flux) | (self.flux <= 0.0) | np.isnan(self.error) | np.isinf(self.error)
		self.mask_emissionlines(emlines)
		self.final_mask = (self.bad_data | self.lines_mask)

		self.bad_flags = self.bad_flags[(self.final_mask==False)]
		self.restframe_wavelength = self.restframe_wavelength[(self.final_mask==False)] 
		self.wavelength = self.wavelength[(self.final_mask==False)]
		self.flux = self.flux[(self.final_mask==False)]
		self.error = self.error[(self.final_mask==False)]
		self.r_instrument = self.r_instrument[(self.final_mask==False)]
		        
		if self.milky_way_reddening:
		# gets the amount of MW reddening on the models
			self.ebv_mw = get_dust_radec(self.ra,self.dec,'ebv')
		else:
			self.ebv_mw = 0.0
		#print(self.ebv_mw)	


	def openMANGASpectrum(self, path_to_logcube, path_to_dapall, bin_number, plate_number, ifu_number, emlines,mpl='mpl-9'):
		"""Loads an observed MaNGA spectrum in.
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
		velocity_dispersion_correction = maps_header['STELLAR_SIGMACORR'].data[0,:,:]
		
		if velocity_dispersion[x_position,y_position] > velocity_dispersion_correction[x_position,y_position]:
			correction = np.sqrt((velocity_dispersion[x_position,y_position])**2-(velocity_dispersion_correction[x_position,y_position])**2)
			vdisp = correction
		else:
			vdisp = 0

		
		# Open LOGCUBE to get the flux, wavelength, and error
		header = pyfits.open(path_to_logcube)
		wavelength, flux, emline, bit_mask, inverse_variance = header['WAVE'].data, header['FLUX'].data, header['EMLINE'].data, header['MASK'].data, header['IVAR'].data
		self.wavelength = wavelength
		correct_flux = flux[:,x_position,y_position]
		correct_flux_emline = emline[:, x_position, y_position]
		output_flux = correct_flux - correct_flux_emline
		correct_inverse_variance = inverse_variance[:, x_position, y_position]
		
		LSF = header['LSF'].data[:,x_position,y_position]		# LSF given as sigma of Gaussian in Angstrom
		sig2fwhm        = 2.0 * np.sqrt(2.0 * np.log(2.0))
		LSF_FWHM = LSF*sig2fwhm
		RES = wavelength/LSF_FWHM
		
		self.r_instrument = RES
		self.error = np.sqrt(1.0/(correct_inverse_variance))
		self.bad_flags = np.ones(len(output_flux))
		self.flux = output_flux
		self.vdisp = vdisp

		if (mpl=='mpl-10') or (mpl=='mpl-11'):
			ext=2
		else:
			ext=1
		
		dap_all = pyfits.open(path_to_dapall)
		get = np.where(dap_all[ext].data['PLATEIFU']==str(plate_number)+'-'+str(ifu_number))
		c = const.c.value/1000
		# Use redshift as measured from the stellar kinematics by the DAP.
		redshift = dap_all[ext].data['STELLAR_Z'][get][0]
		# If redshift measurement failed, use redshift estimate from NSA or ancillary programs.
		if redshift<0:
			redshift = dap_all[ext].data['Z'][get][0]
			
		sys_vel = maps_header[0].header['SCINPVEL']
		bin_vel = maps_header['STELLAR_VEL'].data[x_position,y_position]	
			
		if redshift<0:
			print('WARNING: The redshift of this object is negative.')
			print('z = {}'.format(redshift))
		
		redshift_corr = (sys_vel+bin_vel)/c
		self.redshift = redshift
		self.restframe_wavelength = self.wavelength / (1.0+redshift_corr)

		bitmask = bit_mask[:,x_position,y_position]&2**0+2**1+2**2+2**3+2**4
		self.mask_emissionlines(emlines)
		self.final_mask = (bitmask | self.lines_mask)

		self.wavelength = self.wavelength[(self.final_mask==False)] 
		self.restframe_wavelength = self.restframe_wavelength[(self.final_mask==False)] 
		self.flux = self.flux[(self.final_mask==False)] 
		self.error = self.error[(self.final_mask==False)]
		self.bad_flags = self.bad_flags[(self.final_mask==False)]
					
		# Get Trust flag, object_id, xpos, ypos and instrumental resolution.
# 		self.trust_flag, self.objid, self.r_instrument = True, 0, np.loadtxt(os.path.join(os.environ['FF_DIR'],'data/MaNGA_spectral_resolution.txt'))
		self.trust_flag, self.objid= True, 0
# 		self.r_instrument = self.r_instrument[0:self.r_instrument.shape[0]//2]
		self.r_instrument = self.r_instrument[(self.final_mask==False)]
		self.xpos, self.ypos = ra, dec
		
		# gets the amount of MW reddening on the models
		if self.milky_way_reddening :
			self.ebv_mw = get_dust_radec(ra, dec, 'ebv')
		else:
			self.ebv_mw = 0.0
