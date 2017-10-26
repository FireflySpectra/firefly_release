"""
.. moduleauthor:: Johan Comparat <johan.comparat__at__gmail.com>
.. contributor :: Sofia Meneses-Goytia <s.menesesgoytia__at__gmail.com>
.. minor modifications :: Violeta Gonzalez-Perez <violegp__at__gmail.com>

General purpose:
................

The class StellarPopulationModel is a wrapper dedicated to handling the fit of stellar population models on observed spectra.
It gathers all inputs : from the model and from the data.

*Imports*::

	import numpy as np
	import astropy.io.fits as pyfits
	import astropy.units as u
	import glob
	import pandas as pd
	import os
	from firefly_instrument import *
	from firefly_dust import *
	from firefly_fitter import *
	from firefly_library import *

"""
import numpy as np
import astropy.io.fits as pyfits
import astropy.units as u
import glob
import pandas as pd
import os,sys
from os.path import join
import copy
#from scipy.stats import sigmaclip
from estimations_3d import estimation
#from firefly_dust import *
#import firefly_dust as f_dust
from firefly_dust import hpf, unred, determine_attenuation
from firefly_instrument import downgrade
from firefly_fitter import fitter
from firefly_library import airtovac, convert_chis_to_probs, light_weights_to_mass, calculate_averages_pdf, normalise_spec, match_data_models
import matplotlib.pyplot as plt

default_value = -9999
EPS = 10.E-10
dict_imfs = {'cha': 'Chabrier', 'ss': 'Salpeter', 'kr': 'Kroupa'}

def trylog10(value):
	if (value<EPS):
		logv = default_value
	else:
		logv = np.log10(value)
	return logv


class StellarPopulationModel:
	"""
	:param specObs: specObs observed spectrum object initiated with the  GalaxySpectrumFIREFLY class.
	:param models: choose between 'm11', 'bc03' or 'm09'.

		* m11 corresponds to all the models compared in `Maraston and Stromback 2011  <http://adsabs.harvard.edu/abs/2011MNRAS.418.2785M>`_.
		* m09 to `Maraston et al. 2009 <http://adsabs.harvard.edu/abs/2009A%26A...493..425M>`_.
		* bc03 to the `Bruzual and Charlot 2003 models <http://adsabs.harvard.edu/abs/2003MNRAS.344.1000B>`_.

	:param model_libs: only necessary if using m11.
	Choose between `MILES <http://adsabs.harvard.edu/abs/2011A%26A...532A..95F>`_, MILES revisednearIRslope, MILES UVextended, `STELIB <http://adsabs.harvard.edu/abs/2003A%26A...402..433L>`_, `ELODIE <http://adsabs.harvard.edu/abs/2007astro.ph..3658P>`_, `MARCS <http://adsabs.harvard.edu/abs/2008A%26A...486..951G>`_.

		* MILES, MILES revisednearIRslope, MILES UVextended, STELIB, ELODIE are empirical libraries.
		* MARCS is a theoretical library.

	:param imfs: choose the `initial mass function <https://en.wikipedia.org/wiki/Initial_mass_function>`_:

		* 'ss' for `Salpeter <http://adsabs.harvard.edu/abs/1955ApJ...121..161S>`_or
		* 'kr' for `Kroupa <http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1112.3340>`_ or
		* 'cha' for `Chabrier <http://adsabs.harvard.edu/abs/2003PASP..115..763C>`_.

	:param hpf_mode: 'on' means the code uses HPF to dereden the spectrum, if 'hpf_only' then EBV=0.

	 Notes
	 -----

	.. note::
		*This is how it proceeds :*
		 #. reads the parameter file by using parameters_obtain(parameters.py)
		 #. It opens the data file, model files, then it matches their resolutions by downgrading the models to instrumental and velocity dispersion resolution
		 #. Determines dust attenuation curve to be applied to the models. Two options : through HPF fitting (3.1.) or through filtered values to determing SP properties (3.2.).
		 #. It fits the models to the data
		 #. Gets mass-weighted SSP contributions using saved M/L ratio.
		 #. Convert chis into probabilities and calculates all average properties and errors (assuming the number of degrees of freedom is the number of wavelength points)
		 #. Optionally produces a plot
		 #. Finally, it writes the output files



	"""
	def __init__(self, specObs, outputFile, cosmo, models = 'm11', model_libs = ['MILES_UVextended'], imfs = ['ss','kr'], hpf_mode = 'on', age_limits = [6,10.1], downgrade_models = True, dust_law = 'calzetti', max_ebv = 1.5, num_dust_vals = 200, dust_smoothing_length = 200, max_iterations = 10, fit_per_iteration_cap = 1000, pdf_sampling = 300, data_wave_medium = 'vacuum', Z_limits = [-0.1,0.1], wave_limits = [0,99999990], suffix = "",use_downgraded_models = False, write_results=True, flux_units=10**-17):
		self.cosmo = cosmo
		self.specObs = specObs
		self.outputFile = outputFile
		#################### STARTS HERE ####################
		# sets the models
		self.models = models # m11/bc03 / m09
		self.model_libs = model_libs
		self.suffix = suffix
		self.deltal_libs = []
		self.vdisp_round = int(round(self.specObs.vdisp/5.0)*5.0) # rounding vDisp for the models
		self.use_downgraded_models = use_downgraded_models
		self.write_results = write_results
		self.flux_units = flux_units
		if self.models == 'm11':
			for m in self.model_libs:
				if m == 'MILES' or m == 'MILES_revisednearIRslope' or m == 'MILES_UVextended':
					self.deltal_libs.append(2.55)
				elif m == 'STELIB':
					self.deltal_libs.append(3.40)
				elif m == 'ELODIE':
					self.deltal_libs.append(0.55)
				elif m == 'MARCS':
					self.deltal_libs.append(0.1)

		elif self.models=='bc03':
			self.model_libs = ['STELIB_BC03']
			imfs        = ['cha']
			self.deltal_libs = [3.00]

		elif self.models == 'm09':
			self.model_libs = ['M09']
			if downgrade_models:
				self.deltal_libs = [0.4]
			else:
				self.deltal_libs = [3.6]
		# sets the Initial mass function
		self.imfs = imfs
		self.hpf_mode = hpf_mode
		self.age_limits = age_limits

		self.downgrade_models = downgrade_models
		self.dust_law = dust_law
		self.max_ebv = max_ebv
		self.num_dust_vals = num_dust_vals
		self.dust_smoothing_length = dust_smoothing_length
		# Specific fitting options
		self.max_iterations = max_iterations
		self.fit_per_iteration_cap = fit_per_iteration_cap
		# Sampling size when calculating the maximum pdf (100=recommended)
		self.pdf_sampling = pdf_sampling
		# Default is air, unless manga is used
		self.data_wave_medium = data_wave_medium
		self.Z_limits = Z_limits
		self.wave_limits = wave_limits

	def get_model(self, model_used, imf_used, deltal, vdisp, wave_instrument, r_instrument, ebv_mw):
		"""
		Retrieves all relevant model files, in their downgraded format.
		If they aren't downgraded to the correct resolution / velocity dispersion,
		takes the base models in their native form and converts to downgraded files.

		:param model_used: list of models to be used, for example ['m11', 'm09'].
		:param imf_used: list of imf to be used, for example ['ss', 'cha'].
		:param deltal: delta lambda in the models.
		:param vdisp: velocity dispersion observed in the galaxy.
		:param wave_instrument: wavelength array from the observations
		:param r_instrument: resolution array from the observations
		:param  ebv_mw: E(B-V) from the dust maps for the galaxy.

		Workflow
		----------
			A. loads the models m11 or m09: maps parameters to the right files. Then it constructs the model array. Finally converts wavelengths to air or vacuum.
			B. downgrades the model to match data resolution
			C. applies attenuation
			D. stores models in
				self.model_wavelength,
				self.model_flux,
				self.age_model,
				self.metal_model

			and returns it as well

		"""
		# first the m11 case
		if self.models == 'm11':
			first_file  = True
			model_files = []
			#print('yes we are in here')
			#stop
			if self.use_downgraded_models :
				if model_used == 'MILES_UVextended' or model_used == 'MILES_revisedIRslope':
					model_path 		= join(os.environ['STELLARPOPMODELS_DIR'],'data','SSP_M11_MILES_downgraded','ssp_M11_' + model_used+ '.' + imf_used)
				else:
					model_path 		= join(os.environ['STELLARPOPMODELS_DIR'],'data','SSP_M11_'+ model_used + '_downgraded', 'ssp_M11_' +model_used +'.' + imf_used)
			else:
				if model_used == 'MILES_UVextended' or model_used == 'MILES_revisedIRslope':
					model_path 		= join(os.environ['STELLARPOPMODELS_DIR'],'data','SSP_M11_MILES', 'ssp_M11_'+model_used+'.'+imf_used)
				else:
					model_path 		= join(os.environ['STELLARPOPMODELS_DIR'],'data','SSP_M11_'+model_used ,'ssp_M11_' +model_used +'.' + imf_used)


			# Constructs the metallicity array of models :
			all_metal_files = glob.glob(model_path+'*')
			#print(model_path)
			#print(all_metal_files)
			#stop	
			## # print all_metal_files
			metal_files 	= []
			metal 	    = []
			for z in range(len(all_metal_files)):
				zchar = all_metal_files[z][len(model_path):]
				if zchar == 'z001':
					#znum = -0.3
					znum = 0.5
				elif zchar == 'z002':
					#znum = 0.0
					znum = 1.0
				elif zchar == 'z004':
					#znum = 0.3
					znum = 2.0
				elif zchar == 'z0001.bhb':
					#znum = -1.301
					znum = 10**-1.301
				elif zchar == 'z0001.rhb':
					#znum = -1.302
					znum = 10**-1.302
				elif zchar == 'z10m4.bhb':
					#znum = -2.301
					znum = 10**-2.301
				elif zchar == 'z10m4.rhb':
					#znum = -2.302
					znum = 10**-2.302
				elif zchar == 'z10m4':
					#znum = -2.300
					znum = 10**-2.300
				else:
					raise NameError('Unrecognised metallicity! Check model file names.')

				if znum>self.Z_limits[0] and znum<self.Z_limits[1]:
					metal_files.append(all_metal_files[z])
					metal.append(znum)
			#print(metal_files)
			#stop
			# constructs the model array
			model_flux, age_model, metal_model = [],[],[]
			for zi,z in enumerate(metal_files):
				# print "Retrieving and downgrading models for "+z
				model_table = pd.read_table(z,converters={'Age':np.float64}, header=None ,usecols=[0,2,3], names=['Age','wavelength_model','flux_model'], delim_whitespace=True)
				age_data = np.unique(model_table['Age'].values.ravel())
#				print(age_data)
#				stop
				for a in age_data:
					logyrs_a = trylog10(a)+9.0
					## print "age model selection:", self.age_limits[0], logyrs_a, self.age_limits[1]
					if (((10**(logyrs_a-9)) < self.age_limits[0]) or ((10**(logyrs_a-9)) > self.age_limits[1])):
						continue
					else:
						spectrum = model_table.loc[model_table.Age == a, ['wavelength_model', 'flux_model'] ].values
						wavelength_int,flux = spectrum[:,0],spectrum[:,1]

						# converts to air wavelength
						if self.data_wave_medium == 'vacuum':
							wavelength = airtovac(wavelength_int)
						else:
							wavelength = wavelength_int

						# downgrades the model
						if self.downgrade_models:
							mf = downgrade(wavelength,flux,deltal,self.vdisp_round, wave_instrument, r_instrument)
						else:
							mf = copy.copy(flux)

						# Reddens the models
						if ebv_mw != 0:
							attenuations = unred(wavelength,ebv=0.0-ebv_mw)
							model_flux.append(mf*attenuations)
						else:
							model_flux.append(mf)

						age_model.append(a)
						metal_model.append(metal[zi])
						first_model = False
			#print(wavelength)
			#stop

			# print "Retrieved all models!"
			self.model_wavelength, self.model_flux, self.age_model, self.metal_model = wavelength, model_flux, age_model, metal_model
			return wavelength, model_flux, age_model, metal_model

		elif self.models == 'm09':
			first_file  = True
			model_files = []
			if self.use_downgraded_models:
				model_path = join(os.environ['STELLARPOPMODELS_DIR'],'data','UVmodels_Marastonetal08b_downgraded')
			else:
				model_path = join(os.environ['STELLARPOPMODELS_DIR'],'data','UVmodels_Marastonetal08b')
			# Gathers the list of models with metallicities and ages of interest:
			all_metal_files = glob.glob(model_path+'*')
			metal_files 	= []
			metal 			= []
			for z in range(len(all_metal_files)):
				zchar = all_metal_files[z].split('.')[1][2:]
				if zchar == 'z001':
					#znum = -0.3
					znum = 10**-0.3
				elif zchar == 'z002':
					#znum = 0.0
					znum = 1.0
				elif zchar == 'z004':
					#znum = 0.3
					znum = 10**0.3
				elif zchar == 'z0001':
					#znum = -1.300
					znum = 10**-1.300
				else:
					raise NameError('Unrecognised metallicity! Check model file names.')

				if znum>self.Z_limits[0] and znum<self.Z_limits[1]:
					metal_files.append(all_metal_files[z])
					metal.append(znum)

			# constructs the model array
			model_flux, age_model, metal_model = [],[],[]
			for zi,z in enumerate(metal_files):
				# print "Retrieving and downgrading models for "+z
				model_table = pd.read_table(z,converters={'Age':np.float64}, header=None ,usecols=[0,2,3], names=['Age','wavelength_model','flux_model'], delim_whitespace=True)
				age_data = np.unique(model_table['Age'].values.ravel())
				for a in age_data:
					logyrs_a = trylog10(a)+9.0
					## print "age model selection:", self.age_limits[0], logyrs_a, self.age_limits[1]
					if (((10**(logyrs_a-9)) < self.age_limits[0]) or ((10**(logyrs_a-9)) > self.age_limits[1])):
						continue
					else:
						spectrum = model_table.loc[model_table.Age == a, ['wavelength_model', 'flux_model'] ].values
						wavelength_int,flux = spectrum[:,0],spectrum[:,1]

						# converts to air wavelength
						if self.data_wave_medium == 'vacuum':
							wavelength = airtovac(wavelength_int)
						else:
							wavelength = wavelength_int

						# downgrades the model
						if self.downgrade_models:
							mf = downgrade(wavelength,flux,deltal,self.vdisp_round, wave_instrument, r_instrument)
						else:
							mf = copy.copy(flux)

						# Reddens the models
						if ebv_mw != 0:
							attenuations = unred(wavelength,ebv=0.0-ebv_mw)
							model_flux.append(mf*attenuations)
						else:
							model_flux.append(mf)

						age_model.append(a)
						metal_model.append(metal[zi])
						first_model = False

			# print "Retrieved all models!"
			self.model_wavelength, self.model_flux, self.age_model, self.metal_model = wavelength, model_flux, age_model, metal_model
			return wavelength, model_flux, age_model, metal_model

	def fit_models_to_data(self):
		"""
		Once the data and models are loaded, then execute this function to find the best model. It loops overs the models to be fitted on the data:
		 #. gets the models
		 #. matches the model and data to the same resolution
		 #. normalises the spectra
		"""
		for mi,mm in enumerate(self.model_libs):
			# loop over the models
			for ii in self.imfs:
				# loop over the IMFs
				# A. gets the models
				# print "getting the models"
				deltal = self.deltal_libs[mi]
				model_wave_int, model_flux_int, age, metal = self.get_model( mm, ii, deltal, self.specObs.vdisp, self.specObs.restframe_wavelength, self.specObs.r_instrument, self.specObs.ebv_mw)
				# B. matches the model and data to the same resolution
				# print "Matching models to data"
				wave, data_flux, error_flux, model_flux_raw = match_data_models( self.specObs.restframe_wavelength, self.specObs.flux, self.specObs.bad_flags, self.specObs.error, model_wave_int, model_flux_int, self.wave_limits[0], self.wave_limits[1], saveDowngradedModel = False)
				# C. normalises the models to the median value of the data
				# print "Normalising the models"
				model_flux, mass_factors = normalise_spec(data_flux, model_flux_raw)

			# 3. Corrects from dust attenuation
			if self.hpf_mode=='on':
				# 3.1. Determining attenuation curve through HPF fitting, apply attenuation curve to models and renormalise spectra
				best_ebv, attenuation_curve = determine_attenuation(wave, data_flux, error_flux, model_flux, self, age, metal)
				model_flux_atten = np.zeros(np.shape(model_flux_raw))
				for m in range(len(model_flux_raw)):
					model_flux_atten[m] = attenuation_curve * model_flux_raw[m]

				model_flux, mass_factors = normalise_spec(data_flux, model_flux_atten)
				# 4. Fits the models to the data
				light_weights, chis, branch = fitter(wave, data_flux, error_flux, model_flux, self)

			elif self.hpf_mode == 'hpf_only':

				# 3.2. Uses filtered values to determing SP properties only."
				smoothing_length = self.dust_smoothing_length
				hpf_data    = hpf(data_flux)
				hpf_models  = np.zeros(np.shape(model_flux))
				for m in range(len(model_flux)):
					hpf_models[m] = hpf(model_flux[m])

				zero_dat = np.where( (np.isnan(hpf_data)) & (np.isinf(hpf_data)) )
				hpf_data[zero_dat] = 0.0
				for m in range(len(model_flux)):
					hpf_models[m,zero_dat] = 0.0
				hpf_error    = np.zeros(len(error_flux))
				hpf_error[:] = np.median(error_flux)/np.median(data_flux) * np.median(hpf_data)
				hpf_error[zero_dat] = np.max(hpf_error)*999999.9

				best_ebv = 0.0
				hpf_models,mass_factors = normalise_spec(hpf_data,hpf_models)
				# 4. Fits the models to the data
				light_weights, chis, branch = fitter(wave, hpf_data,hpf_error, hpf_models, self)

			# 5. Get mass-weighted SSP contributions using saved M/L ratio.
			unnorm_mass, mass_weights = light_weights_to_mass(light_weights, mass_factors)
			# print "Fitting complete"
			if np.all(np.isnan(mass_weights)):
				tbhdu = self.create_dummy_hdu()
			else:			
				# print "Calculating average properties and outputting"
				# 6. Convert chis into probabilities and calculates all average properties and errors
				self.dof = len(wave)
        			probs = convert_chis_to_probs(chis, self.dof)
        			dist_lum	= self.cosmo.luminosity_distance( self.specObs.redshift).to( u.cm ).value
        			
        			#print(light_weights)
        			#print(np.shape(light_weights))	
        			#stop
        			averages = calculate_averages_pdf(probs, light_weights, mass_weights, unnorm_mass, age, metal, self.pdf_sampling, dist_lum, self.flux_units)
        
        			unique_ages 				= np.unique(age)
        			marginalised_age_weights 	= np.zeros(np.shape(unique_ages))
        			marginalised_age_weights_int = np.sum(mass_weights.T,1)
        			for ua in range(len(unique_ages)):
        				marginalised_age_weights[ua] = np.sum(marginalised_age_weights_int[np.where(age==unique_ages[ua])])
        
        			best_fit_index = [np.argmin(chis)]
        			best_fit = np.dot(light_weights[best_fit_index],model_flux)[0]
        			# stores outputs in the object
        			self.best_fit_index = best_fit_index
        			self.best_fit = best_fit
        			self.model_flux = model_flux
        			self.dist_lum = dist_lum
        			self.age = np.array(age)
        			self.metal = np.array(metal)
        			self.mass_weights = mass_weights
        			self.light_weights = light_weights
        			self.chis = chis
        			self.branch = branch
        			self.unnorm_mass = unnorm_mass
        			self.probs = probs
        			self.best_fit = best_fit
        			self.averages = averages
				self.wave = wave
        
        			bf_mass = (self.mass_weights[self.best_fit_index]>0)[0]
        			bf_light = (self.light_weights[self.best_fit_index]>0)[0]
        			mass_per_ssp = self.unnorm_mass[self.best_fit_index[0]][bf_mass]*self.flux_units* 4 * np.pi * self.dist_lum**2.0
        
                                age_per_ssp = self.age[bf_mass]
                                metal_per_ssp = self.metal[bf_mass]
                                weight_mass_per_ssp = self.mass_weights[self.best_fit_index[0]][bf_mass]
                                weight_light_per_ssp = self.light_weights[self.best_fit_index[0]][bf_light]
                                order = np.argsort(-weight_light_per_ssp)
        
        			# Do we want to put this all in another function??
        			# We could provide it with the arrays and call something like get_massloss_parameters()?
        			# I think it looks a little untidy still because of my bad coding.
        	
        			# Gets the mass loss factors.
        			if dict_imfs[self.imfs[0]] == 'Salpeter':
        				ML_metallicity, ML_age, ML_totM, ML_alive, ML_wd, ML_ns, ML_bh, ML_turnoff = np.loadtxt(join(os.environ['STELLARPOPMODELS_DIR'],'data','massloss_salpeter.txt'), unpack=True, skiprows=2)
                                        # First build the grids of the quantities. Make sure they are in linear units.                  
                                        estimate_ML_totM, estimate_ML_alive, estimate_ML_wd = estimation(10**ML_metallicity, ML_age, ML_totM), estimation(10**ML_metallicity, ML_age, ML_alive), estimation(10**ML_metallicity, ML_age, ML_wd)
                                        estimate_ML_ns, estimate_ML_bh, estimate_ML_turnoff = estimation(10**ML_metallicity, ML_age, ML_ns), estimation(10**ML_metallicity, ML_age, ML_bh), estimation(10**ML_metallicity, ML_age, ML_turnoff)
                                        # Now loop through SSPs to find the nearest values for each.
                                        final_ML_totM, final_ML_alive, final_ML_wd, final_ML_ns, final_ML_bh, final_ML_turnoff, final_gas_fraction = [], [], [], [], [], [], []
                                        for number in range(len(age_per_ssp)):
                                                new_ML_totM = estimate_ML_totM.estimate(metal_per_ssp[number],age_per_ssp[number])
                                                new_ML_alive = estimate_ML_alive.estimate(metal_per_ssp[number],age_per_ssp[number])
                                                new_ML_wd = estimate_ML_wd.estimate(metal_per_ssp[number],age_per_ssp[number])
                                                new_ML_ns = estimate_ML_ns.estimate(metal_per_ssp[number],age_per_ssp[number])
                                                new_ML_bh = estimate_ML_bh.estimate(metal_per_ssp[number],age_per_ssp[number])
                                                new_ML_turnoff = estimate_ML_turnoff.estimate(metal_per_ssp[number],age_per_ssp[number])
                                                final_ML_totM.append(mass_per_ssp[number]*new_ML_totM)
                                                final_ML_alive.append(mass_per_ssp[number]*new_ML_alive)
                                                final_ML_wd.append(mass_per_ssp[number]*new_ML_wd)
                                                final_ML_ns.append(mass_per_ssp[number]*new_ML_ns)
                                                final_ML_bh.append(mass_per_ssp[number]*new_ML_bh)
                                                final_ML_turnoff.append(mass_per_ssp[number]*new_ML_turnoff)
                                                final_gas_fraction.append(mass_per_ssp[number]-new_ML_totM)
                                        final_ML_totM, final_ML_alive, final_ML_wd, final_ML_ns, final_ML_bh, final_ML_turnoff, final_gas_fraction= np.array(final_ML_totM), np.array(final_ML_alive), np.array(final_ML_wd), np.array(final_ML_ns), np.array(final_ML_bh), np.array(final_ML_turnoff), np.array(final_gas_fraction)

        			if (dict_imfs[self.imfs[0]] == 'Chabrier'):
        				ML_metallicity, ML_age, ML_totM, ML_alive, ML_wd, ML_ns, ML_bh, ML_turnoff = np.loadtxt(join(os.environ['STELLARPOPMODELS_DIR'],'data', 'massloss_chabrier.txt'), unpack=True, skiprows=2)
        				# First build the grids of the quantities. Make sure they are in linear units.			
        				estimate_ML_totM, estimate_ML_alive, estimate_ML_wd = estimation(10**ML_metallicity, ML_age, ML_totM), estimation(10**ML_metallicity, ML_age, ML_alive), estimation(10**ML_metallicity, ML_age, ML_wd)
        				estimate_ML_ns, estimate_ML_bh, estimate_ML_turnoff = estimation(10**ML_metallicity, ML_age, ML_ns), estimation(10**ML_metallicity, ML_age, ML_bh), estimation(10**ML_metallicity, ML_age, ML_turnoff)
        				# Now loop through SSPs to find the nearest values for each.
        				final_ML_totM, final_ML_alive, final_ML_wd, final_ML_ns, final_ML_bh, final_ML_turnoff, final_gas_fraction = [], [], [], [], [], [], []
        				for number in range(len(age_per_ssp)):
        					new_ML_totM = estimate_ML_totM.estimate(metal_per_ssp[number],age_per_ssp[number])
        					new_ML_alive = estimate_ML_alive.estimate(metal_per_ssp[number],age_per_ssp[number])
        					new_ML_wd = estimate_ML_wd.estimate(metal_per_ssp[number],age_per_ssp[number])
        					new_ML_ns = estimate_ML_ns.estimate(metal_per_ssp[number],age_per_ssp[number])
        					new_ML_bh = estimate_ML_bh.estimate(metal_per_ssp[number],age_per_ssp[number])
        					new_ML_turnoff = estimate_ML_turnoff.estimate(metal_per_ssp[number],age_per_ssp[number])
        					final_ML_totM.append(mass_per_ssp[number]*new_ML_totM)
        					final_ML_alive.append(mass_per_ssp[number]*new_ML_alive)
        					final_ML_wd.append(mass_per_ssp[number]*new_ML_wd)
        					final_ML_ns.append(mass_per_ssp[number]*new_ML_ns)
        					final_ML_bh.append(mass_per_ssp[number]*new_ML_bh)
        					final_ML_turnoff.append(mass_per_ssp[number]*new_ML_turnoff)
        					final_gas_fraction.append(mass_per_ssp[number]-new_ML_totM)
        				final_ML_totM, final_ML_alive, final_ML_wd, final_ML_ns, final_ML_bh, final_ML_turnoff, final_gas_fraction= np.array(final_ML_totM), np.array(final_ML_alive), np.array(final_ML_wd), np.array(final_ML_ns), np.array(final_ML_bh), np.array(final_ML_turnoff), np.array(final_gas_fraction)
					
        			if (dict_imfs[self.imfs[0]] == 'Kroupa'):
        				ML_metallicity, ML_age, ML_totM, ML_alive, ML_wd, ML_ns, ML_bh, ML_turnoff = np.loadtxt(join(os.environ['STELLARPOPMODELS_DIR'],'data','massloss_kroupa.txt'), unpack=True, skiprows=2)
        				# First build the grids of the quantities. Make sure they are in linear units.			
        				estimate_ML_totM, estimate_ML_alive, estimate_ML_wd = estimation(10**ML_metallicity, ML_age, ML_totM), estimation(10**ML_metallicity, ML_age, ML_alive), estimation(10**ML_metallicity, ML_age, ML_wd)
        				estimate_ML_ns, estimate_ML_bh, estimate_ML_turnoff = estimation(10**ML_metallicity, ML_age, ML_ns), estimation(10**ML_metallicity, ML_age, ML_bh), estimation(10**ML_metallicity, ML_age, ML_turnoff)
        				# Now loop through SSPs to find the nearest values for each.
        				final_ML_totM, final_ML_alive, final_ML_wd, final_ML_ns, final_ML_bh, final_ML_turnoff, final_gas_fraction = [], [], [], [], [], [], []
        				for number in range(len(age_per_ssp)):
        					new_ML_totM = estimate_ML_totM.estimate(metal_per_ssp[number],age_per_ssp[number])
        					new_ML_alive = estimate_ML_alive.estimate(metal_per_ssp[number],age_per_ssp[number])
        					new_ML_wd = estimate_ML_wd.estimate(metal_per_ssp[number],age_per_ssp[number])
        					new_ML_ns = estimate_ML_ns.estimate(metal_per_ssp[number],age_per_ssp[number])
        					new_ML_bh = estimate_ML_bh.estimate(metal_per_ssp[number],age_per_ssp[number])
        					new_ML_turnoff = estimate_ML_turnoff.estimate(metal_per_ssp[number],age_per_ssp[number])			
                                                final_ML_totM.append(mass_per_ssp[number]*new_ML_totM)
                                                final_ML_alive.append(mass_per_ssp[number]*new_ML_alive)
                                                final_ML_wd.append(mass_per_ssp[number]*new_ML_wd)
                                                final_ML_ns.append(mass_per_ssp[number]*new_ML_ns)
                                                final_ML_bh.append(mass_per_ssp[number]*new_ML_bh)
                                                final_ML_turnoff.append(mass_per_ssp[number]*new_ML_turnoff)
                                                final_gas_fraction.append(mass_per_ssp[number]-new_ML_totM)
        				final_ML_totM, final_ML_alive, final_ML_wd, final_ML_ns, final_ML_bh, final_ML_turnoff, final_gas_fraction= np.array(final_ML_totM), np.array(final_ML_alive), np.array(final_ML_wd), np.array(final_ML_ns), np.array(final_ML_bh), np.array(final_ML_turnoff), np.array(final_gas_fraction)
        			
        
        			# Calculate the total mass loss from all the SSP contributions.
        			combined_ML_totM = np.sum(final_ML_totM)
        			combined_ML_alive = np.sum(final_ML_alive)
        			combined_ML_wd = np.sum(final_ML_wd)
        			combined_ML_ns = np.sum(final_ML_ns)
        			combined_ML_bh = np.sum(final_ML_bh)		
        			combined_gas_fraction = np.sum(mass_per_ssp - final_ML_totM)
        
        			# 8. It writes the output file
        			waveCol = pyfits.Column(name="wavelength",format="D", unit="Angstrom", array= wave)
        			dataCol = pyfits.Column(name="original_data",format="D", unit="1e-17erg/s/cm2/Angstrom", array= data_flux)
        			errorCol = pyfits.Column(name="flux_error",format="D", unit="1e-17erg/s/cm2/Angstrom", array= error_flux)
        			best_fitCol = pyfits.Column(name="firefly_model",format="D", unit="1e-17erg/s/cm2/Angstrom", array= best_fit)
        			cols = pyfits.ColDefs([ waveCol, dataCol, errorCol, best_fitCol])
        			tbhdu = pyfits.BinTableHDU.from_columns(cols)
        			#tbhdu.header['HIERARCH age_universe (Gyr)'] = trylog10(self.cosmo.age(self.specObs.redshift).value*10**9)
        			tbhdu.header['HIERARCH redshift'] = self.specObs.redshift
        			tbhdu.header['HIERARCH Age_unit'] = 'log (age/Gyr)'
        			tbhdu.header['HIERARCH Metallicity_unit'] = '[Z/H]'
        			tbhdu.header['HIERARCH Mass_unit'] = 'log (M/Msun)'
        			tbhdu.header['HIERARCH SSP_sfr'] = 'log (M*/Age(Gyr))'			
        			tbhdu.header['IMF'] = dict_imfs[self.imfs[0]]
        			tbhdu.header['Model'] = self.model_libs[0]
        			tbhdu.header['HIERARCH converged'] = 'True'
        			tbhdu.header['HIERARCH age_lightW'] = trylog10(averages['light_age'])
        			tbhdu.header['HIERARCH age_lightW_up_1sig'] = trylog10(averages['light_age_1_sig_plus'])
        			tbhdu.header['HIERARCH age_lightW_low_1sig'] = trylog10(averages['light_age_1_sig_minus'])
                                tbhdu.header['HIERARCH age_lightW_up_2sig'] = trylog10(averages['light_age_2_sig_plus'])
                                tbhdu.header['HIERARCH age_lightW_low_2sig'] = trylog10(averages['light_age_2_sig_minus'])
                                tbhdu.header['HIERARCH age_lightW_up_3sig'] = trylog10(averages['light_age_3_sig_plus'])
                                tbhdu.header['HIERARCH age_lightW_low_3sig'] = trylog10(averages['light_age_3_sig_minus'])
        			tbhdu.header['HIERARCH metallicity_lightW'] = trylog10(averages['light_metal'])
        			tbhdu.header['HIERARCH metallicity_lightW_up_1sig'] = trylog10(averages['light_metal_1_sig_plus'])
        			tbhdu.header['HIERARCH metallicity_lightW_low_1sig'] = trylog10(averages['light_metal_1_sig_minus'])
                                tbhdu.header['HIERARCH metallicity_lightW_up_2sig'] = trylog10(averages['light_metal_2_sig_plus'])
                                tbhdu.header['HIERARCH metallicity_lightW_low_2sig'] = trylog10(averages['light_metal_2_sig_minus'])
                                tbhdu.header['HIERARCH metallicity_lightW_up_3sig'] = trylog10(averages['light_metal_3_sig_plus'])
                                tbhdu.header['HIERARCH metallicity_lightW_low_3sig'] = trylog10(averages['light_metal_3_sig_minus'])
        			tbhdu.header['HIERARCH age_massW'] = trylog10(averages['mass_age'])
        			tbhdu.header['HIERARCH age_massW_up_1sig'] = trylog10(averages['mass_age_1_sig_plus'])
        			tbhdu.header['HIERARCH age_massW_low_1sig'] = trylog10(averages['mass_age_1_sig_minus'])
                                tbhdu.header['HIERARCH age_massW_up_2sig'] = trylog10(averages['mass_age_2_sig_plus'])
                                tbhdu.header['HIERARCH age_massW_low_2sig'] = trylog10(averages['mass_age_2_sig_minus'])
                                tbhdu.header['HIERARCH age_massW_up_3sig'] = trylog10(averages['mass_age_3_sig_plus'])
                                tbhdu.header['HIERARCH age_massW_low_3sig'] = trylog10(averages['mass_age_3_sig_minus'])
        			tbhdu.header['HIERARCH metallicity_massW'] = trylog10(averages['mass_metal'])
        			tbhdu.header['HIERARCH metallicity_massW_up_1sig'] = trylog10(averages['mass_metal_1_sig_plus'])
        			tbhdu.header['HIERARCH metallicity_massW_low_1sig'] = trylog10(averages['mass_metal_1_sig_minus'])
                                tbhdu.header['HIERARCH metallicity_massW_up_2sig'] = trylog10(averages['mass_metal_2_sig_plus'])
                                tbhdu.header['HIERARCH metallicity_massW_low_2sig'] = trylog10(averages['mass_metal_2_sig_minus'])
                                tbhdu.header['HIERARCH metallicity_massW_up_3sig'] = trylog10(averages['mass_metal_3_sig_plus'])
                                tbhdu.header['HIERARCH metallicity_massW_low_3sig'] = trylog10(averages['mass_metal_3_sig_minus'])
        			tbhdu.header['HIERARCH stellar_mass'] = trylog10(averages['stellar_mass'])
                                tbhdu.header['HIERARCH living_stars_mass'] = trylog10(combined_ML_totM)
                                tbhdu.header['HIERARCH remnant_mass'] = trylog10(combined_ML_alive)
                                tbhdu.header['HIERARCH remnant_mass_in_whitedwarfs'] = trylog10(combined_ML_wd)
                                tbhdu.header['HIERARCH remnant_mass_in_neutronstars'] = trylog10(combined_ML_ns)
                                tbhdu.header['HIERARCH remnant_mass_blackholes'] = trylog10(combined_ML_bh)
                                tbhdu.header['HIERARCH mass_of_ejecta'] = trylog10(combined_gas_fraction)
        			tbhdu.header['HIERARCH stellar_mass_up_1sig'] = trylog10(averages['stellar_mass_1_sig_plus'])
        			tbhdu.header['HIERARCH stellar_mass_low_1sig'] = trylog10(averages['stellar_mass_1_sig_minus'])
        			tbhdu.header['HIERARCH stellar_mass_up_2sig'] = trylog10(averages['stellar_mass_2_sig_plus'])
                                tbhdu.header['HIERARCH stellar_mass_low_2sig'] = trylog10(averages['stellar_mass_2_sig_minus'])
        			tbhdu.header['HIERARCH stellar_mass_up_3sig'] = trylog10(averages['stellar_mass_3_sig_plus'])
                                tbhdu.header['HIERARCH stellar_mass_low_3sig'] = trylog10(averages['stellar_mass_3_sig_minus'])
        			tbhdu.header['HIERARCH EBV'] = best_ebv
        			tbhdu.header['HIERARCH ssp_number'] =len(order)
        
        			# quantities per SSP
        			for iii in range(len(order)):
        				tbhdu.header['HIERARCH stellar_mass_ssp_'+str(iii)] = trylog10(mass_per_ssp[order][iii])
        				tbhdu.header['HIERARCH living_stars_mass_ssp_'+str(iii)] = trylog10(final_ML_totM[order][iii])
        				tbhdu.header['HIERARCH remnant_mass_ssp_'+str(iii)] = trylog10(final_ML_alive[order][iii])
        				tbhdu.header['HIERARCH remnant_mass_in_whitedwarfs_ssp_'+str(iii)] = trylog10(final_ML_wd[order][iii])
        				tbhdu.header['HIERARCH remnant_mass_in_neutronstars_ssp_'+str(iii)] = trylog10(final_ML_ns[order][iii])
        				tbhdu.header['HIERARCH remnant_mass_in_blackholes_ssp_'+str(iii)] = trylog10(final_ML_bh[order][iii])
        				tbhdu.header['HIERARCH mass_of_ejecta_ssp_'+str(iii)] = trylog10(mass_per_ssp[order][iii] - final_ML_totM[order][iii])
        				tbhdu.header['HIERARCH log_age_ssp_'+str(iii)] = trylog10(age_per_ssp[order][iii])
        				tbhdu.header['HIERARCH metal_ssp_'+str(iii)] = trylog10(metal_per_ssp[order][iii])
        				tbhdu.header['HIERARCH SFR_ssp_'+str(iii)] = trylog10(mass_per_ssp[order][iii]/age_per_ssp[order][iii])	
        				tbhdu.header['HIERARCH weightMass_ssp_'+str(iii)] = weight_mass_per_ssp[order][iii]
        				tbhdu.header['HIERARCH weightLight_ssp_'+str(iii)] = weight_light_per_ssp[order][iii]
        
			self.tbhdu = tbhdu
        
			prihdr = pyfits.Header()
			prihdr['file'] = self.specObs.path_to_spectrum
			prihdr['model'] = self.models
			prihdr['ageMin'] = self.age_limits[0]
			prihdr['ageMax'] = self.age_limits[1]
			prihdr['Zmin'] = self.Z_limits[0]
			prihdr['Zmax'] = self.Z_limits[1]
			prihdu = pyfits.PrimaryHDU(header=prihdr)
			self.thdulist = pyfits.HDUList([prihdu, tbhdu])

			if self.write_results :
				if os.path.isfile(self.outputFile + self.suffix ):
					os.remove(self.outputFile + self.suffix )
				#print self.outputFile + self.suffix , thdulist, thdulist[1].data, thdulist[0].header
				self.thdulist.writeto(self.outputFile + self.suffix )
				return 1.

			else :
				return 0.


	def create_dummy_hdu(self):
		"""
		creates an empty HDU table in case computation did not converge
		"""
		default_array = np.array([default_value,default_value])
		waveCol = pyfits.Column(name="wavelength",format="D", unit="Angstrom", array= default_array)
		dataCol = pyfits.Column(name="original_data",format="D", unit="1e-17erg/s/cm2/Angstrom", array= default_array)
		errorCol = pyfits.Column(name="flux_error",format="D", unit="1e-17erg/s/cm2/Angstrom", array= default_array)
		best_fitCol = pyfits.Column(name="firefly_model",format="D", unit="1e-17erg/s/cm2/Angstrom", array= default_array)
		cols = pyfits.ColDefs([ waveCol, dataCol, errorCol, best_fitCol])
		tbhdu = pyfits.BinTableHDU.from_columns(cols)

		tbhdu.header['IMF'] = dict_imfs[self.imfs[0]]
		tbhdu.header['library'] = self.model_libs[0]
		tbhdu.header['HIERARCH converged'] = 'False'
		tbhdu.header['HIERARCH age_lightW'] = default_value
		tbhdu.header['HIERARCH age_lightW_up'] = default_value
		tbhdu.header['HIERARCH age_lightW_low'] = default_value
		tbhdu.header['HIERARCH metallicity_lightW'] = default_value
		tbhdu.header['HIERARCH metallicity_lightW_up'] = default_value
		tbhdu.header['HIERARCH metallicity_lightW_low'] = default_value
		tbhdu.header['HIERARCH age_massW']             = default_value
		tbhdu.header['HIERARCH age_massW_up']          = default_value
		tbhdu.header['HIERARCH age_massW_low']         = default_value
		tbhdu.header['HIERARCH metallicity_massW']     = default_value
		tbhdu.header['HIERARCH metallicity_massW_up']  = default_value
		tbhdu.header['HIERARCH metallicity_massW_low'] = default_value
		tbhdu.header['HIERARCH stellar_mass']      = default_value
		tbhdu.header['HIERARCH stellar_mass_up']   = default_value
		tbhdu.header['HIERARCH stellar_mass_low']  = default_value
		tbhdu.header['HIERARCH EBV']                   = default_value
		tbhdu.header['HIERARCH ssp_number']            = default_value
		return tbhdu
