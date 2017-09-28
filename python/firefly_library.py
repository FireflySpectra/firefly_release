from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
if sys.version > '3':
    long = int

import numpy as np
import copy
import cPickle
from scipy.stats import chi2 


#-----------------------------------------------------------------------
def airtovac(wave_air):
    """ 
	__author__ = 'Kyle B. Westfall'

    Wavelengths are corrected for the index of refraction of air under
    standard conditions.  Wavelength values below 2000 A will not be
    altered.  Uses formula from Ciddor 1996, Applied Optics 62, 958.

    Args:
        wave_air (int or float): Wavelength in Angstroms, scalar or
            vector. If this is the only parameter supplied, it will be
            updated on output to contain double precision vacuum
            wavelength(s). 

    Returns:
        numpy.float64 : The wavelength of the line in vacuum.

    Example:
        If the air wavelength is  W = 6056.125 (a Krypton line), then
        :func:`airtovac` returns vacuum wavelength of W = 6057.8019.
 
    *Revision history*:
        | Written W. Landsman                November 1991
        | Use Ciddor (1996) formula for better accuracy in the infrared 
        |   Added optional output vector, W Landsman Mar 2011
        | Iterate for better precision W.L./D. Schlegel  Mar 2011
        | Transcribed to python, K.B. Westfall Apr 2015

    .. note::
        Take care within 1 A of 2000 A.   Wavelengths below 2000 A *in
        air* are not altered.       

    """

    # Copy the data
    wave_vac = wave_air.astype(np.float64) if hasattr(wave_air, "__len__") else float(wave_air)
    g = wave_vac > 2000.0                           # Only modify above 2000 A
    Ng = np.sum(g)
    
    if Ng > 0:
        # Handle both arrays and scalars
        if hasattr(wave_air, "__len__"):
            _wave_air = wave_air[g].astype(np.float64)
            _wave_vac = wave_vac[g]
        else:
            _wave_air = float(wave_air)
            _wave_vac = float(wave_vac)

        for i in range(0,2):
            sigma2 = np.square(1.0e4/_wave_vac)     #Convert to wavenumber squared
            fact = 1.0 + 5.792105e-2/(238.0185 - sigma2) + 1.67917e-3/(57.362 - sigma2)
            _wave_vac = _wave_air*fact

        if hasattr(wave_air, "__len__"):        # Save the result
            wave_vac[g] = _wave_vac
        else:
            wave_vac = _wave_vac

    return wave_vac

#-----------------------------------------------------------------------
def bisect_array(array):
	"""
	It takes an array as input and returns the bisected array : 
	bisected array[i] = (array[i] + array[i+1] )/2. Its lenght is one less than the array.

	:param array: input array
	"""
	bisected_array = np.zeros(len(array) - 1)
	for ai in range(len(bisected_array)):
		bisected_array[ai] = (array[ai] + array[ai + 1])/2.0
	return bisected_array


#-----------------------------------------------------------------------
def max_pdf(probs,property,sampling):
	"""
	determines the maximum of a pdf of a property for a given sampling

	:param probs: probabilities
	:param  property: property
	:param  sampling: sampling of the property
	"""
	lower_limit 	= np.min(property)
	upper_limit 	= np.max(property)
	error_interval = np.round(upper_limit, 2) - np.round(lower_limit, 2)

	if np.round(upper_limit, 2) == np.round(lower_limit, 2) or error_interval <= abs((upper_limit/100.)*3):
		return np.asarray(property),np.ones(len(probs))/np.size(probs)

	property_pdf_int= np.arange(lower_limit, upper_limit * 1.001, (upper_limit-lower_limit) /sampling ) + ( upper_limit - lower_limit) * 0.000001	
	prob_pdf 		= np.zeros(len(property_pdf_int))

	for p in range(len(property_pdf_int)-1):
		match_prop = np.where( (property <= property_pdf_int[p+1]) & (property > property_pdf_int[p]) )
		if np.size(match_prop) == 0:
			continue
		else:
			prob_pdf[p] = np.max( probs[match_prop] )

	property_pdf = bisect_array(property_pdf_int)
	return property_pdf,prob_pdf[:-1]/np.sum(prob_pdf)

#-----------------------------------------------------------------------
def convert_chis_to_probs(chis,dof):
	"""
	Converts chi squares to probabilities.

	:param chis: array containing the chi squares.
	:param dof: array of degrees of freedom.
	"""
	chis = chis / np.min(chis) * dof
	prob =  1.0 - chi2.cdf(chis,dof)
	prob = prob / np.sum(prob)
	return prob

#-----------------------------------------------------------------------
def light_weights_to_mass(light_weights,mass_factors):
	"""
	Uses the data/model mass-to-light ratio to convert
	SSP contribution (weights) by light into 
	SSP contributions by mass.

	:param light_weights: light (luminosity) weights obtained when model fitting
	:param mass_factors: mass factors obtained when normalizing the spectrum
	"""
	mass_weights 	= np.zeros(np.shape(light_weights))
	unnorm_mass 	= np.zeros(np.shape(light_weights))
	for w in range(len(light_weights)):
		unnorm_mass[w]	= light_weights[w] * mass_factors
		mass_weights[w] = unnorm_mass[w] / np.sum(unnorm_mass[w])
	return unnorm_mass,mass_weights

#-----------------------------------------------------------------------
def find_closest(A, target):
	"""
	returns the id of the target in the array A.
	:param A: Array, must be sorted
	:param target: target value to be located in the array.
	"""
	idx = A.searchsorted(target)
	idx = np.clip(idx, 1, len(A)-1)
	left = A[idx-1]
	right = A[idx]
	idx -= target - left < right - target
	return idx

#-----------------------------------------------------------------------
def averages_and_errors(probs,prop,sampling):
	"""
	determines the average and error of a property for a given sampling
	
	returns : an array with the best fit value, +/- 1, 2, 3 sigma values.

	:param probs: probabilities
	:param  property: property
	:param  sampling: sampling of the property
	"""
	# This prevents galaxies with 1 unique solution from going any further. This is because the code crashes when constructing the likelihood
	# distributions. HACKY, but we need to think about this...
	if ((len(probs) <= 1) or (len(prop[~np.isnan(prop)]) <= 1)):
		best_fit, upper_onesig,lower_onesig, upper_twosig,lower_twosig, upper_thrsig,lower_thrsig = 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0
	else:
		xdf,y = max_pdf(probs,prop,sampling)
		cdf = np.zeros(np.shape(y))
		cdf_probspace = np.zeros(np.shape(y))
	
		for m in range(len(y)):
			cdf[m] = np.sum(y[:m])

		cdf = cdf / np.max(cdf)
		area_probspace = y*(xdf[1]-xdf[0])
		area_probspace = area_probspace/np.sum(area_probspace)
		indx_probspace = np.argsort(area_probspace)[::-1]
		desc_probspace = np.sort(area_probspace)[::-1]

		cdf_probspace = np.zeros(np.shape(desc_probspace))
		for m in range(len(desc_probspace)):
			cdf_probspace[m] = np.sum(desc_probspace[:m])

		av_sigs = [0.6827,0.9545,0.9973] # Median, + / - 1 sig, + / - 2 sig, + / - 3 sig

		# Sorts results by likelihood and calculates confidence intervals on sorted space
		index_close = find_closest(cdf_probspace, av_sigs)
		
		best_fit 					= xdf[indx_probspace[0]]
		upper_onesig,lower_onesig 	= np.max(xdf[indx_probspace[:index_close[0]]]),np.min(xdf[indx_probspace[:index_close[0]]])
		upper_twosig,lower_twosig 	= np.max(xdf[indx_probspace[:index_close[1]]]),np.min(xdf[indx_probspace[:index_close[1]]])
		upper_thrsig,lower_thrsig 	= np.max(xdf[indx_probspace[:index_close[2]]]),np.min(xdf[indx_probspace[:index_close[2]]])

		if np.size(xdf) == 0:
			raise Exception('No solutions found??? FIREFLY error (see statistics.py)')
	
	return [best_fit,upper_onesig,lower_onesig,upper_twosig,lower_twosig,upper_thrsig,lower_thrsig]

#-----------------------------------------------------------------------
def calculate_averages_pdf(probs,light_weights,mass_weights,unnorm_mass,age,metal,sampling,dist_lum, flux_units):

	"""
	Calculates light- and mass-averaged age and metallicities.
	Also outputs stellar mass and mass-to-light ratios.
	And errors on all of these properties.

	It works by taking the complete set of probs-properties and
	maximising over the parameter range (such that solutions with
	equivalent values but poorer probabilities are excluded). Then,
	we calculate the median and 1/2 sigma confidence intervals from 
	the derived 'max-pdf'.

	NB: Solutions with identical SSP component contributions 
	are re-scaled such that the sum of probabilities with that
	component = the maximum of the probabilities with that component.
	i.e. prob_age_ssp1 = max(all prob_age_ssp1) / sum(all prob_age_ssp1) 
	This is so multiple similar solutions do not count multiple times.

	Outputs a dictionary of:
	- light_[property], light_[property]_[1/2/3]_sigerror
	- mass_[property], mass_[property]_[1/2/3]_sigerror
	- stellar_mass, stellar_mass_[1/2/3]_sigerror
	- mass_to_light, mass_to_light_[1/2/3]_sigerror
	- maxpdf_[property]
	- maxpdf_stellar_mass
	where [property] = [age] or [metal]

	:param probs: probabilities
	:param light_weights: light (luminosity) weights obtained when model fitting
	:param mass_weights: mass weights obtained when normalizing models to data
	:param unnorm_mass: mass weights obtained from the mass to light ratio
	:param age: age
	:param metal: metallicity
	:param sampling: sampling of the property
	:param dist_lum: luminosity distance in cm
	"""

	# Sampling number of max_pdf (100:recommended) from options
	# Keep the age in linear units of Age(Gyr)
	log_age = age
	
	av = {} # dictionnary where values are stored :
	av['light_age'],av['light_age_1_sig_plus'],av['light_age_1_sig_minus'], av['light_age_2_sig_plus'], av['light_age_2_sig_minus'], av['light_age_3_sig_plus'], av['light_age_3_sig_minus'] = averages_and_errors(probs,np.dot(light_weights,log_age),sampling)
	
	av['light_metal'], av['light_metal_1_sig_plus'], av['light_metal_1_sig_minus'], av['light_metal_2_sig_plus'], av['light_metal_2_sig_minus'], av['light_metal_3_sig_plus'], av['light_metal_3_sig_minus'] = averages_and_errors(probs, np.dot(light_weights, metal), sampling)
	
	av['mass_age'], av['mass_age_1_sig_plus'], av['mass_age_1_sig_minus'], av['mass_age_2_sig_plus'], av['mass_age_2_sig_minus'], av['mass_age_3_sig_plus'], av['mass_age_3_sig_minus'] = averages_and_errors(probs, np.dot(mass_weights, log_age), sampling)
	
	av['mass_metal'], av['mass_metal_1_sig_plus'], av['mass_metal_1_sig_minus'], av['mass_metal_2_sig_plus'], av['mass_metal_2_sig_minus'], av['mass_metal_3_sig_plus'], av['mass_metal_3_sig_minus'] = averages_and_errors(probs, np.dot(mass_weights, metal), sampling)
	
	conversion_factor 	= flux_units * 4 * np.pi * dist_lum**2.0 # unit 1e-17 cm2 

	# Keep the mass in linear units until later M/M_{odot}.
	tot_mass = np.sum(unnorm_mass, 1) * conversion_factor
	av['stellar_mass'], av['stellar_mass_1_sig_plus'], av['stellar_mass_1_sig_minus'], av['stellar_mass_2_sig_plus'], av['stellar_mass_2_sig_minus'], av['stellar_mass_3_sig_plus'], av['stellar_mass_3_sig_minus'] = averages_and_errors(probs,tot_mass,sampling)


	return av


#-----------------------------------------------------------------------
def normalise_spec(data_flux,model_flux):
	"""
	Normalises all models to the median value of the spectrum.
	Saves the factors for later use.

	Outputs : normed models and translation factors.

	:param data_flux: observed flux in the data
	:param model_flux: flux from the models
	"""
	data_norm 				= np.median(data_flux)
	num_mods 				= len(model_flux)
	model_norm,mass_factor 	= np.zeros(num_mods),np.zeros(num_mods)
	normed_model_flux 		= np.zeros((num_mods,len(model_flux[0])))

	for m in range(len(model_flux)):
		model_norm[m] 			= np.median(model_flux[m])
		mass_factor[m] 			= data_norm/model_norm[m]
		normed_model_flux[m] 	= model_flux[m] / model_norm[m] * data_norm

	return normed_model_flux,mass_factor

#-----------------------------------------------------------------------
def match_data_models( data_wave_int, data_flux_int, data_flags, error_flux_int, model_wave_int, model_flux_int, min_wave_in, max_wave_in, saveDowngradedModel = True, downgradedModelFile = "DGmodel.txt"):
	"""
	 * 0.Take data and models as inputs 
	 * 1. interpolate data and model to the lowest sampled array.
		* 1.1. Defines the wavelength range on the model and on the data
		* 1.2. Downgrades the array, model or data, that has most sampling 
	 	* 1.3. integrate between them to output a matched resolution array for data and model
	 * 2. Returns the matched wavelength array, the corresponding data, error and model arrays : matched_wave,matched_data,matched_error,matched_model

	:param data_wave_int: data wavelength array in the restframe
	:param data_flux_int: data flux array
	:param data_flags: data quality flag array : 1 for good data
	:param error_flux_int: data flux error array
	:param model_wave_int: model wavelength array (in the rest frame)
	:param model_flux_int: model flux array
	:param min_wave_in: minimum wavelength to be considered
	:param max_wave_in: maximum wavelength to be considered
	:param saveDowngradedModel: if True it will save the downgraded models
	:param downgradedModelFile: location where downgreaded models will be saved
	"""  
	# 1. interpolate onto the bisection of lowest sampled array.
	num_models = len(model_flux_int)
	# 1.1. Defines the wavelength range on the model and on the data
	min_wave = np.max([np.min(data_wave_int[np.where(data_flags==1)]), np.min(model_wave_int),min_wave_in])
	max_wave = np.min([np.max(data_wave_int[np.where(data_flags==1)]), np.max(model_wave_int),max_wave_in])
	#print np.min(data_wave_int[np.where(data_flags==1)]), np.min(model_wave_int), min_wave_in
	#print np.max(data_wave_int[np.where(data_flags==1)]), np.max(model_wave_int), max_wave_in
	loc_model 	= np.array(( model_wave_int <= max_wave) & (model_wave_int >= min_wave))
	if np.sum(loc_model)==0:
		raise ValueError("The wavelength range input is below or above model wavelength coverage!")
	model_wave 	= model_wave_int[loc_model]
	num_mod  	= np.sum(loc_model)
	model_flux 	= np.zeros((num_models,num_mod))

	for m in range(num_models):
		model_flux[m] = model_flux_int[m][loc_model]
	
	loc_data 	= np.array(( data_wave_int <= max_wave) & (data_wave_int >= min_wave)) 
	if np.sum(loc_data)==0:
		raise ValueError("The wavelength range input is below or above data wavelength coverage!")
	num_dat  	= np.sum(loc_data)
	data_wave 	= data_wave_int[loc_data]
	data_flux 	= data_flux_int[loc_data]
	error_flux 	= error_flux_int[loc_data]
	# 1.2. Downgrades the array, model or data, that has most sampling 
	if num_mod >= num_dat:
		#print "More model points than data points! Downgrading models to data sampling ..."
		bisect_data = bisect_array(data_wave) + np.min(data_wave)*0.0000000001
		matched_model = np.zeros((num_models,len(bisect_data) - 1))
		for m in range(num_models):
			model_flux_bounds 	= np.interp(bisect_data, model_wave, model_flux[m])
			combined_wave_int 	= np.concatenate((model_wave,bisect_data))
			combined_flux_int 	= np.concatenate((model_flux[m],model_flux_bounds))
			sort_indices 		= np.argsort(combined_wave_int)

			combined_wave 		= np.sort(combined_wave_int)
			boundary_indices 	= np.searchsorted(combined_wave,bisect_data)
			combined_flux 		= combined_flux_int[sort_indices]

			len_combo = len(combined_flux)
			# 1.3. produces a matched resolution array
			for l in range(len(boundary_indices) - 1):
				if boundary_indices[l + 1] >= len_combo:
					matched_model[m][l] = matched_model[m][l - 1]
				else:
					matched_model[m][l] = np.trapz(combined_flux[boundary_indices[l] : boundary_indices[l + 1] +  1], x=combined_wave[boundary_indices[l] :boundary_indices[l + 1] + 1]) / (combined_wave[boundary_indices[l + 1]] -  combined_wave[boundary_indices[l] ])

		matched_wave = data_wave[1:-1]
		matched_data = data_flux[1:-1]
		matched_error = error_flux[1:-1]
		# OPTION : saves the downgraded models.
		if saveDowngradedModel:
			#print "saving downgraded models to ",downgradedModelFile
			f.open(downgradedModelFile,'w')
			cPickle.dump([matched_wave, matched_data, matched_error],f)
			f.close()

	else:
		#print "More data points than model points! Downgrading data to model sampling ..."
		bisect_model = bisect_array(model_wave) + np.min(model_wave)*0.0000000001
		boundaries 	= np.searchsorted(data_wave,bisect_model)

		data_flux_bounds 	= np.interp(bisect_model, data_wave, data_flux)
		error_flux_bounds 	= np.interp(bisect_model, data_wave, error_flux)
		combined_wave_int 	= np.concatenate((data_wave,bisect_model))
		combined_flux_int 	= np.concatenate((data_flux,data_flux_bounds))
		combined_error_int 	= np.concatenate((data_flux,error_flux_bounds))
		sort_indices 		= np.argsort(combined_wave_int)
	
		combined_wave 		= np.sort(combined_wave_int)
		boundary_indices 	= np.searchsorted(combined_wave,bisect_model)
		combined_flux 		= combined_flux_int[sort_indices]
		combined_error 		= combined_error_int[sort_indices]

		# 1.3. produces a matched resolution array
		matched_data,matched_error= np.zeros(len(boundary_indices) - 1),np.zeros(len(boundary_indices) - 1)

		len_combo = len(combined_flux)
		for l in range(len(boundary_indices) - 1):
			if boundary_indices[l + 1] >= len_combo:
				matched_data[l] 	= matched_data[l - 1]
				matched_error[l] 	= matched_error[l - 1]
			else:
				matched_data[l] 	= np.trapz(combined_flux[boundary_indices[l]:boundary_indices[l + 1] + 1], x=combined_wave[boundary_indices[l]: boundary_indices[l + 1] + 1])/  (combined_wave[boundary_indices[l + 1]] - combined_wave[boundary_indices[l]])
				matched_error[l] 	= np.trapz(combined_error[boundary_indices[l]:boundary_indices[l + 1] + 1], x=combined_wave[boundary_indices[l]:boundary_indices[l + 1] + 1])/ (combined_wave[boundary_indices[l + 1]] - combined_wave[boundary_indices[l]])

		matched_wave 		= model_wave[1:-1]
		matched_model 		= np.zeros((num_models,len(matched_wave)))
		for m in range(num_models):
			matched_model[m][:] = model_flux[m][1:-1]

	return matched_wave,matched_data,matched_error,matched_model

