"""
.. moduleauthor:: Johan Comparat <johan.comparat__at__gmail.com>

General purpose:
................

The class StellarMassFunction is a wrapper dedicated to handling the fit of stellar mass function. 

*Imports*::

	import numpy as np
	import astropy.io.fits as pyfits
	import astropy.units as u
	import glob
	import pandas as pd
	import os
	
"""
import numpy as np
import astropy.io.fits as pyfits
import astropy.units as u
import glob
import pandas as pd
import os
import astropy.cosmology as co
cosmo = co.Planck15

class StellarMassFunction:
	"""
	:param imf_name: choose the `initial mass function <https://en.wikipedia.org/wiki/Initial_mass_function>`_:

		* 'ss' for `Salpeter <http://adsabs.harvard.edu/abs/1955ApJ...121..161S>`_or 
		* 'kr' for `Kroupa <http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1112.3340>`_ or 
		* 'cha' for `Chabrier <http://adsabs.harvard.edu/abs/2003PASP..115..763C>`_.
	
	:param params: List of parameters that characterize the mass function
	:param cosmo: cosmology class to be considered. Default Planck 15.
	
	 Notes
	 -----

	.. note::
		 * Mass function : number of stars, :math:`N`, in a volume, :math:`V`, observed at time, :math:`t`, in the logarithmic mass interval, :math:`dlogm`. :math:`\xi(log(m)) = d(N/V)/d(log(m))`. It is coded in self.mass_function(logm, params )
		 * Mass spectrum : number density per mass interval :math:`\xi(m)=\xi(\log(m))/(m * ln(m))` It is coded in self.mass_spectrum(m, params)
		 * the present day mass function is related to the observed luminosity function via magnitude - mass relationships.
		 * the initial mass function differs from the present day mass function by the evolution of the massive star. In the 'scalo86' parametrization it does differ. In the 'chabrier03' parametrization it does not.
		 

	"""
	def __init__(self, imf_name = 'salpeter', params=n.array([1e-3, -2.35]), cosmo = cosmo):
		self.cosmo = cosmo
		self.imf_name = imf_name
		self.params = params
		#
		
		if imf_name == 'salpeter':
			# power law as defined by Salpeter (1955) :
			self.mass_function = lambda logm : 0.001 * (10**logm)**-2.35 * (u.parsec)**(-3.)
			
			
		if imf_name == 'scalo86':
			self.mass_function = n.piecewise( logm, 
				[0<=logm & logm<=0.54, 0.54<logm & logm<=1.26, 1.26<logm & logm<=1.80 ],
				[lambda logm :  (10**logm)**(-4.37) * 0.044 * (u.parsec)**(-3.), 
				lambda logm :  (10**logm)**(-3.53) * 0.015 * (u.parsec)**(-3.), 
				lambda logm :  (10**logm)**(-2.11) * 0.0025 * (u.parsec)**(-3.)]
				)
			self.present_day_mf = lambda logm : mass_function(logm)
			self.initial_mf = lambda logm : (10**logm)**(-1.3) * 0.0443 * (u.parsec)**(-3.)
	
				
		if imf_name == 'chabrier03':
			self.mass_function = lambda logm : 0.158 * n.e**(- (logm - n.log10(0.079))**2./ (2*0.69**2.) ) * (u.parsec)**(-3.)
			self.present_day_mf = lambda logm : mass_function(logm)
			self.initial_mf = lambda logm : mass_function(logm)
	
					
		self.mass_spectrum = lambda m : self.mass_function(n.log10(m/u.solMass) ) / (m * n.log(10)) / u.solMass
			
		self.creation_function_logm = lambda logm, t : m*t * (u.parsec)**(-3.)
		self.present_day_mf_logm = lambda m, t : m*t * (u.parsec)**(-3.)
		self.initial_mf_logm = lambda m : m*t * (u.parsec)**(-3.)
		
		self.creation_function = lambda m, t : m*t * (u.parsec)**(-3.)
		self.present_day_mf = lambda m, t : m*t * (u.parsec)**(-3.)
		self.initial_mf = lambda m, t : m*t * (u.parsec)**(-3.)



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
		return True
