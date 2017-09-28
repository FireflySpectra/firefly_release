"""
@author: Johan Comparat
@license: Gnu Public Licence
"""
from scipy.interpolate import interp1d
from scipy.integrate import quad
import numpy as np
import os 

class Photo():
	"""
	Class derived from Instrument class for imaging.

	Attributes:
		* Throughput as a function of wavelength in Angtroms
		* Collecting area in square meters
		* Telescope diameter in meters
		* ReadNoise in e-/pix
		* Dark Current in e-/s/pix
		* Pixel scale in arcsec
		* list of filters    
		* Functions prototype:
		* loadMegacam()
		* loadSDSS()
	"""
	def __init__(self, name="SDSS"):
		"""
		:type name: string
		:param name: name of the instrument, CFHT or SDSS
		""" 
		self.c= 299792458. # speed of light m/s
		self.name = name
		
		if self.name=="SDSS":
			self.loadSDSS()
			
		if self.name=="CFHT":
			self.loadMegacam()
			self.CollectingArea=31.4
			self.TelescopeDiameter=3.58
			self.ReadNoise=5.
			self.DarkCurrent=0.0005
			self.PixelScale=0.187
			
		
	def loadMegacam(self):
		"""
		Loads the filters of Megacam, wavelength and percent. Not normalized
		"""
		uf=np.loadtxt(os.path.join(os.environ["GIT_SPM"], "filter/cfht/megacam_filter/uMegacam.filter"), unpack=True)                  
		gf=np.loadtxt(os.path.join(os.environ["GIT_SPM"], "filter/cfht/megacam_filter/gMegacam.filter"), unpack=True)
		rf=np.loadtxt(os.path.join(os.environ["GIT_SPM"], "filter/cfht/megacam_filter/rMegacam.filter"), unpack=True)
		yf=np.loadtxt(os.path.join(os.environ["GIT_SPM"], "filter/cfht/megacam_filter/yMegacam.filter"), unpack=True)
		zf=np.loadtxt(os.path.join(os.environ["GIT_SPM"], "filter/cfht/megacam_filter/zMegacam.filter"), unpack=True)
		self.uFilterMegaCam=interp1d(uf[0],uf[1])
		self.gFilterMegaCam=interp1d(gf[0],gf[1])
		self.rFilterMegaCam=interp1d(rf[0],rf[1])
		self.yFilterMegaCam=interp1d(yf[0],yf[1])
		self.zFilterMegaCam=interp1d(zf[0],zf[1])
		self.nFilters=5
		self.interpolatedFilterList=np.array([self.uFilterMegaCam,self.gFilterMegaCam,self.rFilterMegaCam,self.yFilterMegaCam,self.zFilterMegaCam])        
		self.interpolatedFilterDict={"CFHT_u": self.uFilterMegaCam, "CFHT_g" : self.gFilterMegaCam, "CFHT_r" : self.rFilterMegaCam,  "CFHT_i" : self.yFilterMegaCam, "CFHT_z" : self.zFilterMegaCam}
		
	def loadSDSS(self):
		"""
		Loads the filters of SDSS camera, wavelength and percent. Not normalized.
		"""        
		self.filter_names = ["SDSS_u", "SDSS_g", "SDSS_r", "SDSS_i", "SDSS_z"]
		self.nFilters=5
		
		uf=np.loadtxt(os.path.join(os.environ["GIT_SPM"], "filter/sdss/uSDSS.filter"), unpack=True)                  
		gf=np.loadtxt(os.path.join(os.environ["GIT_SPM"], "filter/sdss/gSDSS.filter"), unpack=True)
		rf=np.loadtxt(os.path.join(os.environ["GIT_SPM"], "filter/sdss/rSDSS.filter"), unpack=True)
		yf=np.loadtxt(os.path.join(os.environ["GIT_SPM"], "filter/sdss/iSDSS.filter"), unpack=True)
		zf=np.loadtxt(os.path.join(os.environ["GIT_SPM"], "filter/sdss/zSDSS.filter"), unpack=True)
		
		self.lambda_min={'SDSS_u': np.min(uf[0]), 'SDSS_g':  np.min(gf[0]),'SDSS_r':  np.min(rf[0]),'SDSS_i':  np.min(yf[0]), 'SDSS_z':  np.min(zf[0]) }        
		self.lambda_max={'SDSS_u': np.max(uf[0]), 'SDSS_g':  np.max(gf[0]),'SDSS_r':  np.max(rf[0]),'SDSS_i':  np.max(yf[0]), 'SDSS_z':  np.max(zf[0]) }        
		self.lambda_eff={'SDSS_u': 3560., 'SDSS_g': 4830., 'SDSS_r': 6260.,'SDSS_i': 7670., 'SDSS_z': 9100. }        
				
		self.filterDict = {'SDSS_u': interp1d(uf[0],uf[1]), 'SDSS_g': interp1d(gf[0],gf[1]),'SDSS_r': interp1d(rf[0],rf[1]),'SDSS_i': interp1d(yf[0],yf[1]),'SDSS_z': interp1d(zf[0],zf[1])}        
		
		self.normDict = {}
		for filter in self.filter_names:
			integrand = lambda lb : self.filterDict[filter](lb)#/lb
			self.normDict[filter] = quad(integrand, self.lambda_min[filter], self.lambda_max[filter])[0]#/self.lambda_eff[filter]
				
			
	def computeMagnitudes(self, spectrum, distance_modulus = 0. ):
		"""
		This method computes the magnitude for a set of filters of an object with spectral energy distribution fLambda.
		:param spectrum : galaxy spectrum, interp1d object
		:param distance_modulus : 5 log(r/r0)
		"""
		# about the spectrum coverage
		wl_min = np.min(spectrum.x)
		wl_max = np.max(spectrum.x)
		#conversion to f nu convention erg/cm2/s/A => erg/cm2/s/Hz 
		#nus = self.c * 10**10 / spectrum.x # A => Hz
		#print( nus.min(), nus.max() )
		fnus = spectrum.y * spectrum.x **2.*10**(-10)/self.c
		#fNu = interp1d(nus[::-1],fnus[::-1])
		magAB = []
		contains_filter = []
		for filter in self.filter_names:
			# check overlap
			if wl_min<self.lambda_min[filter] and wl_max>self.lambda_max[filter] :
				contains_filter.append(True)
				# integrand to obtain the flux in the band
				#print(filter)
				#integrand  = lambda nu : fNu(nu) * self.filterDict[filter](10**(10)*self.c / nu)
				#integrand = lambda lb : self.filterDict[filter](lb)
				integrand2 = lambda lb : spectrum(lb) * self.filterDict[filter](lb) / self.normDict[filter]
				#print( 10**(10)*self.c /filter.x.min(), 10**(10)*self.c /filter.x.max() )
				#out=quad(integrand, 10**(10)*self.c /self.lambda_max[filter], 10**(10)*self.c /self.lambda_min[filter])[0]
				#out=quad(integrand, self.lambda_min[filter], self.lambda_max[filter])[0]#/self.lambda_eff[filter]
				f_lambda = quad(integrand2, self.lambda_min[filter], self.lambda_max[filter])[0]#/self.lambda_eff[filter]
				# conversion to magnitude   
				f_nu = f_lambda * self.lambda_eff[filter] **2.*10**(-10)/self.c
				mag = -2.5*np.log10(f_nu) - 48.6 - distance_modulus
				magAB.append( mag ) # -2.5*np.log10(out2)-48.6)
			else:
				contains_filter.append(False)
				magAB.append(-9999)
			
			
		return np.array(magAB), contains_filter

