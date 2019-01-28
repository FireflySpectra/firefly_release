# This script can be run alone using the example spectra:
#> python one_spectra.py 'example_data/spectra/0266/spec-0266-51630-0623.fits'
import numpy as np
import sys, os
from os.path import join
import astropy.cosmology as co
import time
import numpy as np
# Firefly modules
import GalaxySpectrumFIREFLY as gs
import StellarPopulationModel as spm

cosmo = co.Planck15
models_key = 'm11'

def runSpec(specLiteFile):
	baseN = os.path.basename(specLiteFile).split('-')
	plate = baseN[1] 
	mjd   = baseN[2]
	fibre, dum = os.path.basename(baseN[3]).split('.')
	
	t0=time.time()
	spec=gs.GalaxySpectrumFIREFLY(specLiteFile, milky_way_reddening=True)
	spec.openObservedSDSSSpectrum(survey='sdssMain')

	ageMin = 0. ; ageMax = 15.
	ZMin = 0.001 ; ZMax = 4.
	if spec.hdulist[2].data['CLASS'][0]=="GALAXY" and spec.hdulist[2].data['Z'][0] >  spec.hdulist[2].data['Z_ERR'][0] and spec.hdulist[2].data['Z_ERR'][0]>0 and spec.hdulist[2].data['ZWARNING'][0] ==0 :
		
		# This needs to be changed to your own directory
		outputFolder = join(os.environ['FF_DIR'],'output')
		output_file = join(outputFolder , 'spFly-' + os.path.basename(specLiteFile)[5:-5] )+".fits"

		print( 'Output file:'              )
		print( output_file                 )
		print( "--------------------------")
		if os.path.isdir(outputFolder)==False:
			os.mkdir(outputFolder)
				
		prihdr = spm.pyfits.Header()
		prihdr['FILE']          = os.path.basename(output_file)
		prihdr['PLATE']         = plate 
		prihdr['MJD']           = mjd   
		prihdr['FIBERID']       = fibre 
		prihdr['MODELS']	= models_key
		prihdr['FITTER']	= "FIREFLY"	
		prihdr['AGEMIN']	= str(ageMin)		
		prihdr['AGEMAX']	= str(ageMax)
		prihdr['ZMIN']	        = str(ZMin)
		prihdr['ZMAX']	        = str(ZMax)
		prihdr['redshift']	= spec.hdulist[2].data['Z'][0]
		prihdr['HIERARCH age_universe']	= np.round(cosmo.age(spec.hdulist[2].data['Z'][0]).value,3)
		prihdu = spm.pyfits.PrimaryHDU(header=prihdr)

		tables = [prihdu]
		did_not_converged = 0.

		try :
			model_1 = spm.StellarPopulationModel(spec, output_file, cosmo, models = models_key, model_libs = ['MILES'], imfs = ['cha'], age_limits = [ageMin,ageMax], downgrade_models = False, data_wave_medium = 'vacuum', Z_limits = [ZMin,ZMax], use_downgraded_models = True, write_results = False)
			model_1.fit_models_to_data()
			tables.append( model_1.tbhdu )
			# print "m1", time.time()-t0
		except (ValueError):
			tables.append( model_1.create_dummy_hdu() )
			did_not_converged +=1

		try :
			model_1 = spm.StellarPopulationModel(spec, output_file, cosmo, models = models_key, model_libs = ['ELODIE'], imfs = ['cha'], age_limits = [ageMin,ageMax], downgrade_models = False, data_wave_medium = 'vacuum', Z_limits = [ZMin,ZMax], use_downgraded_models = True, write_results = False)
			model_1.fit_models_to_data()
			tables.append( model_1.tbhdu )
			# print "m1", time.time()-t0
		except (ValueError):
			tables.append( model_1.create_dummy_hdu() )
			did_not_converged +=1

		try :
			model_1 = spm.StellarPopulationModel(spec, output_file, cosmo, models = models_key, model_libs = ['STELIB'], imfs = ['cha'], age_limits = [ageMin,ageMax], downgrade_models = False, data_wave_medium = 'vacuum', Z_limits = [ZMin,ZMax], use_downgraded_models = True, write_results = False)
			model_1.fit_models_to_data()
			tables.append( model_1.tbhdu )
			# print "m1", time.time()-t0
		except (ValueError):
			tables.append( model_1.create_dummy_hdu() )
			did_not_converged +=1

		
		if did_not_converged < 4 :
			complete_hdus = spm.pyfits.HDUList(tables)
			if os.path.isfile(output_file):
				os.remove(output_file)
				
			complete_hdus.writeto(output_file)
	
	print ("time used =", time.time()-t0 ,"seconds")
	return spec

def main():

	# Get argument from file
	file_name = sys.argv[1]
	print file_name
	

	# Make sure output is in correct place.
	output_function = runSpec(file_name)

main()
