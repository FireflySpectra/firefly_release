# This script can be run as follows:
#> python run_stellarpop.py ../example_data/spectra/0266/ ../../output/
from os.path import join
import sys, os
import time
import numpy as np
import glob
import astropy.cosmology as co
# Firefly modules
import GalaxySpectrumFIREFLY as gs
import StellarPopulationModel as spm

def runSpec(specLiteFile):
	baseN = os.path.basename(specLiteFile).split('-')
	plate = baseN[1] 
	mjd   = baseN[2]
	fibre, dum = os.path.basename(baseN[3]).split('.')
	
	t0=time.time()
	spec=gs.GalaxySpectrumFIREFLY(specLiteFile, milky_way_reddening=True)
	spec.openObservedSDSSSpectrum(survey='sdssMain')

	ageMin = 0. ; ageMax = np.log10(cosmo.age(spec.redshift).value*1e9)
	ZMin = 0.001 ; ZMax = 4.

	if spec.hdulist[2].data['CLASS'][0]=="GALAXY" and spec.hdulist[2].data['Z'][0] >  spec.hdulist[2].data['Z_ERR'][0] and spec.hdulist[2].data['Z_ERR'][0]>0 and spec.hdulist[2].data['ZWARNING'][0] ==0 :
		
		# This needs to be changed to your own directory
		outputFolder = join( os.environ['FF_DIR'], 'output','plate')
		output_file = join(outputFolder , 'spFly-' + os.path.basename(specLiteFile)[5:-5] )+".fits"
		print( "Start time=",t0            )
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
			model_2 = spm.StellarPopulationModel(spec, output_file, cosmo, models = models_key, model_libs = ['MILES'], imfs = ['ss'], age_limits = [ageMin,ageMax], downgrade_models = False, data_wave_medium = 'vacuum', Z_limits = [ZMin,ZMax], use_downgraded_models = True, write_results = False)
			model_2.fit_models_to_data()
			tables.append( model_2.tbhdu )
			# print "m2", time.time()-t0
		except (ValueError):
			tables.append( model_2.create_dummy_hdu() )
			did_not_converged +=1

		try :
			model_3 = spm.StellarPopulationModel(spec, output_file, cosmo, models = models_key, model_libs = ['MILES'], imfs = ['kr'], age_limits = [ageMin,ageMax], downgrade_models = False, data_wave_medium = 'vacuum', Z_limits = [ZMin,ZMax], use_downgraded_models = True, write_results = False)
			model_3.fit_models_to_data()
			tables.append( model_3.tbhdu )
			# print "m3", time.time()-t0
		except (ValueError):
			tables.append( model_3.create_dummy_hdu() )
			did_not_converged +=1
			
		try :
			model_4 = spm.StellarPopulationModel(spec, output_file, cosmo, models = models_key, model_libs = ['ELODIE'], imfs = ['cha'], age_limits = [ageMin,ageMax], downgrade_models = False, data_wave_medium = 'vacuum', Z_limits = [ZMin,ZMax], use_downgraded_models = True, write_results = False)
			model_4.fit_models_to_data()
			# print "m4", time.time()-t0
			tables.append( model_4.tbhdu )
			did_not_converged +=1

		except (ValueError):
			tables.append( model_4.create_dummy_hdu() )
			

		try :
			model_5 = spm.StellarPopulationModel(spec, output_file, cosmo, models = models_key, model_libs = ['ELODIE'], imfs = ['ss'], age_limits = [ageMin,ageMax], downgrade_models = False, data_wave_medium = 'vacuum', Z_limits = [ZMin,ZMax], use_downgraded_models = True, write_results = False)
			model_5.fit_models_to_data()
			tables.append( model_5.tbhdu )
			# print "m5", time.time()-t0
		except (ValueError):
			tables.append( model_5.create_dummy_hdu() )
			did_not_converged +=1
			

		try :
			model_6 = spm.StellarPopulationModel(spec, output_file, cosmo, models = models_key, model_libs = ['ELODIE'], imfs = ['kr'], age_limits = [ageMin,ageMax], downgrade_models = False, data_wave_medium = 'vacuum', Z_limits = [ZMin,ZMax], use_downgraded_models = True, write_results = False)
			model_6.fit_models_to_data()
			tables.append( model_6.tbhdu )
			# print "m6", time.time()-t0
		except (ValueError):
			tables.append( model_6.create_dummy_hdu() )
			did_not_converged +=1
		
		try :
			model_7 = spm.StellarPopulationModel(spec, output_file, cosmo, models = models_key, model_libs = ['STELIB'], imfs = ['cha'], age_limits = [ageMin,ageMax], downgrade_models = False, data_wave_medium = 'vacuum', Z_limits = [ZMin,ZMax], use_downgraded_models = True, write_results = False)
			model_7.fit_models_to_data()
			# print "m4", time.time()-t0
			tables.append( model_7.tbhdu )

		except (ValueError):
			tables.append( model_7.create_dummy_hdu() )
			did_not_converged +=1
			

		try :
			model_8 = spm.StellarPopulationModel(spec, output_file, cosmo, models = models_key, model_libs = ['STELIB'], imfs = ['ss'], age_limits = [ageMin,ageMax], downgrade_models = False, data_wave_medium = 'vacuum', Z_limits = [ZMin,ZMax], use_downgraded_models = True, write_results = False)
			model_8.fit_models_to_data()
			tables.append( model_8.tbhdu )
			# print "m5", time.time()-t0
		except (ValueError):
			tables.append( model_8.create_dummy_hdu() )
			did_not_converged +=1
			

		try :
			model_9 = spm.StellarPopulationModel(spec, output_file, cosmo, models = models_key, model_libs = ['STELIB'], imfs = ['kr'], age_limits = [ageMin,ageMax], downgrade_models = False, data_wave_medium = 'vacuum', Z_limits = [ZMin,ZMax], use_downgraded_models = True, write_results = False)
			model_9.fit_models_to_data()
			tables.append( model_9.tbhdu )
			# print "m6", time.time()-t0
		except (ValueError):
			tables.append( model_9.create_dummy_hdu() )
			did_not_converged +=1
		
		if did_not_converged < 9 :
			complete_hdus = spm.pyfits.HDUList(tables)
			if os.path.isfile(output_file):
				os.remove(output_file)
				
			complete_hdus.writeto(output_file)

	print ("time used =", time.time()-t0 ,"seconds")

	return spec


#-------------------------------
indir = sys.argv[1] #; print 'Input directory: ',indir
outdir = sys.argv[2] #; print 'Output directory: ',outdir

cosmo = co.Planck15
models_key = 'm11'

fileList = np.array(glob.glob(indir+'spec-*.fits'))

overwrite = True ; print 'Overwrite = ',overwrite

for el in fileList:
    outputFile = outdir+'spFly-'+os.path.basename(el)[5:-5]+'.fits'
    if (not os.path.isfile(outputFile) or overwrite) :
	spec = runSpec(el)
    else:
        print outputFile
		
print 'The end'
