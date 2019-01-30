"""
Postprocessing of firefly's outputs.

Merges the specObjAll with firefly outputs at the plate level.
It creates a fits summary table for each plate - mjd combination

1. measures chi2 and ndof for each fit

2. creates a catalog per plate, adds chi2 and ndof.

3. re write per plate - mjd, delete per plate files

Example command :
python create_summary_tables.py 3785 EBOSSDR16_DIR

"""
import time
t0t=time.time()
from os.path import join
import sys, os
import numpy as n
import glob 
import astropy.io.fits as fits
from scipy.interpolate import interp1d

# order of the hdus in the model files
#hdu_header_prefix = n.array(["Chabrier_MILES_", "Salpeter_UVMILES_" , "Kroupa_MILES_"   , "Chabrier_ELODIE_", "Salpeter_ELODIE_", "Kroupa_ELODIE_"  , "Chabrier_STELIB_", "Salpeter_STELIB_", "Kroupa_STELIB_" ])
hdu_header_prefix = n.array(["Chabrier_MILES_", "Chabrier_ELODIE_" ])

dV=-9999

plate = sys.argv[1] # '9003'
env = sys.argv[2] # 'EBOSSDR16_DIR'

#print( plate, env )
if env == 'EBOSSDR16_DIR':
	z_name = 'Z_NOQSO'
if env == 'SDSSDR12_DIR':
	z_name = 'Z'

# initial catalog
init_cat = join( os.environ[env], "catalogs", "perPlate", "sp-"+plate.zfill(4)+".fits")

hdu_orig_table = fits.open(init_cat)
orig_table = hdu_orig_table[1].data
orig_cols = orig_table.columns

out_dir = os.path.join(os.environ[env], 'catalogs', 'FlyPlates')
path_2_out_file = join( out_dir, "spFlyPlate-"+plate.zfill(4)+".fits")

if os.path.isdir(out_dir)==False:
	os.makedirs(out_dir)

#############################################
#############################################
# STEP 1, computes chi2, ndof
#############################################
#############################################

# loops over id_spec to create the 

chi2 = n.ones((len(orig_table['MJD']),2))*dV
ndof = n.ones((len(orig_table['MJD']),2))*dV
absolute_magnitudes = n.ones((len(orig_table['MJD']),5))*dV
absolute_magnitudes_err = n.ones((len(orig_table['MJD']),5))*dV
#absolute_magnitudes_cha_elodie = n.ones((len(orig_table['MJD']),5))*dV

def get_spFly(plate, mjd, fiberid, id_spec):
		"""
		Computes the chi2 and number of degrees of freedom for each model.
		"""
		print(plate, mjd, fiberid, id_spec)
		#try :
		spFly_file = os.path.join(os.environ[env], 'firefly', plate, 'spFly-'+plate+'-'+mjd+'-'+fiberid+".fits")
		if os.path.isfile(spFly_file) :
			#print(spFly_file)
			# defines the output paths
			im_file = 'spFly-'+plate+'-'+mjd+'-'+fiberid+'.png'
			# opens the model file with 
			model = fits.open(spFly_file)
			# verifies it has converged
			converged = n.array([hdu.header['converged']=='True' for hdu in model[1:]])
			cvm = n.arange(len(model[1:]))[(converged==True)]
			##print(converged, len(cvm))
			if len(cvm)>=1:
				# opens the observation file
				spec_file =os.path.join(os.environ[env], 'spectra', plate, 'spec-'+plate+'-'+mjd+'-'+fiberid+".fits")
				obs = fits.open(spec_file)
				# interpolates the observations and the errors
				wl_data = 10**obs[1].data['loglam']/(1+obs[2].data[z_name])
				fl_data = obs[1].data['flux']
				err_data = obs[1].data['ivar']**(-0.5)
				ok_data = (obs[1].data['ivar']>0)
				spec = interp1d(wl_data[ok_data], fl_data[ok_data])
				err = interp1d(wl_data[ok_data], err_data[ok_data])
				wl_data_max = n.max(wl_data[ok_data])
				wl_data_min = n.min(wl_data[ok_data])
				N_data_points = len(wl_data)
				# loops over the models to get the chi2
				for j_hdu, hdu in enumerate(model[1:]):
					if hdu.header['converged']=='True':
						ok_model = (hdu.data['wavelength']>wl_data_min)&(hdu.data['wavelength']<wl_data_max)
						wl_model = hdu.data['wavelength'][ok_model]
						chi2s=(spec(wl_model)-hdu.data['firefly_model'][ok_model])/err(wl_model)
						chi2[id_spec][j_hdu] =  n.sum(chi2s**2.) 
						ndof[id_spec][j_hdu] =  len(chi2s)-2.

				return 1.
			else:
				return 0.
		else:
			return 0.
	#except ValueError:
		#return 0.
	
for id_spec in range(len(orig_table['MJD'])):
	mjd     = orig_table['MJD'][id_spec]
	fiberid = orig_table['FIBERID'][id_spec] 
	get_spFly(plate, str(mjd).zfill(5), str(fiberid).zfill(4), id_spec)


#############################################
#############################################
# STEP 2
#############################################
#############################################

prihdr = fits.Header()
prihdr['file']   = os.path.basename(path_2_out_file)
prihdr['plate']  = int(plate)
prihdr['models'] = 'Maraston_2011'
#prihdr['library'] = 'MILES'
prihdr['fitter'] = 'FIREFLY'
#prihdr['author'] = 'johan comparat'
prihdr['DR'] = 16


def get_table_entry_full(hduSPM):
	"""
	Convert the results located in the headers of the spFly files into a table.
	Changes the resulting numbers to units
	"""
	if hduSPM.header['converged']=='True':
		prefix = hduSPM.header['IMF'] + "_" + hduSPM.header['MODEL'] + "_"
		table_entry = [
		  1e9*10**hduSPM.header['age_lightW'] 
		, 1e9*10**hduSPM.header['age_lightW_up_1sig']            
		, 1e9*10**hduSPM.header['age_lightW_low_1sig']           
		, 1e9*10**hduSPM.header['age_lightW_up_2sig']            
		, 1e9*10**hduSPM.header['age_lightW_low_2sig']           
		, 1e9*10**hduSPM.header['age_lightW_up_3sig']            
		, 1e9*10**hduSPM.header['age_lightW_low_3sig']           
		,     10**hduSPM.header['metallicity_lightW']            
		,     10**hduSPM.header['metallicity_lightW_up_1sig'] 
		,     10**hduSPM.header['metallicity_lightW_low_1sig']
		,     10**hduSPM.header['metallicity_lightW_up_2sig'] 
		,     10**hduSPM.header['metallicity_lightW_low_2sig']
		,     10**hduSPM.header['metallicity_lightW_up_3sig'] 
		,     10**hduSPM.header['metallicity_lightW_low_3sig']
		, 1e9*10**hduSPM.header['age_massW']                  
		, 1e9*10**hduSPM.header['age_massW_up_1sig']          
		, 1e9*10**hduSPM.header['age_massW_low_1sig']         
		, 1e9*10**hduSPM.header['age_massW_up_2sig']          
		, 1e9*10**hduSPM.header['age_massW_low_2sig']         
		, 1e9*10**hduSPM.header['age_massW_up_3sig']          
		, 1e9*10**hduSPM.header['age_massW_low_3sig']         
		,     10**hduSPM.header['metallicity_massW']          
		,     10**hduSPM.header['metallicity_massW_up_1sig']  
		,     10**hduSPM.header['metallicity_massW_low_1sig'] 
		,     10**hduSPM.header['metallicity_massW_up_2sig']    
		,     10**hduSPM.header['metallicity_massW_low_2sig']   
		,     10**hduSPM.header['metallicity_massW_up_3sig']    
		,     10**hduSPM.header['metallicity_massW_low_3sig']   
		,     10**hduSPM.header['total_mass']                   
		,     10**hduSPM.header['stellar_mass']                 
		,     10**hduSPM.header['living_stars_mass']            
		,     10**hduSPM.header['remnant_mass']                 
		,     10**hduSPM.header['remnant_mass_in_whitedwarfs']  
		,     10**hduSPM.header['remnant_mass_in_neutronstars'] 
		,     10**hduSPM.header['remnant_mass_blackholes']      
		,     10**hduSPM.header['mass_of_ejecta']     
		,     10**hduSPM.header['total_mass_up_1sig'] 
		,     10**hduSPM.header['total_mass_low_1sig']
		,     10**hduSPM.header['total_mass_up_2sig'] 
		,     10**hduSPM.header['total_mass_low_2sig']
		,     10**hduSPM.header['total_mass_up_3sig'] 
		,     10**hduSPM.header['total_mass_low_3sig']
		,         hduSPM.header['EBV']                
		,         hduSPM.header['ssp_number']         
		]
		if hduSPM.header['ssp_number'] >0 :
			ssp_num = hduSPM.header['ssp_number']
		else :
			ssp_num = 0
		#print(ssp_num)

		##print hduSPM.header
		for iii in n.arange(ssp_num):
			#header_listB = n.array([
				#' '+prefix+'total_mass_ssp_'+str(iii)                                     
				#,' '+prefix+'stellar_mass_ssp_'+str(iii)                                   
				#,' '+prefix+'living_stars_mass_ssp_'+str(iii)                              
				#,' '+prefix+'remnant_mass_ssp_'+str(iii)                                   
				#,' '+prefix+'remnant_mass_in_whitedwarfs_ssp_'+str(iii)                    
				#,' '+prefix+'remnant_mass_in_neutronstars_ssp_'+str(iii)                   
				#,' '+prefix+'remnant_mass_in_blackholes_ssp_'+str(iii)                     
				#,' '+prefix+'mass_of_ejecta_ssp_'+str(iii)                                 
				#,' '+prefix+'log_age_ssp_'+str(iii)                                        
				#,' '+prefix+'metal_ssp_'+str(iii)                                          
				#,' '+prefix+'SFR_ssp_'+str(iii)                                            
				#,' '+prefix+'weightMass_ssp_'+str(iii)                                     
				#,' '+prefix+'weightLight_ssp_'+str(iii)   
			#])
			#headerB = "".join(header_listB)
			#print(headerB)
			# values
			table_entry.append( 10**hduSPM.header['total_mass_ssp_'+str(iii)] )
			table_entry.append( 10**hduSPM.header['stellar_mass_ssp_'+str(iii)] )
			table_entry.append( 10**hduSPM.header['living_stars_mass_ssp_'+str(iii)] )
			table_entry.append( 10**hduSPM.header['remnant_mass_ssp_'+str(iii)] )
			table_entry.append( 10**hduSPM.header['remnant_mass_in_whitedwarfs_ssp_'+str(iii)] )
			table_entry.append( 10**hduSPM.header['remnant_mass_in_neutronstars_ssp_'+str(iii)] )
			table_entry.append( 10**hduSPM.header['remnant_mass_in_blackholes_ssp_'+str(iii)] )
			table_entry.append( 10**hduSPM.header['mass_of_ejecta_ssp_'+str(iii)] )
			table_entry.append( 1e9*10**hduSPM.header['log_age_ssp_'+str(iii)] )
			table_entry.append( 10**hduSPM.header['metal_ssp_'+str(iii)] )
			table_entry.append( hduSPM.header['SFR_ssp_'+str(iii)] )
			table_entry.append( hduSPM.header['weightMass_ssp_'+str(iii)] )
			table_entry.append( hduSPM.header['weightLight_ssp_'+str(iii)] )
			# concatenates headers
			#headerA += headerB
		
		if ssp_num<8 :
			for iii in n.arange(ssp_num, 8, 1):
				#header_listB = n.array([
					#' '+prefix+'total_mass_ssp_'+str(iii)                                     
					#,' '+prefix+'stellar_mass_ssp_'+str(iii)                                   
					#,' '+prefix+'living_stars_mass_ssp_'+str(iii)                              
					#,' '+prefix+'remnant_mass_ssp_'+str(iii)                                   
					#,' '+prefix+'remnant_mass_in_whitedwarfs_ssp_'+str(iii)                    
					#,' '+prefix+'remnant_mass_in_neutronstars_ssp_'+str(iii)                   
					#,' '+prefix+'remnant_mass_in_blackholes_ssp_'+str(iii)                     
					#,' '+prefix+'mass_of_ejecta_ssp_'+str(iii)                                 
					#,' '+prefix+'log_age_ssp_'+str(iii)                                        
					#,' '+prefix+'metal_ssp_'+str(iii)                                          
					#,' '+prefix+'SFR_ssp_'+str(iii)                                            
					#,' '+prefix+'weightMass_ssp_'+str(iii)                                     
					#,' '+prefix+'weightLight_ssp_'+str(iii)   
				#])
				#headerB = "".join(header_listB)

				table_entry.append([dV, dV, dV, dV, dV, dV, dV, dV, dV, dV, dV, dV, dV])
				#headerA += headerB

		table_entry = n.array( n.hstack((table_entry)) )
		##print table_entry.shape
		return n.hstack((table_entry))

	else:
		return n.ones(148)*dV

# step 2 : match to the created data set	

table_all = n.ones(( len(orig_table['FIBERID']), 296)) * dV
headers = ""
for index, (fiber, mjd) in enumerate(zip(orig_table['FIBERID'], orig_table['MJD'])):
	spFly_file = os.path.join(os.environ[env], 'firefly', plate, 'spFly-'+plate+'-'+str(mjd).zfill(5)+'-'+str(fiber).zfill(4)+".fits")
	if os.path.isfile(spFly_file):
		print( spFly_file )
		hduSPM=fits.open(spFly_file)
		table_entry_1 = get_table_entry_full( hduSPM[1] )
		table_entry_2 = get_table_entry_full( hduSPM[2] )
		table_all[index] = n.hstack((table_entry_1, table_entry_2))

newDat = n.transpose(table_all)

headers = " Chabrier_MILES_age_lightW Chabrier_MILES_age_lightW_up_1sig Chabrier_MILES_age_lightW_low_1sig Chabrier_MILES_age_lightW_up_2sig Chabrier_MILES_age_lightW_low_2sig Chabrier_MILES_age_lightW_up_3sig Chabrier_MILES_age_lightW_low_3sig Chabrier_MILES_metallicity_lightW Chabrier_MILES_metallicity_lightW_up_1sig Chabrier_MILES_metallicity_lightW_low_1sig Chabrier_MILES_metallicity_lightW_up_2sig Chabrier_MILES_metallicity_lightW_low_2sig Chabrier_MILES_metallicity_lightW_up_3sig Chabrier_MILES_metallicity_lightW_low_3sig Chabrier_MILES_age_massW Chabrier_MILES_age_massW_up_1sig Chabrier_MILES_age_massW_low_1sig Chabrier_MILES_age_massW_up_2sig Chabrier_MILES_age_massW_low_2sig Chabrier_MILES_age_massW_up_3sig Chabrier_MILES_age_massW_low_3sig Chabrier_MILES_metallicity_massW Chabrier_MILES_metallicity_massW_up_1sig Chabrier_MILES_metallicity_massW_low_1sig Chabrier_MILES_metallicity_massW_up_2sig Chabrier_MILES_metallicity_massW_low_2sig Chabrier_MILES_metallicity_massW_up_3sig Chabrier_MILES_metallicity_massW_low_3sig Chabrier_MILES_total_mass Chabrier_MILES_stellar_mass Chabrier_MILES_living_stars_mass Chabrier_MILES_remnant_mass Chabrier_MILES_remnant_mass_in_whitedwarfs Chabrier_MILES_remnant_mass_in_neutronstars Chabrier_MILES_remnant_mass_blackholes Chabrier_MILES_mass_of_ejecta Chabrier_MILES_total_mass_up_1sig Chabrier_MILES_total_mass_low_1sig Chabrier_MILES_total_mass_up_2sig Chabrier_MILES_total_mass_low_2sig Chabrier_MILES_total_mass_up_3sig Chabrier_MILES_total_mass_low_3sig Chabrier_MILES_spm_EBV Chabrier_MILES_nComponentsSSP Chabrier_MILES_total_mass_ssp_0 Chabrier_MILES_stellar_mass_ssp_0 Chabrier_MILES_living_stars_mass_ssp_0 Chabrier_MILES_remnant_mass_ssp_0 Chabrier_MILES_remnant_mass_in_whitedwarfs_ssp_0 Chabrier_MILES_remnant_mass_in_neutronstars_ssp_0 Chabrier_MILES_remnant_mass_in_blackholes_ssp_0 Chabrier_MILES_mass_of_ejecta_ssp_0 Chabrier_MILES_log_age_ssp_0 Chabrier_MILES_metal_ssp_0 Chabrier_MILES_SFR_ssp_0 Chabrier_MILES_weightMass_ssp_0 Chabrier_MILES_weightLight_ssp_0 Chabrier_MILES_total_mass_ssp_1 Chabrier_MILES_stellar_mass_ssp_1 Chabrier_MILES_living_stars_mass_ssp_1 Chabrier_MILES_remnant_mass_ssp_1 Chabrier_MILES_remnant_mass_in_whitedwarfs_ssp_1 Chabrier_MILES_remnant_mass_in_neutronstars_ssp_1 Chabrier_MILES_remnant_mass_in_blackholes_ssp_1 Chabrier_MILES_mass_of_ejecta_ssp_1 Chabrier_MILES_log_age_ssp_1 Chabrier_MILES_metal_ssp_1 Chabrier_MILES_SFR_ssp_1 Chabrier_MILES_weightMass_ssp_1 Chabrier_MILES_weightLight_ssp_1 Chabrier_MILES_total_mass_ssp_2 Chabrier_MILES_stellar_mass_ssp_2 Chabrier_MILES_living_stars_mass_ssp_2 Chabrier_MILES_remnant_mass_ssp_2 Chabrier_MILES_remnant_mass_in_whitedwarfs_ssp_2 Chabrier_MILES_remnant_mass_in_neutronstars_ssp_2 Chabrier_MILES_remnant_mass_in_blackholes_ssp_2 Chabrier_MILES_mass_of_ejecta_ssp_2 Chabrier_MILES_log_age_ssp_2 Chabrier_MILES_metal_ssp_2 Chabrier_MILES_SFR_ssp_2 Chabrier_MILES_weightMass_ssp_2 Chabrier_MILES_weightLight_ssp_2 Chabrier_MILES_total_mass_ssp_3 Chabrier_MILES_stellar_mass_ssp_3 Chabrier_MILES_living_stars_mass_ssp_3 Chabrier_MILES_remnant_mass_ssp_3 Chabrier_MILES_remnant_mass_in_whitedwarfs_ssp_3 Chabrier_MILES_remnant_mass_in_neutronstars_ssp_3 Chabrier_MILES_remnant_mass_in_blackholes_ssp_3 Chabrier_MILES_mass_of_ejecta_ssp_3 Chabrier_MILES_log_age_ssp_3 Chabrier_MILES_metal_ssp_3 Chabrier_MILES_SFR_ssp_3 Chabrier_MILES_weightMass_ssp_3 Chabrier_MILES_weightLight_ssp_3 Chabrier_MILES_total_mass_ssp_4 Chabrier_MILES_stellar_mass_ssp_4 Chabrier_MILES_living_stars_mass_ssp_4 Chabrier_MILES_remnant_mass_ssp_4 Chabrier_MILES_remnant_mass_in_whitedwarfs_ssp_4 Chabrier_MILES_remnant_mass_in_neutronstars_ssp_4 Chabrier_MILES_remnant_mass_in_blackholes_ssp_4 Chabrier_MILES_mass_of_ejecta_ssp_4 Chabrier_MILES_log_age_ssp_4 Chabrier_MILES_metal_ssp_4 Chabrier_MILES_SFR_ssp_4 Chabrier_MILES_weightMass_ssp_4 Chabrier_MILES_weightLight_ssp_4 Chabrier_MILES_total_mass_ssp_5 Chabrier_MILES_stellar_mass_ssp_5 Chabrier_MILES_living_stars_mass_ssp_5 Chabrier_MILES_remnant_mass_ssp_5 Chabrier_MILES_remnant_mass_in_whitedwarfs_ssp_5 Chabrier_MILES_remnant_mass_in_neutronstars_ssp_5 Chabrier_MILES_remnant_mass_in_blackholes_ssp_5 Chabrier_MILES_mass_of_ejecta_ssp_5 Chabrier_MILES_log_age_ssp_5 Chabrier_MILES_metal_ssp_5 Chabrier_MILES_SFR_ssp_5 Chabrier_MILES_weightMass_ssp_5 Chabrier_MILES_weightLight_ssp_5 Chabrier_MILES_total_mass_ssp_6 Chabrier_MILES_stellar_mass_ssp_6 Chabrier_MILES_living_stars_mass_ssp_6 Chabrier_MILES_remnant_mass_ssp_6 Chabrier_MILES_remnant_mass_in_whitedwarfs_ssp_6 Chabrier_MILES_remnant_mass_in_neutronstars_ssp_6 Chabrier_MILES_remnant_mass_in_blackholes_ssp_6 Chabrier_MILES_mass_of_ejecta_ssp_6 Chabrier_MILES_log_age_ssp_6 Chabrier_MILES_metal_ssp_6 Chabrier_MILES_SFR_ssp_6 Chabrier_MILES_weightMass_ssp_6 Chabrier_MILES_weightLight_ssp_6 Chabrier_MILES_total_mass_ssp_7 Chabrier_MILES_stellar_mass_ssp_7 Chabrier_MILES_living_stars_mass_ssp_7 Chabrier_MILES_remnant_mass_ssp_7 Chabrier_MILES_remnant_mass_in_whitedwarfs_ssp_7 Chabrier_MILES_remnant_mass_in_neutronstars_ssp_7 Chabrier_MILES_remnant_mass_in_blackholes_ssp_7 Chabrier_MILES_mass_of_ejecta_ssp_7 Chabrier_MILES_log_age_ssp_7 Chabrier_MILES_metal_ssp_7 Chabrier_MILES_SFR_ssp_7 Chabrier_MILES_weightMass_ssp_7 Chabrier_MILES_weightLight_ssp_7 Chabrier_ELODIE_age_lightW Chabrier_ELODIE_age_lightW_up_1sig Chabrier_ELODIE_age_lightW_low_1sig Chabrier_ELODIE_age_lightW_up_2sig Chabrier_ELODIE_age_lightW_low_2sig Chabrier_ELODIE_age_lightW_up_3sig Chabrier_ELODIE_age_lightW_low_3sig Chabrier_ELODIE_metallicity_lightW Chabrier_ELODIE_metallicity_lightW_up_1sig Chabrier_ELODIE_metallicity_lightW_low_1sig Chabrier_ELODIE_metallicity_lightW_up_2sig Chabrier_ELODIE_metallicity_lightW_low_2sig Chabrier_ELODIE_metallicity_lightW_up_3sig Chabrier_ELODIE_metallicity_lightW_low_3sig Chabrier_ELODIE_age_massW Chabrier_ELODIE_age_massW_up_1sig Chabrier_ELODIE_age_massW_low_1sig Chabrier_ELODIE_age_massW_up_2sig Chabrier_ELODIE_age_massW_low_2sig Chabrier_ELODIE_age_massW_up_3sig Chabrier_ELODIE_age_massW_low_3sig Chabrier_ELODIE_metallicity_massW Chabrier_ELODIE_metallicity_massW_up_1sig Chabrier_ELODIE_metallicity_massW_low_1sig Chabrier_ELODIE_metallicity_massW_up_2sig Chabrier_ELODIE_metallicity_massW_low_2sig Chabrier_ELODIE_metallicity_massW_up_3sig Chabrier_ELODIE_metallicity_massW_low_3sig Chabrier_ELODIE_total_mass Chabrier_ELODIE_stellar_mass Chabrier_ELODIE_living_stars_mass Chabrier_ELODIE_remnant_mass Chabrier_ELODIE_remnant_mass_in_whitedwarfs Chabrier_ELODIE_remnant_mass_in_neutronstars Chabrier_ELODIE_remnant_mass_blackholes Chabrier_ELODIE_mass_of_ejecta Chabrier_ELODIE_total_mass_up_1sig Chabrier_ELODIE_total_mass_low_1sig Chabrier_ELODIE_total_mass_up_2sig Chabrier_ELODIE_total_mass_low_2sig Chabrier_ELODIE_total_mass_up_3sig Chabrier_ELODIE_total_mass_low_3sig Chabrier_ELODIE_spm_EBV Chabrier_ELODIE_nComponentsSSP Chabrier_ELODIE_total_mass_ssp_0 Chabrier_ELODIE_stellar_mass_ssp_0 Chabrier_ELODIE_living_stars_mass_ssp_0 Chabrier_ELODIE_remnant_mass_ssp_0 Chabrier_ELODIE_remnant_mass_in_whitedwarfs_ssp_0 Chabrier_ELODIE_remnant_mass_in_neutronstars_ssp_0 Chabrier_ELODIE_remnant_mass_in_blackholes_ssp_0 Chabrier_ELODIE_mass_of_ejecta_ssp_0 Chabrier_ELODIE_log_age_ssp_0 Chabrier_ELODIE_metal_ssp_0 Chabrier_ELODIE_SFR_ssp_0 Chabrier_ELODIE_weightMass_ssp_0 Chabrier_ELODIE_weightLight_ssp_0 Chabrier_ELODIE_total_mass_ssp_1 Chabrier_ELODIE_stellar_mass_ssp_1 Chabrier_ELODIE_living_stars_mass_ssp_1 Chabrier_ELODIE_remnant_mass_ssp_1 Chabrier_ELODIE_remnant_mass_in_whitedwarfs_ssp_1 Chabrier_ELODIE_remnant_mass_in_neutronstars_ssp_1 Chabrier_ELODIE_remnant_mass_in_blackholes_ssp_1 Chabrier_ELODIE_mass_of_ejecta_ssp_1 Chabrier_ELODIE_log_age_ssp_1 Chabrier_ELODIE_metal_ssp_1 Chabrier_ELODIE_SFR_ssp_1 Chabrier_ELODIE_weightMass_ssp_1 Chabrier_ELODIE_weightLight_ssp_1 Chabrier_ELODIE_total_mass_ssp_2 Chabrier_ELODIE_stellar_mass_ssp_2 Chabrier_ELODIE_living_stars_mass_ssp_2 Chabrier_ELODIE_remnant_mass_ssp_2 Chabrier_ELODIE_remnant_mass_in_whitedwarfs_ssp_2 Chabrier_ELODIE_remnant_mass_in_neutronstars_ssp_2 Chabrier_ELODIE_remnant_mass_in_blackholes_ssp_2 Chabrier_ELODIE_mass_of_ejecta_ssp_2 Chabrier_ELODIE_log_age_ssp_2 Chabrier_ELODIE_metal_ssp_2 Chabrier_ELODIE_SFR_ssp_2 Chabrier_ELODIE_weightMass_ssp_2 Chabrier_ELODIE_weightLight_ssp_2 Chabrier_ELODIE_total_mass_ssp_3 Chabrier_ELODIE_stellar_mass_ssp_3 Chabrier_ELODIE_living_stars_mass_ssp_3 Chabrier_ELODIE_remnant_mass_ssp_3 Chabrier_ELODIE_remnant_mass_in_whitedwarfs_ssp_3 Chabrier_ELODIE_remnant_mass_in_neutronstars_ssp_3 Chabrier_ELODIE_remnant_mass_in_blackholes_ssp_3 Chabrier_ELODIE_mass_of_ejecta_ssp_3 Chabrier_ELODIE_log_age_ssp_3 Chabrier_ELODIE_metal_ssp_3 Chabrier_ELODIE_SFR_ssp_3 Chabrier_ELODIE_weightMass_ssp_3 Chabrier_ELODIE_weightLight_ssp_3 Chabrier_ELODIE_total_mass_ssp_4 Chabrier_ELODIE_stellar_mass_ssp_4 Chabrier_ELODIE_living_stars_mass_ssp_4 Chabrier_ELODIE_remnant_mass_ssp_4 Chabrier_ELODIE_remnant_mass_in_whitedwarfs_ssp_4 Chabrier_ELODIE_remnant_mass_in_neutronstars_ssp_4 Chabrier_ELODIE_remnant_mass_in_blackholes_ssp_4 Chabrier_ELODIE_mass_of_ejecta_ssp_4 Chabrier_ELODIE_log_age_ssp_4 Chabrier_ELODIE_metal_ssp_4 Chabrier_ELODIE_SFR_ssp_4 Chabrier_ELODIE_weightMass_ssp_4 Chabrier_ELODIE_weightLight_ssp_4 Chabrier_ELODIE_total_mass_ssp_5 Chabrier_ELODIE_stellar_mass_ssp_5 Chabrier_ELODIE_living_stars_mass_ssp_5 Chabrier_ELODIE_remnant_mass_ssp_5 Chabrier_ELODIE_remnant_mass_in_whitedwarfs_ssp_5 Chabrier_ELODIE_remnant_mass_in_neutronstars_ssp_5 Chabrier_ELODIE_remnant_mass_in_blackholes_ssp_5 Chabrier_ELODIE_mass_of_ejecta_ssp_5 Chabrier_ELODIE_log_age_ssp_5 Chabrier_ELODIE_metal_ssp_5 Chabrier_ELODIE_SFR_ssp_5 Chabrier_ELODIE_weightMass_ssp_5 Chabrier_ELODIE_weightLight_ssp_5 Chabrier_ELODIE_total_mass_ssp_6 Chabrier_ELODIE_stellar_mass_ssp_6 Chabrier_ELODIE_living_stars_mass_ssp_6 Chabrier_ELODIE_remnant_mass_ssp_6 Chabrier_ELODIE_remnant_mass_in_whitedwarfs_ssp_6 Chabrier_ELODIE_remnant_mass_in_neutronstars_ssp_6 Chabrier_ELODIE_remnant_mass_in_blackholes_ssp_6 Chabrier_ELODIE_mass_of_ejecta_ssp_6 Chabrier_ELODIE_log_age_ssp_6 Chabrier_ELODIE_metal_ssp_6 Chabrier_ELODIE_SFR_ssp_6 Chabrier_ELODIE_weightMass_ssp_6 Chabrier_ELODIE_weightLight_ssp_6 Chabrier_ELODIE_total_mass_ssp_7 Chabrier_ELODIE_stellar_mass_ssp_7 Chabrier_ELODIE_living_stars_mass_ssp_7 Chabrier_ELODIE_remnant_mass_ssp_7 Chabrier_ELODIE_remnant_mass_in_whitedwarfs_ssp_7 Chabrier_ELODIE_remnant_mass_in_neutronstars_ssp_7 Chabrier_ELODIE_remnant_mass_in_blackholes_ssp_7 Chabrier_ELODIE_mass_of_ejecta_ssp_7 Chabrier_ELODIE_log_age_ssp_7 Chabrier_ELODIE_metal_ssp_7 Chabrier_ELODIE_SFR_ssp_7 Chabrier_ELODIE_weightMass_ssp_7 Chabrier_ELODIE_weightLight_ssp_7"


all_cols = []


for data_array, head in zip(newDat, headers.split()):
	all_cols.append(fits.Column(name=head, format='D', array=data_array))

for id_col, (col_chi2, col_ndof) in enumerate(zip(n.transpose(chi2), n.transpose(ndof))):
	all_cols.append(fits.Column(name=hdu_header_prefix[id_col]+"chi2", format='D', array=col_chi2))
	all_cols.append(fits.Column(name=hdu_header_prefix[id_col]+"ndof", format='D', array=col_ndof))


#all_cols.append(fits.Column(name="abs_mag_u_spec", format='D', array=absolute_magnitudes.T[0]))
#all_cols.append(fits.Column(name="abs_mag_g_spec", format='D', array=absolute_magnitudes.T[1]))
#all_cols.append(fits.Column(name="abs_mag_r_spec", format='D', array=absolute_magnitudes.T[2]))
#all_cols.append(fits.Column(name="abs_mag_i_spec", format='D', array=absolute_magnitudes.T[3]))
##all_cols.append(fits.Column(name="abs_mag_z_spec", format='D', array=absolute_magnitudes.T[4]))

#all_cols.append(fits.Column(name="abs_mag_u_noise", format='D', array=absolute_magnitudes_err.T[0]))
#all_cols.append(fits.Column(name="abs_mag_g_noise", format='D', array=absolute_magnitudes_err.T[1]))
#all_cols.append(fits.Column(name="abs_mag_r_noise", format='D', array=absolute_magnitudes_err.T[2]))
#all_cols.append(fits.Column(name="abs_mag_i_noise", format='D', array=absolute_magnitudes_err.T[3]))
##all_cols.append(fits.Column(name="abs_mag_z_noise", format='D', array=absolute_magnitudes_err.T[4]))

#all_cols.append(fits.Column(name="abs_mag_u_model_Chabrier_ELODIE", format='D', array=absolute_magnitudes_cha_elodie.T[0]))
#all_cols.append(fits.Column(name="abs_mag_g_model_Chabrier_ELODIE", format='D', array=absolute_magnitudes_cha_elodie.T[1]))
#all_cols.append(fits.Column(name="abs_mag_r_model_Chabrier_ELODIE", format='D', array=absolute_magnitudes_cha_elodie.T[2]))
#all_cols.append(fits.Column(name="abs_mag_i_model_Chabrier_ELODIE", format='D', array=absolute_magnitudes_cha_elodie.T[3]))

headers = " Chabrier_MILES_age_lightW Chabrier_MILES_age_lightW_up_1sig Chabrier_MILES_age_lightW_low_1sig Chabrier_MILES_age_lightW_up_2sig Chabrier_MILES_age_lightW_low_2sig Chabrier_MILES_age_lightW_up_3sig Chabrier_MILES_age_lightW_low_3sig Chabrier_MILES_metallicity_lightW Chabrier_MILES_metallicity_lightW_up_1sig Chabrier_MILES_metallicity_lightW_low_1sig Chabrier_MILES_metallicity_lightW_up_2sig Chabrier_MILES_metallicity_lightW_low_2sig Chabrier_MILES_metallicity_lightW_up_3sig Chabrier_MILES_metallicity_lightW_low_3sig Chabrier_MILES_age_massW Chabrier_MILES_age_massW_up_1sig Chabrier_MILES_age_massW_low_1sig Chabrier_MILES_age_massW_up_2sig Chabrier_MILES_age_massW_low_2sig Chabrier_MILES_age_massW_up_3sig Chabrier_MILES_age_massW_low_3sig Chabrier_MILES_metallicity_massW Chabrier_MILES_metallicity_massW_up_1sig Chabrier_MILES_metallicity_massW_low_1sig Chabrier_MILES_metallicity_massW_up_2sig Chabrier_MILES_metallicity_massW_low_2sig Chabrier_MILES_metallicity_massW_up_3sig Chabrier_MILES_metallicity_massW_low_3sig Chabrier_MILES_total_mass Chabrier_MILES_stellar_mass Chabrier_MILES_living_stars_mass Chabrier_MILES_remnant_mass Chabrier_MILES_remnant_mass_in_whitedwarfs Chabrier_MILES_remnant_mass_in_neutronstars Chabrier_MILES_remnant_mass_blackholes Chabrier_MILES_mass_of_ejecta Chabrier_MILES_total_mass_up_1sig Chabrier_MILES_total_mass_low_1sig Chabrier_MILES_total_mass_up_2sig Chabrier_MILES_total_mass_low_2sig Chabrier_MILES_total_mass_up_3sig Chabrier_MILES_total_mass_low_3sig Chabrier_MILES_spm_EBV Chabrier_MILES_nComponentsSSP Chabrier_MILES_total_mass_ssp_0 Chabrier_MILES_stellar_mass_ssp_0 Chabrier_MILES_living_stars_mass_ssp_0 Chabrier_MILES_remnant_mass_ssp_0 Chabrier_MILES_remnant_mass_in_whitedwarfs_ssp_0 Chabrier_MILES_remnant_mass_in_neutronstars_ssp_0 Chabrier_MILES_remnant_mass_in_blackholes_ssp_0 Chabrier_MILES_mass_of_ejecta_ssp_0 Chabrier_MILES_log_age_ssp_0 Chabrier_MILES_metal_ssp_0 Chabrier_MILES_SFR_ssp_0 Chabrier_MILES_weightMass_ssp_0 Chabrier_MILES_weightLight_ssp_0 Chabrier_MILES_total_mass_ssp_1 Chabrier_MILES_stellar_mass_ssp_1 Chabrier_MILES_living_stars_mass_ssp_1 Chabrier_MILES_remnant_mass_ssp_1 Chabrier_MILES_remnant_mass_in_whitedwarfs_ssp_1 Chabrier_MILES_remnant_mass_in_neutronstars_ssp_1 Chabrier_MILES_remnant_mass_in_blackholes_ssp_1 Chabrier_MILES_mass_of_ejecta_ssp_1 Chabrier_MILES_log_age_ssp_1 Chabrier_MILES_metal_ssp_1 Chabrier_MILES_SFR_ssp_1 Chabrier_MILES_weightMass_ssp_1 Chabrier_MILES_weightLight_ssp_1 Chabrier_MILES_total_mass_ssp_2 Chabrier_MILES_stellar_mass_ssp_2 Chabrier_MILES_living_stars_mass_ssp_2 Chabrier_MILES_remnant_mass_ssp_2 Chabrier_MILES_remnant_mass_in_whitedwarfs_ssp_2 Chabrier_MILES_remnant_mass_in_neutronstars_ssp_2 Chabrier_MILES_remnant_mass_in_blackholes_ssp_2 Chabrier_MILES_mass_of_ejecta_ssp_2 Chabrier_MILES_log_age_ssp_2 Chabrier_MILES_metal_ssp_2 Chabrier_MILES_SFR_ssp_2 Chabrier_MILES_weightMass_ssp_2 Chabrier_MILES_weightLight_ssp_2 Chabrier_MILES_total_mass_ssp_3 Chabrier_MILES_stellar_mass_ssp_3 Chabrier_MILES_living_stars_mass_ssp_3 Chabrier_MILES_remnant_mass_ssp_3 Chabrier_MILES_remnant_mass_in_whitedwarfs_ssp_3 Chabrier_MILES_remnant_mass_in_neutronstars_ssp_3 Chabrier_MILES_remnant_mass_in_blackholes_ssp_3 Chabrier_MILES_mass_of_ejecta_ssp_3 Chabrier_MILES_log_age_ssp_3 Chabrier_MILES_metal_ssp_3 Chabrier_MILES_SFR_ssp_3 Chabrier_MILES_weightMass_ssp_3 Chabrier_MILES_weightLight_ssp_3 Chabrier_MILES_total_mass_ssp_4 Chabrier_MILES_stellar_mass_ssp_4 Chabrier_MILES_living_stars_mass_ssp_4 Chabrier_MILES_remnant_mass_ssp_4 Chabrier_MILES_remnant_mass_in_whitedwarfs_ssp_4 Chabrier_MILES_remnant_mass_in_neutronstars_ssp_4 Chabrier_MILES_remnant_mass_in_blackholes_ssp_4 Chabrier_MILES_mass_of_ejecta_ssp_4 Chabrier_MILES_log_age_ssp_4 Chabrier_MILES_metal_ssp_4 Chabrier_MILES_SFR_ssp_4 Chabrier_MILES_weightMass_ssp_4 Chabrier_MILES_weightLight_ssp_4 Chabrier_MILES_total_mass_ssp_5 Chabrier_MILES_stellar_mass_ssp_5 Chabrier_MILES_living_stars_mass_ssp_5 Chabrier_MILES_remnant_mass_ssp_5 Chabrier_MILES_remnant_mass_in_whitedwarfs_ssp_5 Chabrier_MILES_remnant_mass_in_neutronstars_ssp_5 Chabrier_MILES_remnant_mass_in_blackholes_ssp_5 Chabrier_MILES_mass_of_ejecta_ssp_5 Chabrier_MILES_log_age_ssp_5 Chabrier_MILES_metal_ssp_5 Chabrier_MILES_SFR_ssp_5 Chabrier_MILES_weightMass_ssp_5 Chabrier_MILES_weightLight_ssp_5 Chabrier_MILES_total_mass_ssp_6 Chabrier_MILES_stellar_mass_ssp_6 Chabrier_MILES_living_stars_mass_ssp_6 Chabrier_MILES_remnant_mass_ssp_6 Chabrier_MILES_remnant_mass_in_whitedwarfs_ssp_6 Chabrier_MILES_remnant_mass_in_neutronstars_ssp_6 Chabrier_MILES_remnant_mass_in_blackholes_ssp_6 Chabrier_MILES_mass_of_ejecta_ssp_6 Chabrier_MILES_log_age_ssp_6 Chabrier_MILES_metal_ssp_6 Chabrier_MILES_SFR_ssp_6 Chabrier_MILES_weightMass_ssp_6 Chabrier_MILES_weightLight_ssp_6 Chabrier_MILES_total_mass_ssp_7 Chabrier_MILES_stellar_mass_ssp_7 Chabrier_MILES_living_stars_mass_ssp_7 Chabrier_MILES_remnant_mass_ssp_7 Chabrier_MILES_remnant_mass_in_whitedwarfs_ssp_7 Chabrier_MILES_remnant_mass_in_neutronstars_ssp_7 Chabrier_MILES_remnant_mass_in_blackholes_ssp_7 Chabrier_MILES_mass_of_ejecta_ssp_7 Chabrier_MILES_log_age_ssp_7 Chabrier_MILES_metal_ssp_7 Chabrier_MILES_SFR_ssp_7 Chabrier_MILES_weightMass_ssp_7 Chabrier_MILES_weightLight_ssp_7 Chabrier_ELODIE_age_lightW Chabrier_ELODIE_age_lightW_up_1sig Chabrier_ELODIE_age_lightW_low_1sig Chabrier_ELODIE_age_lightW_up_2sig Chabrier_ELODIE_age_lightW_low_2sig Chabrier_ELODIE_age_lightW_up_3sig Chabrier_ELODIE_age_lightW_low_3sig Chabrier_ELODIE_metallicity_lightW Chabrier_ELODIE_metallicity_lightW_up_1sig Chabrier_ELODIE_metallicity_lightW_low_1sig Chabrier_ELODIE_metallicity_lightW_up_2sig Chabrier_ELODIE_metallicity_lightW_low_2sig Chabrier_ELODIE_metallicity_lightW_up_3sig Chabrier_ELODIE_metallicity_lightW_low_3sig Chabrier_ELODIE_age_massW Chabrier_ELODIE_age_massW_up_1sig Chabrier_ELODIE_age_massW_low_1sig Chabrier_ELODIE_age_massW_up_2sig Chabrier_ELODIE_age_massW_low_2sig Chabrier_ELODIE_age_massW_up_3sig Chabrier_ELODIE_age_massW_low_3sig Chabrier_ELODIE_metallicity_massW Chabrier_ELODIE_metallicity_massW_up_1sig Chabrier_ELODIE_metallicity_massW_low_1sig Chabrier_ELODIE_metallicity_massW_up_2sig Chabrier_ELODIE_metallicity_massW_low_2sig Chabrier_ELODIE_metallicity_massW_up_3sig Chabrier_ELODIE_metallicity_massW_low_3sig Chabrier_ELODIE_total_mass Chabrier_ELODIE_stellar_mass Chabrier_ELODIE_living_stars_mass Chabrier_ELODIE_remnant_mass Chabrier_ELODIE_remnant_mass_in_whitedwarfs Chabrier_ELODIE_remnant_mass_in_neutronstars Chabrier_ELODIE_remnant_mass_blackholes Chabrier_ELODIE_mass_of_ejecta Chabrier_ELODIE_total_mass_up_1sig Chabrier_ELODIE_total_mass_low_1sig Chabrier_ELODIE_total_mass_up_2sig Chabrier_ELODIE_total_mass_low_2sig Chabrier_ELODIE_total_mass_up_3sig Chabrier_ELODIE_total_mass_low_3sig Chabrier_ELODIE_spm_EBV Chabrier_ELODIE_nComponentsSSP Chabrier_ELODIE_total_mass_ssp_0 Chabrier_ELODIE_stellar_mass_ssp_0 Chabrier_ELODIE_living_stars_mass_ssp_0 Chabrier_ELODIE_remnant_mass_ssp_0 Chabrier_ELODIE_remnant_mass_in_whitedwarfs_ssp_0 Chabrier_ELODIE_remnant_mass_in_neutronstars_ssp_0 Chabrier_ELODIE_remnant_mass_in_blackholes_ssp_0 Chabrier_ELODIE_mass_of_ejecta_ssp_0 Chabrier_ELODIE_log_age_ssp_0 Chabrier_ELODIE_metal_ssp_0 Chabrier_ELODIE_SFR_ssp_0 Chabrier_ELODIE_weightMass_ssp_0 Chabrier_ELODIE_weightLight_ssp_0 Chabrier_ELODIE_total_mass_ssp_1 Chabrier_ELODIE_stellar_mass_ssp_1 Chabrier_ELODIE_living_stars_mass_ssp_1 Chabrier_ELODIE_remnant_mass_ssp_1 Chabrier_ELODIE_remnant_mass_in_whitedwarfs_ssp_1 Chabrier_ELODIE_remnant_mass_in_neutronstars_ssp_1 Chabrier_ELODIE_remnant_mass_in_blackholes_ssp_1 Chabrier_ELODIE_mass_of_ejecta_ssp_1 Chabrier_ELODIE_log_age_ssp_1 Chabrier_ELODIE_metal_ssp_1 Chabrier_ELODIE_SFR_ssp_1 Chabrier_ELODIE_weightMass_ssp_1 Chabrier_ELODIE_weightLight_ssp_1 Chabrier_ELODIE_total_mass_ssp_2 Chabrier_ELODIE_stellar_mass_ssp_2 Chabrier_ELODIE_living_stars_mass_ssp_2 Chabrier_ELODIE_remnant_mass_ssp_2 Chabrier_ELODIE_remnant_mass_in_whitedwarfs_ssp_2 Chabrier_ELODIE_remnant_mass_in_neutronstars_ssp_2 Chabrier_ELODIE_remnant_mass_in_blackholes_ssp_2 Chabrier_ELODIE_mass_of_ejecta_ssp_2 Chabrier_ELODIE_log_age_ssp_2 Chabrier_ELODIE_metal_ssp_2 Chabrier_ELODIE_SFR_ssp_2 Chabrier_ELODIE_weightMass_ssp_2 Chabrier_ELODIE_weightLight_ssp_2 Chabrier_ELODIE_total_mass_ssp_3 Chabrier_ELODIE_stellar_mass_ssp_3 Chabrier_ELODIE_living_stars_mass_ssp_3 Chabrier_ELODIE_remnant_mass_ssp_3 Chabrier_ELODIE_remnant_mass_in_whitedwarfs_ssp_3 Chabrier_ELODIE_remnant_mass_in_neutronstars_ssp_3 Chabrier_ELODIE_remnant_mass_in_blackholes_ssp_3 Chabrier_ELODIE_mass_of_ejecta_ssp_3 Chabrier_ELODIE_log_age_ssp_3 Chabrier_ELODIE_metal_ssp_3 Chabrier_ELODIE_SFR_ssp_3 Chabrier_ELODIE_weightMass_ssp_3 Chabrier_ELODIE_weightLight_ssp_3 Chabrier_ELODIE_total_mass_ssp_4 Chabrier_ELODIE_stellar_mass_ssp_4 Chabrier_ELODIE_living_stars_mass_ssp_4 Chabrier_ELODIE_remnant_mass_ssp_4 Chabrier_ELODIE_remnant_mass_in_whitedwarfs_ssp_4 Chabrier_ELODIE_remnant_mass_in_neutronstars_ssp_4 Chabrier_ELODIE_remnant_mass_in_blackholes_ssp_4 Chabrier_ELODIE_mass_of_ejecta_ssp_4 Chabrier_ELODIE_log_age_ssp_4 Chabrier_ELODIE_metal_ssp_4 Chabrier_ELODIE_SFR_ssp_4 Chabrier_ELODIE_weightMass_ssp_4 Chabrier_ELODIE_weightLight_ssp_4 Chabrier_ELODIE_total_mass_ssp_5 Chabrier_ELODIE_stellar_mass_ssp_5 Chabrier_ELODIE_living_stars_mass_ssp_5 Chabrier_ELODIE_remnant_mass_ssp_5 Chabrier_ELODIE_remnant_mass_in_whitedwarfs_ssp_5 Chabrier_ELODIE_remnant_mass_in_neutronstars_ssp_5 Chabrier_ELODIE_remnant_mass_in_blackholes_ssp_5 Chabrier_ELODIE_mass_of_ejecta_ssp_5 Chabrier_ELODIE_log_age_ssp_5 Chabrier_ELODIE_metal_ssp_5 Chabrier_ELODIE_SFR_ssp_5 Chabrier_ELODIE_weightMass_ssp_5 Chabrier_ELODIE_weightLight_ssp_5 Chabrier_ELODIE_total_mass_ssp_6 Chabrier_ELODIE_stellar_mass_ssp_6 Chabrier_ELODIE_living_stars_mass_ssp_6 Chabrier_ELODIE_remnant_mass_ssp_6 Chabrier_ELODIE_remnant_mass_in_whitedwarfs_ssp_6 Chabrier_ELODIE_remnant_mass_in_neutronstars_ssp_6 Chabrier_ELODIE_remnant_mass_in_blackholes_ssp_6 Chabrier_ELODIE_mass_of_ejecta_ssp_6 Chabrier_ELODIE_log_age_ssp_6 Chabrier_ELODIE_metal_ssp_6 Chabrier_ELODIE_SFR_ssp_6 Chabrier_ELODIE_weightMass_ssp_6 Chabrier_ELODIE_weightLight_ssp_6 Chabrier_ELODIE_total_mass_ssp_7 Chabrier_ELODIE_stellar_mass_ssp_7 Chabrier_ELODIE_living_stars_mass_ssp_7 Chabrier_ELODIE_remnant_mass_ssp_7 Chabrier_ELODIE_remnant_mass_in_whitedwarfs_ssp_7 Chabrier_ELODIE_remnant_mass_in_neutronstars_ssp_7 Chabrier_ELODIE_remnant_mass_in_blackholes_ssp_7 Chabrier_ELODIE_mass_of_ejecta_ssp_7 Chabrier_ELODIE_log_age_ssp_7 Chabrier_ELODIE_metal_ssp_7 Chabrier_ELODIE_SFR_ssp_7 Chabrier_ELODIE_weightMass_ssp_7 Chabrier_ELODIE_weightLight_ssp_7"


all_cols = []
for data_array, head in zip(newDat, headers.split()):
	all_cols.append(fits.Column(name=head, format='D', array=data_array))

for id_col, (col_chi2, col_ndof) in enumerate(zip(n.transpose(chi2), n.transpose(ndof))):
	all_cols.append(fits.Column(name=hdu_header_prefix[id_col]+"chi2", format='D', array=col_chi2))
	all_cols.append(fits.Column(name=hdu_header_prefix[id_col]+"ndof", format='D', array=col_ndof))

#all_cols.append(fits.Column(name="abs_mag_z_model_Chabrier_ELODIE", format='D', array=absolute_magnitudes_cha_elodie.T[4]))

new_cols = fits.ColDefs(all_cols)
tbhdu = fits.BinTableHDU.from_columns(orig_cols + new_cols)


prihdr['ageMin'] = 0  
prihdr['ageMax'] = 20 
prihdr['Zmin']   = 0.001 
prihdr['Zmax']   = 10  

prihdu = fits.PrimaryHDU(header=prihdr)

hdu = fits.HDUList([prihdu, tbhdu])

if os.path.isfile(path_2_out_file):
    os.remove(path_2_out_file)

hdu.writeto(path_2_out_file)

#############################################
#############################################
# STEP 3
#############################################
#############################################

def rewrite_plate_files(hd, mjd):
	path_2_out_file_2 = join( out_dir, "spFlyPlate-"+plate.zfill(4)+"-"+str(mjd).zfill(5)+".fits")
	selection = (hd[1].data['MJD'] == mjd)
	newtbdata = hd[1].data[selection]
	hdu_1 = fits.BinTableHDU(data=newtbdata)
		
	thdulist = fits.HDUList([hd[0], hdu_1])
	if os.path.isfile(path_2_out_file_2 ):
		os.remove(path_2_out_file_2 )
		thdulist.writeto( path_2_out_file_2 )
		return 0.
	else :
		thdulist.writeto( path_2_out_file_2 )
		return 1.

hd = fits.open(path_2_out_file)
mjds = n.array(list(set(hd[1].data['MJD'])))
for mjd in mjds :
	rewrite_plate_files(hd, mjd)

os.remove(path_2_out_file)


