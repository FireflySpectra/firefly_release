"""
Postprocessing of firefly's outputs.

Merges the specObjAll with firefly outputs at the plate level.
It creates a fits summary table for each plate - mjd combination

1. creates a figure per spectrum, create table with chi2, ndof for each fit

2. match per plate, add chi2 and ndof.

"""
import time
t0t=time.time()
from os.path import join
import os
import numpy as n
import glob 
import sys 
import astropy.io.fits as fits

import GalaxySpectrumFIREFLY as gs


from scipy.interpolate import interp1d
from scipy.stats import norm as gaussD

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as p

import magnitude_library
photo = magnitude_library.Photo()
#print aa.normDict

import astropy.cosmology as co
import astropy.units as uu
cosmo = co.Planck13

# order of the hdus in the model files
hdu_header_prefix = n.array(["Chabrier_MILES_", "Salpeter_MILES_" , "Kroupa_MILES_"   , "Chabrier_ELODIE_", "Salpeter_ELODIE_", "Kroupa_ELODIE_"  , "Chabrier_STELIB_", "Salpeter_STELIB_", "Kroupa_STELIB_" ])

dV=-9999

env = 'ILLUSTRIS_DIR'

spec_list = n.array(glob.glob(os.path.join(os.environ[env], 'spectra', 'broadband_*.fits')))
spec_list.sort()
spec_ids = n.array([os.path.basename(el).split('_')[-1][:-5] for el in spec_list ])

N_spec = len(spec_list)

#############################################
#############################################
# STEP 1
#############################################
#############################################

dir = 'stellarpop' 
suffix = ".fits"

out_dir = os.path.join(os.environ[env], "catalogs")
im_dir = os.path.join(os.environ[env], dir, 'images')

if os.path.isdir(out_dir)==False:
	os.makedirs(out_dir)
if os.path.isdir(im_dir)==False:
	os.makedirs(im_dir)

# loops over id_spec to create the 

chi2 = n.ones((N_spec,9))*dV
ndof = n.ones((N_spec,9))*dV
absolute_magnitudes = n.ones((N_spec,5))*dV
absolute_magnitudes_err = n.ones((N_spec,5))*dV

def plot_spec_spFly(num, id_spec):
	"""
	create a plot per spectrum and computes the chi2 and number of degrees of freedom for each model.
	
	"""
	spFly_file = os.path.join(os.environ[env], 'stellarpop', 'spFly-broadband_'+id_spec+".fits")
	if os.path.isfile(spFly_file) :
		print (spFly_file)
		# defines the output paths
		im_file = 'spFly-'+id_spec+'.png'
		path_2_im_file = os.path.join(im_dir, im_file)
		
		# opens the model file with 9 model hdus
		model = fits.open(spFly_file)
		
		converged = n.array([hdu.header['converged']=='True' for hdu in model[1:]])
		cvm = n.arange(9)[(converged==True)]
		#print(converged, len(cvm))
		if len(cvm)>=1:
			# opens the observation file
			path_2_spec = os.path.join(os.environ[env], 'spectra', 'broadband_'+id_spec+".fits")
			spec=gs.GalaxySpectrumFIREFLY(path_2_spec, milky_way_reddening=False)
			spec.openILLUSTRISsimulatedSpectrum()
			# interpolates the observations and the errors
			wl_data = spec.restframe_wavelength
			fl_data = spec.flux
			err_data = spec.error
			ok_data = (err_data>0)
			spec = interp1d(wl_data[ok_data], fl_data[ok_data])
			err = interp1d(wl_data[ok_data], err_data[ok_data])
			wl_data_max = n.max(wl_data[ok_data])
			wl_data_min = n.min(wl_data[ok_data])
			N_data_points = len(wl_data)
			
			distMod = cosmo.distmod( 0.01 ).value 
			absolute_magnitudes[num], arr = photo.computeMagnitudes(interp1d(wl_data, fl_data * 10. **(-17.)), distMod) 
			absolute_magnitudes_err[num], arr = photo.computeMagnitudes(interp1d(wl_data, err_data * 10. **(-17.)), distMod) 
			print("AbsMag", absolute_magnitudes[num]) 
			print("err", 10**-abs(absolute_magnitudes[num]- absolute_magnitudes_err[num])*10**(absolute_magnitudes[num])) 
			# now creates the figure  
			fig = p.figure(0, figsize = (7, 10), frameon=False)#, tight_layout=True)
			rect = 0.2, 0.15, 0.85, 0.95
			#ax = fig.add_axes(rect, frameon=False)

			# panel with the observed spectrum
			fig.add_subplot(3,1,1)
			p.plot(wl_data[::2], fl_data[::2], 'k', rasterized =True, alpha=0.5)
			p.yscale('log')
			mean_data = n.median(fl_data)
			p.ylim((mean_data/8., mean_data*8.))
			p.xlabel('Wavelength [Angstrom, rest frame]')
			p.ylabel(r'Flux [$f_\lambda$ $10^{-17}$ erg/cm2/s/A]')
			p.title("id=" + id_spec + ", z=0.01")
			for j_hdu, hdu in enumerate(model[1:]):
				if hdu.header['converged']=='True':
					p.plot(hdu.data['wavelength'], hdu.data['firefly_model'], rasterized =True, alpha=0.5, label=hdu.header['IMF']+" "+hdu.header['library'])

			
			# second panel distribution of residuals
			fig.add_subplot(3,1,2)
			# loops over the models
			for j_hdu, hdu in enumerate(model[1:]):
				#print( hdu.header, hdu.data )
				#print( hdu.header['IMF']+"_"+hdu.header['library'] )
				if hdu.header['converged']=='True':
					ok_model = (hdu.data['wavelength']>wl_data_min)&(hdu.data['wavelength']<wl_data_max)
					wl_model = hdu.data['wavelength'][ok_model]
					#p.plot(wl_model, (spec(wl_model)-hdu.data['firefly_model'][ok_model])/err(wl_model), 'k', rasterized =True, alpha=0.5)
					#print "model", hdu.data['firefly_model'][ok_model]
					#print "obs",spec(wl_model)
					chi2s=(spec(wl_model)-hdu.data['firefly_model'][ok_model])/err(wl_model)
					p.hist(chi2s, bins = n.arange(-2,2,0.1), normed = True, histtype='step', label=hdu.header['IMF']+" "+hdu.header['library']+", EBV="+str(n.round(hdu.header['EBV'],3))+r", $\chi^2=$"+str(n.round(n.sum(chi2s**2.)/(len(chi2s)-2.),2)))
					p.ylim((-0.02,1.02))
					#p.yscale('log')
					p.xlabel('(data-model)/error')
					p.ylabel('Normed distribution')
					chi2[num][j_hdu] =  n.sum(chi2s**2.) 
					ndof[num][j_hdu] =  len(chi2s)-2.
								
			p.plot(n.arange(-2,2,0.005), gaussD.pdf(n.arange(-2,2,0.005)), 'k--', label=r'N(0,1)', lw=0.5)
			p.grid()
			p.legend(frameon=False, loc=0, fontsize=8)

			# panel with stellar mass vs. age for the .
			fig.add_subplot(3,1,3)
			# creates the data 
			tpl = n.transpose(n.array([ [
				1e9*10**hdu.header['age_lightW'],
				10**hdu.header['stellar_mass'],
				1e9*10**hdu.header['age_lightW_up']-1e9*10**hdu.header['age_lightW'], 
				1e9*10**hdu.header['age_lightW']-1e9*10**hdu.header['age_lightW_low'],
				10**hdu.header['stellar_mass_up']-10**hdu.header['stellar_mass'],
				10**hdu.header['stellar_mass']-10**hdu.header['stellar_mass_low']]
				for hdu in n.array(model[1:])[cvm] ]))

			
			p.errorbar(tpl[0], tpl[1], xerr=[tpl[2], tpl[3]], yerr=[tpl[4], tpl[5]], barsabove=True, fmt='o')
			#p.axvline(prihdr['age_universe'], color='r', ls='dashed')
			# sorts models according to stellar mass (y-axis coordinate)
			idsUP = n.argsort(tpl[1])

			iterList = n.array(model[1:])[cvm][idsUP]
			for jj, hdu in enumerate(iterList):
				p.annotate(hdu.header['IMF']+" "+hdu.header['library']+r", $\log(Z/Z_\odot)=$"+str(n.round(hdu.header['metallicity_lightW'],4)), 
						xy = (1e9*10**hdu.header['age_lightW'], 10**hdu.header['stellar_mass']), xycoords='data', 
						xytext=(0.85, (jj+0.5)/len(iterList)), textcoords='axes fraction', 
						arrowprops=dict(facecolor='black', shrink=0.05, width=0.2, headwidth=3), 
						horizontalalignment='right', verticalalignment='top', fontsize=9)

			p.ylabel(r'$M/[M_\odot]$')
			p.xlabel(r'$age/[yr]$')
			p.xscale('log')
			p.yscale('log')
			p.xlim((1e8, 1e10))
			p.ylim((1e8, 10**12))
			p.grid()
			p.savefig(path_2_im_file)
			p.clf()
			return 1.
		else:
			return 0.
	else:
		return 0.

for num, id_spec in enumerate(spec_ids):
	plot_spec_spFly(num, id_spec)

print "---------------------------------------------------"
print "step2"
print "---------------------------------------------------"
#############################################
#############################################
# STEP 2
#############################################
#############################################
	
path_2_out_file = join( out_dir, "summary_illustris.fits")

def get_table_entry_full(hduSPM):
	# print "gets entry"
	hduSPM.header
	prefix = hduSPM.header['IMF'] + "_" + hduSPM.header['library'] + "_"
	#print prefix
	headerA =" "+prefix+"age_lightW "+prefix+"age_lightW_up "+prefix+"age_lightW_low "+prefix+"metallicity_lightW "+prefix+"metallicity_lightW_up "+prefix+"metallicity_lightW_low "+prefix+"age_massW "+prefix+"age_massW_up "+prefix+"age_massW_low "+prefix+"metallicity_massW "+prefix+"metallicity_massW_up "+prefix+"metallicity_massW_low "+prefix+"stellar_mass "+prefix+"stellar_mass_up "+prefix+"stellar_mass_low "+prefix+"spm_EBV "+prefix+"nComponentsSSP "
	
	table_entry = [
	  1e9*10**hduSPM.header['age_lightW']          
	, 1e9*10**hduSPM.header['age_lightW_up']       
	, 1e9*10**hduSPM.header['age_lightW_low']      
	, 10**hduSPM.header['metallicity_lightW']  
	, 10**hduSPM.header['metallicity_lightW_up']
	, 10**hduSPM.header['metallicity_lightW_low']
	, 1e9*10**hduSPM.header['age_massW']           
	, 1e9*10**hduSPM.header['age_massW_up']        
	, 1e9*10**hduSPM.header['age_massW_low']       
	, 10**hduSPM.header['metallicity_massW']   
	, 10**hduSPM.header['metallicity_massW_up']
	, 10**hduSPM.header['metallicity_massW_low']
	, 10**hduSPM.header['stellar_mass']        
	, 10**hduSPM.header['stellar_mass_up']     
	, 10**hduSPM.header['stellar_mass_low']    
	, hduSPM.header['EBV'] 
	, hduSPM.header['ssp_number']
	]
	
	if hduSPM.header['ssp_number'] >0 :
		ssp_num = hduSPM.header['ssp_number']
	else :
		ssp_num = 0
	
	#print hduSPM.header
	for iii in n.arange(ssp_num):
		table_entry.append( 10**hduSPM.header['stellar_mass_ssp_'+str(iii)] )
		table_entry.append( 1e9*10**hduSPM.header['age_ssp_'+str(iii)] )
		table_entry.append( 10**hduSPM.header['metal_ssp_'+str(iii)] )
		table_entry.append( hduSPM.header['weightMass_ssp_'+str(iii)] )
		table_entry.append( hduSPM.header['weightLight_ssp_'+str(iii)] )
		headerA += ' '+prefix+'stellar_mass_ssp_'+str(iii) + ' '+prefix+'age_ssp_'+str(iii) + ' '+prefix+'metal_ssp_'+str(iii) + ' '+prefix+'weightMass_ssp_'+str(iii) + ' '+prefix+'weightLight_ssp_'+str(iii)
	
	if ssp_num<8 :
		for iii in n.arange(ssp_num, 8, 1):
			table_entry.append([dV, dV, dV, dV, dV])
			headerA += ' '+prefix+'stellar_mass_ssp_'+str(iii) + ' '+prefix+'age_ssp_'+str(iii) + ' '+prefix+'metal_ssp_'+str(iii) + ' '+prefix+'weightMass_ssp_'+str(iii) + ' '+prefix+'weightLight_ssp_'+str(iii)

	table_entry = n.array( n.hstack((table_entry)) )
	#print table_entry.shape
	return n.hstack((table_entry)), headerA
	
# step 2 : match to thecreated data set	

table_all = n.ones(( N_spec, 513)) * dV
headers = ""
for index, id_spec in enumerate(spec_ids):
	fitFile = join( os.environ[env], dir, "spFly-broadband_"+id_spec+".fits")
	if os.path.isfile(fitFile):
		print fitFile
		hduSPM=fits.open(fitFile)
		
		table_entry_1, headers_1 = get_table_entry_full( hduSPM[1] )
		table_entry_2, headers_2 = get_table_entry_full( hduSPM[2] )
		table_entry_3, headers_3 = get_table_entry_full( hduSPM[3] )

		table_entry_4, headers_4 = get_table_entry_full( hduSPM[4] )
		table_entry_5, headers_5 = get_table_entry_full( hduSPM[5] )
		table_entry_6, headers_6 = get_table_entry_full( hduSPM[6] )

		table_entry_7, headers_7 = get_table_entry_full( hduSPM[7] )
		table_entry_8, headers_8 = get_table_entry_full( hduSPM[8] )
		table_entry_9, headers_9 = get_table_entry_full( hduSPM[9] )

		headers = headers_1 + headers_2 + headers_3 + headers_4 + headers_5 + headers_6 + headers_7 + headers_8 + headers_9
		table_all[index] = n.hstack((table_entry_1, table_entry_2, table_entry_3, table_entry_4, table_entry_5, table_entry_6, table_entry_7, table_entry_8, table_entry_9))
		#print len(table_all[-1])
		fitFileLast = fitFile
	#else:
		#table_all.append(n.ones(57*9)*dV)

newDat = n.transpose(table_all)

all_cols = [fits.Column(name="spec_id", format='K', array=spec_ids)]
for data_array, head in zip(newDat, headers.split()):
	all_cols.append(fits.Column(name=head, format='D', array=data_array))

for id_col, (col_chi2, col_ndof) in enumerate(zip(n.transpose(chi2), n.transpose(ndof))):
	all_cols.append(fits.Column(name=hdu_header_prefix[id_col]+"chi2", format='D', array=col_chi2))
	all_cols.append(fits.Column(name=hdu_header_prefix[id_col]+"ndof", format='D', array=col_ndof))


all_cols.append(fits.Column(name="abs_mag_u_spec", format='D', array=absolute_magnitudes.T[0]))
all_cols.append(fits.Column(name="abs_mag_g_spec", format='D', array=absolute_magnitudes.T[1]))
all_cols.append(fits.Column(name="abs_mag_r_spec", format='D', array=absolute_magnitudes.T[2]))
all_cols.append(fits.Column(name="abs_mag_i_spec", format='D', array=absolute_magnitudes.T[3]))
#all_cols.append(fits.Column(name="abs_mag_z_spec", format='D', array=absolute_magnitudes.T[4]))

all_cols.append(fits.Column(name="abs_mag_u_noise", format='D', array=absolute_magnitudes_err.T[0]))
all_cols.append(fits.Column(name="abs_mag_g_noise", format='D', array=absolute_magnitudes_err.T[1]))
all_cols.append(fits.Column(name="abs_mag_r_noise", format='D', array=absolute_magnitudes_err.T[2]))
all_cols.append(fits.Column(name="abs_mag_i_noise", format='D', array=absolute_magnitudes_err.T[3]))
#all_cols.append(fits.Column(name="abs_mag_z_noise", format='D', array=absolute_magnitudes_err.T[4]))

#all_cols.append(fits.Column(name="abs_mag_u_model_Chabrier_ELODIE", format='D', array=absolute_magnitudes_cha_elodie.T[0]))
#all_cols.append(fits.Column(name="abs_mag_g_model_Chabrier_ELODIE", format='D', array=absolute_magnitudes_cha_elodie.T[1]))
#all_cols.append(fits.Column(name="abs_mag_r_model_Chabrier_ELODIE", format='D', array=absolute_magnitudes_cha_elodie.T[2]))
#all_cols.append(fits.Column(name="abs_mag_i_model_Chabrier_ELODIE", format='D', array=absolute_magnitudes_cha_elodie.T[3]))
#all_cols.append(fits.Column(name="abs_mag_z_model_Chabrier_ELODIE", format='D', array=absolute_magnitudes_cha_elodie.T[4]))

new_cols = fits.ColDefs(all_cols)
tbhdu = fits.BinTableHDU.from_columns(new_cols)

sp_last=fits.open(fitFileLast)[0]

prihdr = fits.Header()

prihdr['author'] = "JC"
prihdr['ageMin'] = sp_last.header['ageMin']
prihdr['ageMax'] = sp_last.header['ageMax']
prihdr['Zmin']   = sp_last.header['Zmin']
prihdr['Zmax']   = sp_last.header['Zmax']

prihdu = fits.PrimaryHDU(header=prihdr)

hdu = fits.HDUList([prihdu, tbhdu])

if os.path.isfile(path_2_out_file):
    os.remove(path_2_out_file)

hdu.writeto(path_2_out_file)

