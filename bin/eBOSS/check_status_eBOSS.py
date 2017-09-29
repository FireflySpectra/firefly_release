
from os.path import join
import os
import numpy as n
import glob 
import sys 
import time
import astropy.io.fits as fits

env = os.environ['EBOSSDR14_DIR']
path_2_file = join(env, "catalogs", "spAll-v5_10_0.fits")

data = fits.open(path_2_file)[1].data

selection = (data['ZWARNING_NOQSO']==0) & (data['CLASS_NOQSO']=="GALAXY") & (data['Z_NOQSO'] > data['Z_ERR_NOQSO']) & (data['Z_ERR_NOQSO']>0) 

bds = n.arange(0,len(data),100000)
for jj in n.arange(0,len(data)+100000,100000)[:-1]:
	out_file = os.environ['DATA_DIR'], 'status', 'status-eBOSS-'+str(bds[jj])+'.txt'
	f = open(out_file, 'w')
	for ii, el in enumerate(data[bds[jj]:bds[jj+1]]) :
		plate = str(int(el['PLATE'])).zfill(4)
		mjd = str(int(el['MJD']))
		fiber = str(int(el['FIBERID'])).zfill(4)
		flyName = "spFly-"+plate+"-"+mjd+"-"+fiber
		exist = 1. * os.path.isfile(join(env, "spectra", plate, flyName + ".fits"))
		done = 1. * os.path.isfile(join(env, "stellarpop", plate, flyName + ".fits"))
		tow = n.vstack((el['PLATE'], el['MJD'], el['FIBERID'], exist, done)).astype('int')
		f.write(tow)
		f.write('\n')
		
	f.close()

