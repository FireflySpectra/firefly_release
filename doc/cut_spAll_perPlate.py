"""
Cuts the spAll file into plates

"""
import time
from os.path import join
import sys, os
import numpy as n
import astropy.io.fits as fits

env ='EBOSSDR16_DIR'

# input catalog : spall
in_cat = join( os.environ[env], "catalogs","spAll-v5_11_0.fits")

# plate list 
plates = n.loadtxt('/home/comparat/software/linux/firefly_release/doc/run_dr16/plate_list', unpack=True).astype('int').astype('str')

plate = plates[0]

for plate in plates:
	# output catalog
	t0t=time.time()
	out_cat = join( os.environ[env], "catalogs", "perPlate", "sp-"+plate.zfill(4)+".fits")
	p0 = """stilts tpipe ifmt=fits in="""+in_cat
	p1 = """ cmd='select "PLATE=="""+plate+""""'"""
	p2 = """ omode=out ofmt=fits out="""+out_cat
	command = p0+p1+p2
	os.system(command)
	print(plate, time.time()-t0t)

