import os
from os.path import join
import numpy as n
def writeScript(rootName, mask):
	f=open(rootName+".sh",'w')
	f.write("#!/bin/bash \n")
	f.write("#PBS -l walltime=260:00:00 \n")
	f.write("#PBS -o "+mask+".o.$PBS_JOBID \n")
	f.write("#PBS -e "+mask+".e$PBS_JOBID \n")
	f.write("#PBS -M comparat@mpe.mpg.de \n")
	f.write("module load apps/anaconda/2.4.1 \n")
	f.write("module load apps/python/2.7.8/gcc-4.4.7 \n")
	f.write("export PYTHONPATH=$PYTHONPATH:/users/comparat/firefly_code/python/ \n")
	f.write(" \n")
	f.write("cd /users/comparat/firefly_code/bin_DEEP2 \n")
	f.write("python run_stellarpop_deep2 "+mask+" \n")
	f.write(" \n")
	f.close()


masks = n.loadtxt( join(os.environ['DEEP2_DIR'], "catalogs", "maskList"), unpack=True, dtype='str')
for mask in masks:
	rootName = join(os.environ['HOME'], "batch_deep2", mask)
	writeScript(rootName, mask)