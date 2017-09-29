import os
from os.path import join
import numpy as n
def writeScript(rootName, plate):
	f=open(rootName+".sh",'w')
	f.write("#!/bin/bash \n")
	f.write("#PBS -l walltime=260:00:00 \n")
	f.write("#PBS -o "+plate+".o.$PBS_JOBID \n")
	f.write("#PBS -e "+plate+".e$PBS_JOBID \n")
	f.write("#PBS -M comparat@mpe.mpg.de \n")
	f.write("module load apps/anaconda/2.4.1 \n")
	f.write("module load apps/python/2.7.8/gcc-4.4.7 \n")
	f.write("export PYTHONPATH=$PYTHONPATH:/users/comparat/firefly_code/python/ \n")
	f.write(" \n")
	f.write("cd /users/comparat/firefly_code/bin_eBOSS \n")
	f.write("python run_stellarpop "+plate+" \n")
	f.write(" \n")
	f.close()


plates = n.loadtxt( join(os.environ['EBOSSDR14_DIR'], "catalogs", "plateNumberList"), unpack=True, dtype='str')
for plate in plates:
	rootName = join(os.environ['HOME'], "batch_v1", plate)
	writeScript(rootName, plate)