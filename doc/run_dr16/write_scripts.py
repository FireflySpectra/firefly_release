import sys, os
from os.path import join
import glob
import numpy as n

run_dir = '/home/comparat/software/linux/FireflySpectra/firefly_dev/doc/run_dr16/'

def writeScript(plate, spec_files):
  f=open(run_dir + plate+".sh",'w')
  f.write("#!/bin/bash \n")
  f.write("#SBATCH --partition=he2srvHighP \n")
  f.write("#SBATCH --time=2000:00:00 \n")
  f.write("#SBATCH --nodes=1 \n")
  f.write("#SBATCH --ntasks=1 \n")
  f.write("#SBATCH --cpus-per-task=1 \n")
  f.write("#SBATCH --job-name="+plate+"-ff \n")
  f.write(" \n")
  f.write(". /home_local/4FSOpsim/py36he2srv/bin/activate \n")
  f.write("export OMP_NUM_THREADS=1 \n")
  f.write(" \n")
  f.write("cd /home/comparat/software/linux/FireflySpectra/firefly_dev/doc \n")
  f.write(" \n")
  for spec_file in spec_files:
    f.write("python one_spectra.py "+spec_file+" \n")
  f.write(" \n")
  f.close()

plates = n.loadtxt('/home/comparat/software/linux/firefly_release/doc/run_dr16/plate_list', unpack=True).astype('int').astype('str')

for plate in plates:
  spec_files = n.array(glob.glob(os.path.join(os.environ['EBOSSDR16_DIR'], plate, '*.fits')))
  spec_files.sort()
  writeScript(plate, spec_files)


