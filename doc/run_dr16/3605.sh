#!/bin/bash 
#SBATCH --time=2000:00:00 
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1 
#SBATCH --job-name=3605-ff 
 
. /home_local/4FSOpsim/py36he2srv/bin/activate 
export OMP_NUM_THREADS=1 
 
cd /home/comparat/software/linux/FireflySpectra/firefly_dev/doc 
 
 
