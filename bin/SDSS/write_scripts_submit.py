import os
from os.path import join
import glob
import numpy as n

def checkdir(indir):
	if os.path.isdir(indir)==False:
		os.mkdir(indir)		

def writeScript(rootName,indir,outdir,oeroot):
	f=open(rootName+".sh",'w')
	f.write("#!/bin/bash \n")
	f.write("#PBS -l walltime=260:00:00 \n")
	f.write("#PBS -o "+oeroot+".o.$PBS_JOBID \n")
	f.write("#PBS -e "+oeroot+".e$PBS_JOBID \n")
	f.write("module load apps/anaconda/2.4.1 \n")
	f.write("module load apps/python/2.7.8/gcc-4.4.7 \n")
	f.write(" \n")
	f.write("python run_stellarpop.py "+indir+" "+outdir+" \n")
	f.write(" \n")
	f.close()

#--------------------------
write_scripts = True
submit_scripts = False
#-------------------------

# Directory with input files 
indir = join(os.environ['FF_DIR'],'data/example_data/spectra/') ; checkdir(indir)
print 'Input: ',indir

# Directory for scripts to be submitted
jobsdir = join(os.environ['FF_DIR'],'jobs/') ; checkdir(jobsdir)
print 'Run scripts: ',jobsdir

# Directory for output files
outdir = join(os.environ['FF_DIR'],'output/')  ; checkdir(outdir)
print 'Output: ',outdir

# Directory for log files
oedir = outdir+'oe/' ; checkdir(oedir)
print 'Log files: ',oedir

if write_scripts:
	path2plates = glob.glob(indir+'*')
	for path2plate in path2plates:
		plate = path2plate.split('spectra/')[1]
		proot = 'plate'+plate

		outplate = outdir+plate+'/' ; checkdir(outplate)

		writeScript(jobsdir+proot,indir+plate+'/',outplate,oedir+proot)

if submit_scripts:
	list_of_jobscripts = glob.glob(jobsdir+'*')

	for i in range(len(list_of_jobscripts)):
		os.system("cd "+jobsdir)
		os.system("qsub "+str(list_of_jobscripts[i]))
		print(str(list_of_jobscripts[i]).replace(jobsdir,'')+' has been submitted')
