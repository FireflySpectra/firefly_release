import numpy as n
import os
import glob

todolist = n.array(glob.glob(os.path.join(os.environ['DATA_DIR'],'status','DNF_*')))
todolist.sort()

plate_list = n.array([os.path.basename(el).split('_')[-1] for el in todolist])

status_list = n.array([os.path.join(os.environ['DATA_DIR'],'status','status-'+os.path.basename(el).split('_')[-1]+'.txt') for el in todolist])

for st in status_list:
	plate, mjd, fiberid, is_spec, is_spm = n.loadtxt(st, unpack=True)
	#to_download = is_spec==0
	#print(fiberid[to_download])
	to_rerun = is_spm==0
	if len(plate[to_rerun])>5:
		print("to rerun",st,len(plate[to_rerun]))

	for pl, mj, fi in zip(plate[to_rerun], mjd[to_rerun], fiberid[to_rerun]):
		print( pl, mj, fi)

[u0936736@eboss 1379]$ rsync -avz spec*fits comparat@login5.sciama.icg.port.ac.uk:/mnt/lustre/sdss-dr12/eBOSS-DR14/spectra/6037/

	plate, mjd, fiberid, is_spec, is_spm = n.loadtxt(status_list[99], unpack=True)

bad_list=[]
for st in status_list:
	plate, mjd, fiberid, is_spec, is_spm = n.loadtxt(st, unpack=True)
	try :
		print len(plate)
except(TypeError):
	print plate
	bad_list.append(int(plate))

	if len(plate)<5:
		print("to rerun",st,len(plate))

import astropy.io.fits as fits

checklist = n.array(glob.glob(os.path.join(os.environ['SDSSDR12_DIR'],'stellarpop', '????','spFlyPlate-*.fits')))
checklist.sort()

for el in checklist:
	if os.path.getsize(el)<3585600:
		os.remove(el)



		, os.path.getsize(el))
	
	, len(fits.open(el)[1].data[0]))


checklist = n.array(glob.glob(os.path.join(os.environ['EBOSSDR14_DIR'],'stellarpop', '????','spFlyPlate-*.fits')))
checklist.sort()

checkplate = n.array(glob.glob(os.path.join(os.environ['EBOSSDR14_DIR'],'stellarpop', '????')))
checkplate.sort()

for pl in checkplate:
	if len(glob.glob(os.path.join(pl,'spFlyPlate-*.fits'))) == 0 :
		print "qsub ",os.path.basename(pl)+".sh" 

for el in checklist:
	if os.path.getsize(el)<3585600:
		print el, os.path.getsize(el)
		
for el in checklist:
	print el, os.path.getsize(el)
		


checkplate = n.array(glob.glob(os.path.join(os.environ['SDSSDR12_DIR'],'stellarpop', '????')))
checkplate.sort()

for pl in checkplate:
	if len(glob.glob(os.path.join(pl,'spFlyPlate-*.fits'))) == 0 :
		print "qsub ",os.path.basename(pl)+".sh" 
