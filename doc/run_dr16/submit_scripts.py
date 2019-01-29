import os
import glob
import numpy as n
#import astropy.io.fits as fits

#d21 = fits.open('/data37s/SDSS/catalogs/ELG_Y1.eboss21.fits')[1].data
#pl21 = n.array(list(set(d21['PLATE']))).astype('str')
#d22 = fits.open('/data37s/SDSS/catalogs/ELG_Y1.eboss22.fits')[1].data
#pl22 = n.array(list(set(d22['PLATE']))).astype('str')
#d23 = fits.open('/data37s/SDSS/catalogs/ELG_Y1.eboss23.fits')[1].data
#pl23 = n.array(list(set(d23['PLATE']))).astype('str')

#plates = n.hstack(( pl21, pl22, pl23 ))

#scripts = n.array([pl+'.sh' for pl in plates ])
scripts = n.array(glob.glob("9*.sh"))
scripts.sort()

for script in scripts:
  os.system('sbatch '+script)

