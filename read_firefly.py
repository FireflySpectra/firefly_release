
from astropy.io import fits
import matplotlib.pyplot as py
import numpy as np
import sys

hdul = fits.open(sys.argv[1])
data=hdul[1].data
wave = data['wavelength']
flux = data['original_data']
model = data['firefly_model']
hdul.close()
hdul.info()

csp_age=np.ndarray(hdul[1].header['ssp_number'])
csp_Z=np.ndarray(hdul[1].header['ssp_number'])
csp_light=np.ndarray(hdul[1].header['ssp_number'])
csp_mass=np.ndarray(hdul[1].header['ssp_number'])
for i in range(len(csp_age)):
	csp_age[i]=hdul[1].header['log_age_ssp_'+str(i)]
	csp_Z[i]=hdul[1].header['metal_ssp_'+str(i)]
	csp_light[i]=hdul[1].header['weightLight_ssp_'+str(i)]
	csp_mass[i]=hdul[1].header['weightMass_ssp_'+str(i)]

print()
print(hdul[0].header)
print(hdul[1].header)
print()
print('age: '+str(np.around(10**hdul[1].header['age_lightW'],decimals=2))+' Gyr')
print('[Z/H]: '+str(np.around(hdul[1].header['metallicity_lightW'],decimals=2))+' dex')
print('log M/Msun: '+str(np.around(hdul[1].header['stellar_mass'],decimals=2)))
print('E(B-V): '+str(np.around(hdul[1].header['EBV'],decimals=2))+' mag')

py.plot(wave,flux)
py.plot(wave,model)
py.show()

fig1=py.figure()
py.xlim(0,15)
py.xlabel('lookback time (Gyr)')
py.ylabel('frequency')
#py.bar(10**(csp_age),csp_light,width=1,align='center',edgecolor='k',linewidth=2)
py.bar(10**(csp_age),csp_light,width=1,align='center',alpha=0.5)
py.scatter(10**(csp_age),csp_light)

fig2=py.figure()
py.xlim(-2,0.5)
py.xlabel('[Z/H] (dex)')
py.ylabel('frequency')
#py.bar(10**(csp_age),csp_light,width=1,align='center',edgecolor='k',linewidth=2)
py.bar(csp_Z,csp_light,width=0.1,align='center',alpha=0.5)
py.scatter(csp_Z,csp_light)
py.show()
