#star_formation_rate.py

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from IPython.core.debugger import Tracer
import warnings

def star_formation_rate(age_in,prob_in):

	"""
	Takes ages and marginalised-in-age likelihoods
	and fits multiple Gaussians.
	Then estimates fractional SF in last bins.
	Age in log(yrs)
	"""

	warnings.filterwarnings("error")

	age_in  = np.arange(6.5,10.2,0.1)
	prob_in = np.exp(-(age_in - 7.5)**2.0)
	#prob_in = np.random.rand(len(age_in))


	age = np.zeros(np.size(age_in)+10)
	age[:-10]   = age_in
	age[-10:-5] = np.arange(np.max(age_in)+0.1,np.max(age_in+0.5)+0.1,0.1)
	age[-5 :] = np.arange(np.min(age_in)-0.6,np.min(age_in)-0.1,0.1)
	prob = np.zeros(np.size(age_in)+10)
	prob[:-10]  = prob_in
	age = age_in
	prob = prob_in

	prob = prob / np.max(prob)

	plt.plot(age,prob,'k',linewidth=2.0)
	plt.draw()
	
	def func(x, *params):
		y = np.zeros_like(x)
		for i in range(0, len(params), 3):
			ctr = params[i]
			amp = params[i+1]
			wid = params[i+2]
			y = y + amp * np.exp( -((x - ctr)/wid)**2)

		return y

	# Peak guess is position, amplitude, e-width
	guess = np.array([7.0,  1.0, 1.0,\
			 8.0,  1.0, 1.0,\
			 9.0,  1.0, 1.0,\
			 10.0, 1.0, 1.0])

	def loop_curve_fit(func,age,prob,guess,loopnum):
		try:
			popt, pcov = curve_fit(func, age, prob, p0=guess,absolute_sigma=False)
			return popt,pcov,loopnum
		except (RuntimeError,Warning) as e:

			if loopnum==1:
				age_inter = np.zeros(np.size(age)+20)
				prob_inter = np.zeros(np.size(age)+20)
				prob_big = np.zeros(np.size(age)+20)
				age_inter[:10] 	= np.arange(np.min(age)-1.0,np.min(age),0.1)
				age_inter[-10:] 	= np.arange(np.max(age)+0.1,np.max(age)+1.1,0.1)
				age_inter[10:-10] = age 
				prob_big[10:-10] = prob 
			else:
				prob_big = prob
				age_inter = age
				prob_inter = np.zeros(np.size(age))

			for a in range(len(age_inter)):
				if a == 0 or a == len(age_inter)-1:
					continue
				else:
					prob_inter[a] = (prob_big[a-1]+prob_big[a]+prob_big[a+1]) / 3.0

			popt,pcov,loopnum_out = loop_curve_fit(func,age_inter,prob_inter,guess,loopnum+1)
			return popt,pcov,loopnum_out

	popt,pcov,numiterations = loop_curve_fit(func,age,prob,guess,1)
	fit = func(age,*popt)

	plt.plot(age,prob)
	plt.plot(age, fit , 'r-',linewidth=3.0)

	all_poss_age 		= np.arange(6,11,0.1)
	tot_mass 			= np.sum(func(all_poss_age,*popt))
	young_age 			= np.arange(6,7,0.1)
	frac_young_mass  	= np.sum(func(young_age,*popt)) / tot_mass
	plt.show()

	cov = np.sqrt(np.diag(np.absolute(pcov)))

	def reject_outliers(data, m=2):
		return data[abs(data - np.mean(data)) < m * np.std(data)]

	bs_mass = []
	for i in range(1000):
		popt_bs = popt + cov * np.random.standard_normal(len(popt))

		popt_bs[2],popt_bs[5],popt_bs[8],popt_bs[11] = \
		(popt_bs[2],popt_bs[5],popt_bs[8],popt_bs[11]) / \
		np.sum([popt_bs[2],popt_bs[5],popt_bs[8],popt_bs[11]])

		all_poss_age 		= np.arange(6,11,0.1)
		tot_mass 			= np.sum(func(all_poss_age,*popt_bs))
		young_age 			= np.arange(6,7,0.1)
		tot_func =  np.sum(func(young_age,*popt_bs))
		if tot_func == 0:
			continue 
		bs_mass.append(tot_func / tot_mass)

	if np.size(bs_mass)==0:
		spread_frac = frac_young_mass
	else:
		#spread_frac = (np.percentile(bs_mass,75)-np.percentile(bs_mass,25)) * np.sqrt(numiterations)
		bs_mass = reject_outliers(np.asarray(bs_mass),m=2)
		spread_frac = np.std(bs_mass)#/np.sqrt(1000)
		#print frac_young_mass,spread_frac
		Tracer()()

	return np.absolute(frac_young_mass),spread_frac

