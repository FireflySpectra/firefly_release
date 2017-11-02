# Firefly
## Fitting Iteratively for Relative Likelihood Analysis (of Stellar Population Spectra, e.g. galaxies, star clusters, model spectra)

FIREFLY is a chi-squared minimisation fitting code for deriving the stellar population properties of stellar systems, be these observed galaxy or star cluster spectra, or model spectra from simulations. FIREFLY fits combinations of single-burst stellar population models to spectroscopic data, following an iterative best-fitting process controlled by the Bayesian Information Criterion. No priors are applied, rather all solutions within a statistical cut are retained with their weight, which is arbitrary. Moreover, no additive or multiplicative polynomia are employed to adjust the spectral shape and no regularisation is imposed. This fitting freedom allows one to map out the effect of intrinsic spectral energy distribution (SED) degeneracies, such as age, metallicity, dust reddening on stellar population properties, and to quantify the effect of varying input model components on such properties. Dust attenuation is included using a new procedure, which employs a High-Pass Filter (HPF) in order to rectify the continuum before fitting. The returned attenuation array is then matched to known analytical approximations to return an E(B-V) value. This procedure allows for removal of large scale modes of the spectrum associated with dust and/or poor flux calibration. The fitting method has been extensively tested with a comprehensive suite of mock galaxies, real galaxies from the Sloan Digital Sky Survey and Milky Way globular clusters. The robustness of the derived properties was assessed as a function of signal-to-noise ratio and adopted wavelength range. FIREFLY is able to recover age, metallicity, stellar mass and even the star formation history remarkably well down to a S/N~5, for moderately dusty systems. 

FIREFLY provides light- and mass-weighted stellar population properties - age, metallicity, E(B-V), stellar mass and its partition in remnants (white dwarfs, neutron stars, black-holes) - for the best fitting model and its components. The star formation rates for the individual components are given, the total past average can be easily obtained from the provided quantities. The star formation history can be easily derived by plotting the SSP contributions with their weights. Errors on these properties are obtained by the likelihood of solutions within the statistical cut. 

The code can in principle fit any model to any spectrum at any spectral resolution and over any wavelength range. At present, the code has been applied to spectra from SDSS, integrated and from Integral field Unit (MANGA), the DEEP2 survey and globular clusters from various sources. We make available customised models that are already matched to the spectral resolution of these datasets. The suite of possibilities will increase through the use of the code with different empirical or simulated datasets. 

The full documentation of the Python code generated with Sphynx can be found [here.](http://www.mpe.mpg.de/~comparat/firefly_doc/)

## Output of Firefly



## Installation

Requirements: python 2.7.8 and its main packages all installable through pip: numpy, scipy, matplotlib, math ...
astro dependencies: astropy, installable with pip

```
git clone https://[username]@github.com/[username]/firefly_release
```

* Use the following lines to download all the stellar population model templates and the files for taking into account stellar mass loss from [Claudia Maraston's repository](http://icg.port.ac.uk/~manga-firefly/stellar_population_models/data/):
```
mkdir stellar_population_models/
cd stellar_population_models/
wget -r -nH --cut-dirs=2 --no-parent --reject="index.html*" http://icg.port.ac.uk/~manga-firefly/stellar_population_models/data/
```

•	Or from the SDSS Data Base (this is password protected at the moment, you can use your SDSS account credentials if you have one):
```
svn checkout https://svn.sdss.org/data/sdss/stellarpopmodels/tags/v1_0_2/ stellar_population_models
```

## Acknowledgment

We are delighted you use our code! If you use our code, please cite the following papers:

* [Wilkinson et al. 2017](https://arxiv.org/abs/)
* [Comparat et al. 2018, in prep.]
* [Goddard et al. 2017](https://arxiv.org/abs/1612.01546).

## Environmental variables
Once you have downloaded and unpacked the code you need to set some environmental variables adequately.

Example for a .bash_profile file (MAC users):
```
export FF_DIR='[your path to Firefly]/firefly_release-master'
export PYTHONPATH='${FF_DIR}/python:$PYTHONPATH'
export STELLARPOPMODELS_DIR=‘[your path to Firefly]/stellar_population_models'
```

Example for a .bashrc file:
```
$FF_DIR='[your path to Firefly]/firefly_release'
$PYTHONPATH=‘${FF_DIR}/python:$PYTHONPATH’
$STELLARPOPMODELS_DIR=‘[your path to Firefly]/stellar_population_models'
```

Example for a .cshrc file:
```
setenv FF_DIR '[your path to Firefly]/firefly_release'
setenv PYTHONPATH ${PYTHONPATH}:${FF_DIR}/python
setenv STELLARPOPMODELS_DIR '[your path to Firefly]/stellar_population_models'
```

## Content 

**bin** Example scripts to run Firefly using certain sets of data and using high performance computing facilities.

**doc** Documentation and detailed examples to help understanding how to run Firefly and read it's output.

**python** The Firefly code.
