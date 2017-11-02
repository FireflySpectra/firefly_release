# Firefly
## Fitting Iteratively for Relative Likelihood Analysis (of Galaxy Spectra)

FIREFLY is a chi-squared minimisation fitting code that for a given input Spectral Energy Distribution (SED), compares combinations of single-burst stellar population models (SSP), following an iterative best-fitting process until convergence is achieved. The weight of each component can be arbitrary and no regularization or additional prior than the adopted model grid is applied. Dust attenuation is added in a novel way, using a High-Pass Filter (HPF) in order to rectify the continuum before fitting. The returned attenuation array is then matched to known analytical approximations to return an E(B-V) value. This procedure allows for removal of large scale modes of the spectrum associated with dust and/or poor flux calibration. FIREFLY provides light- and mass-weighted stellar population properties (age and metallicity), E(B-V) values and stellar mass for the most likely best fitting model. Errors on these properties are obtained by the likelihood of solutions within the statistical cut (of order 100-1000).

The code can fit a wide range of stellar population models and galaxy spectra. At present, the code contains functionality to fit the observed galaxy spectra from SDSS (for the main sample and MaNGA), DEEP2 and globular clusters.

The full documentation of the Python code generated with Sphynx can be found [here.](http://www.mpe.mpg.de/~comparat/firefly_doc/)


## Installation

Requirements: python 2.7.8 and its main packages all installable through pip: numpy, scipy, matplotlib, math ...
astro dependencies: astropy, installable with pip

```
git clone https://[username]@github.com/[username]/firefly_release
```

The Stellar Population Models templates and mass loss files can be downloaded from:
* Use the following line to download all the stellar population model templates and the mass loss files from the [SDSS Data Base](https://svn.sdss.org/data/sdss/stellarpopmodels/tags/v1_0_2/) (this is password protected at the moment, you can use your SDSS account credentials if you have one):
```
svn checkout https://svn.sdss.org/data/sdss/stellarpopmodels/tags/v1_0_2/ stellar_population_models
```

* Use the following lines to download all the stellar population model templates and the mass loss files from [Claudia Maraston's repository](http://icg.port.ac.uk/~manga-firefly/stellar_population_models/data/):
```
mkdir stellar_population_models/
cd stellar_population_models/
wget -r -nH --cut-dirs=2 --no-parent --reject="index.html*" http://icg.port.ac.uk/~manga-firefly/stellar_population_models/data/
```

## Acknowledgment

If you use this code as a resource to produce a scientific result, cite the papers:

* [Wilkinson et al. 2016](https://arxiv.org/abs/1503.01124)
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
