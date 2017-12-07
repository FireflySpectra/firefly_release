# Running Firefly 

Here we provide a guide on how to run Firefly using examples and how to read the fits file of one SDSS output. Examples have been written in Python2. 

The full documentation of the Python code generated with Sphynx can be found [here](http://www.mpe.mpg.de/~comparat/firefly_doc/). 

## Content 

**firefly_output.ipynb** This notebook shows what is typically stored in a Firefly output file, including the best fit spectra compared to the original.

**one_spectra.py** Stand-alone python code to run Firefly over one of the provided spectrum is in 'example_data'. This program is explained step by step in the 'one_spectra_firefly.ipynb' notebook.

**one_spectra_firefly.ipynb** This notebook shows how to run Firefly over one example spectra. This is complementary to the example script: ../bin/SDSS/one_spectra.py.

**GC_spectrum.ipynb** This is a stand-alone python routine to run a globular cluster spectrum (exmple_data/spectra/NGC7099_2016-10-01.fits) with Firefly  

**SDSSinHPC** This directory contains a script to write and submit jobs to an HPC facility such as [Sciama](http://www.sciama.icg.port.ac.uk/), using qsub commands, that call the 'run_stellarpop.py' code.

**example_data** This folder contains one Firefly output example and 3 SDSS spectra to run Firefly over.
