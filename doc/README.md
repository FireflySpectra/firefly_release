# Running Firefly 

Here we provide some examples on how to run Firefly and how to read the fits to one SDSS output. The full documentation of the Python code generated with Sphynx can be found [here](http://www.mpe.mpg.de/~comparat/firefly_doc/). 

## Content 

**firefly_output.ipynb** This notebook shows what is typically stored in a Firefly output file, including the best fit spectra compared to the original.

**one_spectra.py** Stand-alone python code to run Firefly over one of the provided spectrum in 'example_data'. This program is explained step by step in the 'one_spectra_firefly.ipynb' notebook.

**one_spectra_firefly.ipynb** This notebook shows how to run Firefly over one example spectra. This is complementary to the example script: ../bin/SDSS/one_spectra.py.

**SDSSinHPC** This directory contains a script to write and submit jobs to an HPC facility such as [Sciama](http://www.sciama.icg.port.ac.uk/), using qsub commands, that call the 'run_stellarpop.py' code.

**example_data** This folder contains one Firefly output example and 3 SDSS spectra to run Firefly over.


