# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""

Defines a catch-all class for useful constants.  These are meant to be
values that are **not** available elsewhere, such as
`astropy.constants`_.

*License*:
    Copyright (c) 2015, SDSS-IV/MaNGA Pipeline Group
        Licensed under BSD 3-clause license - see LICENSE.rst

*Source location*:
    $MANGADAP_DIR/python/mangadap/util/constants.py

*Imports and python version compliance*:
    ::

        from __future__ import division
        from __future__ import  print_function
        from __future__ import absolute_import
        from __future__ import unicode_literals

        import sys
        if sys.version > '3':
            long = int

        import numpy

*Revision history*:
    | **28 May 2015**: Original implementation by K. Westfall (KBW)

.. _astropy.constants: http://docs.astropy.org/en/stable/constants/index.html

"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import sys
if sys.version > '3':
    long = int

import numpy

__author__ = 'Kyle Westfall'

class constants:
    r"""
    Defines the following set of constants:

    +-----------------------------+-------------------------------+
    | Attribute                   | Value                         |
    +=============================+===============================+
    | :attr:`sig2fwhm`            | :math:`2\sqrt{2\ln(2)}`       |
    +-----------------------------+-------------------------------+
    | :attr:`rad2arcs`            | :math:`3600\frac{180}{\pi}`   |
    +-----------------------------+-------------------------------+
    | :attr:`sidereal_year`       | :math:`31558175.779`          |
    +-----------------------------+-------------------------------+

    Attributes:
        sig2fwhm (float): Conversion factor from the standard deviation,
            :math:`\sigma`, of a Gaussian to its full-width at half
            maximum (FWHM).
        rad2arcs (float): Conversion factor from radians to to
            arcseconds 
        sidereal_year (float): Length of a sidereal year (1.0000385
            Gregorian years) in seconds.

    """
    def __init__(self):

        # Convert from sigma to FWHM: FWHM = sig2fwhm * sig
        self.sig2fwhm = 2.0 * numpy.sqrt(2.0 * numpy.log(2.0))

        # Convert from radians to arcseconds: arcsec = rad2arcs * radians
        self.rad2arcs = 60*60*180/numpy.pi

        # Length of one sidereal year in seconds ()
        self.sidereal_year = 31558175.779


