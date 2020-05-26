'''

; #############################################################################################################################################
; #############################################################################################################################################
;
; Version 1.0
; Adapted from https://stackoverflow.com/questions/36031338/interpolate-z-values-in-a-3d-surface-starting-from-an-irregular-set-of-points
; S. Meneses-Goytia, 2017
; Institute of Cosmology and Gravitation - University of Portsmouth,
;                                          Portmouth, UK
;
; ---------------------------------------------------------------------------------------------------------------------------------------------
;
; The class interpolate z values from a 3D surface, starting from an irregular set of x, y and z points.
; It estimates any position using input data using Inverse Squared Distance method (ISD) which is very stable for estimation - no divisions by 0
;
; ---------------------------------------------------------------------------------------------------------------------------------------------
;
; #############################################################################################################################################
; #############################################################################################################################################

'''

import numpy as np
import matplotlib.pyplot as plt

class estimation():
        def __init__(self,datax,datay,dataz):
            self.x = datax
            self.y = datay
            self.v = dataz

        def estimate(self,x,y,using='ISD'):
            """
            Estimate point at coordinate x,y based on the input data for this
            class.
            """
            if using == 'ISD':
                return self._isd(x,y)

        def _isd(self,x,y):
            d = np.sqrt((x-self.x)**2+(y-self.y)**2)
            if d.min() > 0:
                v = np.sum(self.v*(1/d**2)/np.sum(1/d**2))
                return v
            else:
                return self.v[d.argmin()]

