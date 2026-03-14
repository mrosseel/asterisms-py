# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 22:29:51 2017

@author: lauri.kangas
"""
import numpy as np

def rectilinear(x, inverse=False):
    if inverse:
        return np.arctan(x)
    else:
        return np.tan(x)

def stereographic(x, inverse=False):
    if inverse: # mm (image plane) to radians (celestial sphere)
        return 2*np.arctan(x/2)
    else: # radians (sphere) to mm (image plane)
        return 2*np.tan(x/2)

def unity(x, inverse=False):
    return x


