# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 22:35:50 2017

@author: lauri.kangas
"""

import numpy as np
from numpy import sin,cos,arccos,arctan2,mod,pi
import projections

def rotate_RADEC(RAs, DECs, center_RA, center_DEC, output='xyz'):
    # rotate RA,DEC coordinates to turn center_RA,center_DEC to origin

    # RA can be rotated first
    RArotated_RAs = mod(RAs - center_RA, 2*pi)

    # convert to rectangular coordinates
    RArotated_x, \
    RArotated_y, \
    RArotated_z = RADEC_to_xyz(RArotated_RAs, DECs)

    # now we can rotate by center_DEC.
    RADECrotated_x, \
    RADECrotated_y, \
    RADECrotated_z = tilt_xyz_y(RArotated_x, \
                                RArotated_y, \
                                RArotated_z, center_DEC)

    if output.lower() == 'xyz':
        return RADECrotated_x, RADECrotated_y, RADECrotated_z
    elif output.lower() == 'radec':
        # calculate RA/DEC again
        return None

def RADEC_to_xyz(RA, DEC):
    x = cos(RA)*cos(DEC)
    y = sin(RA)*cos(DEC)
    z = sin(DEC)

    return x,y,z


def tilt_xyz_y(x, y, z, angle, x_only=False):
    # tilt xyz coordinates along y_axis by amount angle
    # x_only: if only radius matters, (for gsc region selection),
    #         don't calculate y and z

    xx = x*cos(angle)+z*sin(angle)
    if x_only:
        return xx

    yy = y
    zz = -x*sin(angle)+z*cos(angle)

    return xx,yy,zz

def xyz_radius_from_origin(x, *args):
    return arccos(x)

def fov_radius(fov, projection=projections.stereographic):
    # return half-diagonal radius of rectangular fov of given width/height
    # with given projection

    fov = np.radians(np.array(fov)) # if fov wasn't already array
    half_fov_angle = fov/2
    half_fov_imageplane = projection(half_fov_angle)
    half_diagonal_imageplane = np.hypot(*half_fov_imageplane)
    half_diagonal_radians = projection(half_diagonal_imageplane, inverse=True)

    return np.degrees(half_diagonal_radians)

def radius2fov(radius, aspect_ratio, projection=projections.stereographic):
    # aspect_ratio = height/width
    half_diagonal_radians = np.radians(radius)
    half_diagonal_imageplane = projection(half_diagonal_radians)
    diagonal_imageplane = 2 * half_diagonal_imageplane

    width_imageplane = diagonal_imageplane**2 / (1 + aspect_ratio**2)
    height_imageplane = aspect_ratio*width_imageplane

    fov_imageplane = np.array([width_imageplane, height_imageplane])
    half_fov_imageplane = fov_imageplane/2
    half_fov_radians = projection(half_fov_imageplane, inverse=True)
    fov_radians = half_fov_radians*2

    return np.degrees(fov_radians), np.array([width_imageplane, height_imageplane])



def xyz_to_imagexy(x, y, z, \
                   rotation=0, projection=projections.stereographic, include_R=False):
    # project xyz coordinates on a sphere to image plane
    # R can be returned for filtering GSR regions

    # calculate angular distance from image center along sphere
    R = xyz_radius_from_origin(x)

    r = projection(R)

    # polar angle of region coordinates in image plane
    T = arctan2(z, y)

    T += rotation

    image_x = -r * cos(T)
    image_y = r * sin(T)

    if include_R:
        return image_x, image_y, R

    return image_x, image_y

# transform X/Y star locations from image plane coordinates to pixel coordinates (non-integer)
# in: X/Y stars, sensor dimensions, pixel counts

def imagexy_to_pixelXY(xy, resolution, sensor_size=None, pixel_scale=None, axis='ij'):
    # x,y star locations on image plane to X,Y pixel coordinates (non-integer)

    x, y = xy

    if axis == 'ij':
        y *= -1
    else: # 'xy'
        pass

    if pixel_scale:
        pixel_imageplane = np.radians(pixel_scale/3600)
        X = x/pixel_imageplane + resolution[0]/2
        Y = y/pixel_imageplane + resolution[1]/2

    if sensor_size:
        sensor_width, sensor_height = sensor_size
        pixels_x, pixels_y = resolution

        X = (x+sensor_width)/sensor_width*pixels_x/2
        Y = (y+sensor_height)/sensor_height*pixels_y/2

    return X, Y

