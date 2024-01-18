#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 17:31:00 2024

@author: ryan
"""

'''
this code uses a divergenge-free vectorfield as input to the SECS method, and the SECS method
must reproduce the vectorfield as well as possible.

various aspects such as standard deviation, etc, are measured and displayed for publication
'''


import numpy as np
from datetime import datetime
import SECSSD as SD
import matplotlib.pyplot as plt
import os


def compute_vector_components(x, y):
    '''
    function defining the divergence-free vectorfield
    '''
    zoom = 1.5
    x = zoom * x #* np.pi/180 # convert to radians
    y = zoom * y #* np.pi/180 # convert to radians
    u = x * np.cos(x + y)
    v = -x * np.cos(x + y) - np.sin(x + y)

    u = 150 * u
    v = 150 * v
    return (u, v)


def return_velocity_mask(velocity_xy, angular_tolerance=0.2, mask_option='random'):
    '''
    this function masks portions of the velocity field to limit what the SECS model can "see"
    '''
    if mask_option == 'random':
        # ...the mask is made from a variety of random points, with the angular tolerance
        # around the random points determining the non-masked area
        
        np.random.seed(17) # choose seed 17 for good distribution
        num_mask_points = 8
        
        #initialize
        mask_points = np.zeros((num_mask_points, 2))
        
        #generate points that the velocity_latlon will be grouped around.
        for i in range(num_mask_points):
            mask_points[i, 0] = np.random.random() * (max(xlim) - min(xlim)) + min(xlim) # x-coordinate
            mask_points[i, 1] = np.random.random() * (max(ylim) - min(ylim)) + min(ylim) # y-coordinate
        np.random.seed() # make random seed again
            
    elif mask_option == 'edge':
        # ...the non-masked areas are around the edge
        # assumes [-1, 1] domain for both x and y
        a = np.linspace(-1, 1, num=50, endpoint=True)[:, np.newaxis] # goes from -1 --> 1
        one = np.ones(np.shape(a))
        
        # make mask points for each edge (separately)
        mask_points1 = np.hstack((a, -one))
        mask_points2 = np.hstack((one, a))
        mask_points3 = np.hstack((a, one))
        mask_points4 = np.hstack((-one, a))
        
        # combine all 4 edges together
        mask_points = np.vstack((mask_points1, mask_points2, mask_points3, mask_points4))
    
    elif mask_option == 'disable':
        # ...then no masking is done
        vel_xy_bool = np.full(np.shape(velocity_xy), True)
        return (vel_xy_bool, None)
    
    ##### END IF
    
    # determine which velocity locations are near the mask_points (which we want to keep)
    vel_xy_bool = SD.compute_num_closeto(velocity_xy, mask_points, angular_tolerance=mask_tolerance) > 0
    
    return (vel_xy_bool, mask_points)
    
############################################
# Begin Code

# generate prediction locations
xlim = (-1, 1)
ylim = (-1 ,1)
step_prediction = 0.1

prediction_xy_poisson = SD.discretize(xlim, ylim, step_prediction, step_prediction, None, 'blue_noise')
prediction_xy_gridded = SD.discretize(xlim, ylim, step_prediction, step_prediction, None, None);


#####################
# generate truth vectorfield


#####################
# mask parts of the vectorfield

# generate velocity equally-spaced everywhere (soon-to-be masked)
step_velocity_xy = 0.1
velocity_xy = SD.discretize(xlim, ylim, step_velocity_xy, step_velocity_xy, None, 'blue_noise') # place equally-randomly
(velocityx, velocityy) = compute_vector_components(velocity_xy[:, [0]], velocity_xy[:, [1]]) # compute the vectorfield components
velocity = np.hstack((velocityx, velocityy))

mask_tolerance = 0.3 # how large is the area that is let through the mask, degrees

# compute the mask!
(velocity_bool, mask_points) = return_velocity_mask(velocity_xy, angular_tolerance=mask_tolerance, mask_option='edge') # try 'random' and 'edge'

# perform the logical indexing
velocity_xy_mask = velocity_xy[velocity_bool, :]
velocity_mask = velocity[velocity_bool, :]


#####################
# generate adaptive poles

step_adaptive_poles = step_velocity_xy*0.3

poles_adaptive = SD.discretize(xlim, ylim, step_adaptive_poles, step_adaptive_poles, velocity_xy_mask, 'num_near_logistic', density_min=step_adaptive_poles, density_max=0.3)


#####################
# generate gridded poles

# to ensure there is the same number of poles for both adaptive and gridded (for a fair comparison),
# the step size of the gridded poles will be determined by the number of adaptive poles to ensure roughly equal numbers
num_adaptive_poles = np.size(poles_adaptive, 0)
step_gridded_poles = 2 / np.sqrt(num_adaptive_poles)

poles_gridded = SD.discretize(xlim, ylim, step_gridded_poles, step_gridded_poles, None, None)


#####################
# run SECS! (gridded poles and adaptive poles)
epsilon = 0.05

# gridded
secs_gridded = SD.predict_with_SECS(velocity_mask, velocity_xy_mask, prediction_xy_gridded, poles_gridded, epsilon=epsilon)

# adaptive
secs_adaptive = SD.predict_with_SECS(velocity_mask, velocity_xy_mask, prediction_xy_gridded, poles_adaptive, epsilon=epsilon)

#####################
# perform post-processing computations



'''
plotting
'''
figsize = [12, 12]
plt.rcParams['figure.figsize'] = figsize
plt.rcParams.update({'font.size': 22})
qscale = 3000
arrow_width=0.005

#####################
# plot the truth vectorfield

plot_step = 0.1
plot_xy = SD.discretize(xlim, ylim, plot_step, plot_step, None, None)
(u, v) = compute_vector_components(plot_xy[:, 0], plot_xy[:, 1])
plt.figure(1)
plt.quiver(plot_xy[:, 0], plot_xy[:, 1], u, v, scale=qscale, width=arrow_width, color='#4d4d4d')
#plt.quiver(velocity_xy_mask[:, 0], velocity_xy_mask[:, 1], velocity_mask[:, 0], velocity_mask[:, 1], scale=qscale, width=arrow_width) # plotting so you can see the arrows are exactly reproducing the truth
plt.title("Divergence-Free Vectorfield: Truth")
plt.xlim(xlim)
plt.ylim(ylim)
#plt.plot(prediction_xy_poisson[:, 0], prediction_xy_poisson[:, 1], 'ro')


#####################
# plot the masked (input) vectorfield
a = np.logical_not(velocity_bool)
fig, ax = plt.subplots(figsize=figsize)
# plot circles
for i in range(np.size(mask_points, 0)):
    c = plt.Circle((mask_points[i, :]), mask_tolerance, color='#cfcfcf')
    ax.add_patch(c)
ax.plot(velocity_xy_mask[:, 0], velocity_xy_mask[:, 1], 'o', color='#2ea300')
ax.plot(velocity_xy[a, 0], velocity_xy[a, 1], 'o', color='#9e9e9e')
ax.quiver(velocity_xy_mask[:, 0], velocity_xy_mask[:, 1], velocity_mask[:, 0], velocity_mask[:, 1], color='#2ea300', scale=qscale, width=arrow_width, label='Input') # plotting so you can see the arrows are exactly reproducing the truth
ax.quiver(velocity_xy[a, 0], velocity_xy[a, 1], velocity[a, 0], velocity[a, 1], color='#9e9e9e', scale=qscale, width=arrow_width, label='Hidden') # plotting so you can see the arrows are exactly reproducing the truth
plt.title(label="Masked Velocity for SECS input")
#plt.gca().legend(['Input', 'Hidden'])
ax.legend()
plt.xlim(xlim)
plt.ylim(ylim)


#####################
# plot the SECS poles (gridded and adaptive)
plt.figure(3)
plt.plot(poles_gridded[:, 0], poles_gridded[:, 1], 'bo')
plt.plot(poles_adaptive[:, 0], poles_adaptive[:, 1], 'ro')
plt.title("Gridded (blue) and Adaptive (red) Poles")
plt.gca().legend(['Gridded', 'Adaptive'])

plt.xlim(xlim)
plt.ylim(ylim)


#####################
# plot the SECS reconstruction (gridded)

secs_prediction_xy = prediction_xy_gridded

plt.figure(4)
plt.quiver(secs_prediction_xy[:, 0], secs_prediction_xy[:, 1], secs_gridded[:, 0], secs_gridded[:, 1], scale=qscale, width=arrow_width, color='#4d4d4d')
plt.title("SECS Reconstruction with Gridded Poles")
plt.xlim(xlim)
plt.ylim(ylim)


#####################
# plot the SECS reconstruction (adaptive)

plt.figure(5)
plt.quiver(secs_prediction_xy[:, 0], secs_prediction_xy[:, 1], secs_adaptive[:, 0], secs_adaptive[:, 1], scale=qscale, width=arrow_width, color='#4d4d4d')
plt.title("SECS Reconstruction with Adaptive Poles")
plt.xlim(xlim)
plt.ylim(ylim)

#####################
# plot the standard deviation (N = something)


#####################
# plot the error


#z_min, z_max = -np.abs(rad).max(), np.abs(rad).max()
#fig, ax = plt.subplots()

#c = ax.pcolormesh(xgrid, ygrid, rad, cmap='RdBu', vmin=z_min, vmax=z_max)
#ax.set_title('pcolormesh')
# set the limits of the plot to the limits of the data
#ax.axis([xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()])
#fig.colorbar(c, ax=ax)