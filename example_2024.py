#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 16:07:51 2024

@author: Sebastijan Mrak
"""

import numpy as np
from datetime import datetime
import SECSSD as SD
import matplotlib.pyplot as plt
import os

folder = os.path.join(os.getcwd(), "northern_2015_march_1"+os.sep)

t0 = datetime(2015,3,1,21,50)
t1 = datetime(2015,3,1,22,10)
D = SD.read_superDARN(folder, start_time=t0, datatype="v3_grid", end_time=t1)

T = datetime(2015,3,1,22,0)

# D1
D_time = D['times_start'].values.astype('datetime64[s]').astype(datetime)
idt = np.isin(D_time, T)
D_los_east = D['velocity_east'][idt].values
D_los_north = D['velocity_north'][idt].values
D_los = np.vstack((D_los_east, D_los_north)).T
D_los_latlon = np.vstack((D['lat'][idt].values, D['lon'][idt].values)).T
D_radar_latlon = np.vstack((D["radar_lat"][idt].values, D["radar_lon"][idt].values)).T
# Convert the array to a list of tuples
pairs_list = [tuple(row) for row in D_radar_latlon]
# Get unique pairs
radar_latlon = np.array(list(set(pairs_list))).squeeze()

xg, yg, rad, prediction_grid = SD.discretize([40,90], [-180,180], 2, 5, velocity_latlon = D_los_latlon, debugging=1, density_function='gauss', density_min=1, density_max=4)

epsilon=0.05
N = 7
secs_velocity = np.nan * np.ones((N, prediction_grid.shape[0], 3))
for i in range(N):
    poles = SD.discretize([40,90], [-180, 180], 2, 5, velocity_latlon = D_los_latlon, density_function='gauss')
    secs_velocity[i] = SD.predict_with_SECS(D_los, D_los_latlon, prediction_grid, poles, epsilon=epsilon)
    print(np.shape(poles))
prediction_velocity = np.nanmedian(secs_velocity, axis=0)
prediction_std = np.nanstd(secs_velocity, axis=0)
prediction_std_magnitude = np.sqrt(prediction_std[:,0]**2, prediction_std[:,1]**2)

vel_close = SD.velocity_isclose(prediction_grid, D_los_latlon, tolerance=300, units="km")

### Plotting

fig = plt.figure(figsize=[8,5])
ax = fig.add_subplot(111)
ax.set_title(f"SECS output at {T}")
Q = ax.quiver(D_los_latlon[:,1], D_los_latlon[:,0], 
              D_los[:,0], D_los[:,1],
              color = 'r', scale = 10000, width=0.0015)
qk = plt.quiverkey(Q, 0.85, 1.025, 500,
                '500 m s$^{-1}$)',
                labelpos='E',
                color='r')
ax.grid()
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")


####

fig = plt.figure(figsize=[8,5])
ax = fig.add_subplot(111)
ax.set_title(f"SECS output at {T}")
Q = ax.pcolormesh(xg, yg, rad, cmap="jet")
ax.scatter(poles[:,1], poles[:,0], s=18, c='m', marker='+', label="Poles")
ax.scatter(D_los_latlon[:,1], D_los_latlon[:,0], s=2, c='w', label="LOS")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.colorbar(Q, label="Density function")
plt.legend(facecolor='k', edgecolor="none", labelcolor='w')

##### SECS Solution

fig = plt.figure(figsize=[8,5])
ax = fig.add_subplot(111)
ax.set_title(f"SECS output at {T}")
Q = ax.quiver(prediction_grid[vel_close,1], prediction_grid[vel_close,0], 
                    prediction_velocity[vel_close,0], prediction_velocity[vel_close,1],
                    color = 'b', scale = 10000, width=0.0015)
ax.quiver(prediction_grid[~vel_close,1], prediction_grid[~vel_close,0], 
          prediction_velocity[~vel_close,0], prediction_velocity[~vel_close,1],
          color = 'k', scale = 10000, width=0.0015)

ax.scatter(radar_latlon[:,1], radar_latlon[:,0], s=5, marker='x', c='r')
qk = plt.quiverkey(Q, 0.85, 1.025, 500,
                '500 m s$^{-1}$)',
                labelpos='E',
                color='b')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid()

##### SECS STD

fig = plt.figure(figsize=[8,5])
ax = fig.add_subplot(111)
ax.set_title(f"SECS Standard Deviation with {N} iterations")
Q = ax.scatter(prediction_grid[:,1], prediction_grid[:,0], 
               s=prediction_std_magnitude/10, edgecolor='k', marker='o', facecolor='none')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.scatter(100, 35, s=50, color='k', marker='o', facecolors='none',)
ax.text(108, 34.3, "sigma=50 m/s")
ax.grid()
