#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 16:07:51 2024

@author: Sebastijan Mrak
"""

import numpy as np
from datetime import datetime
import SECSSD as SD
import matplotlim.pyplot as plt

folder = '/Users/mraks1/Library/CloudStorage/GoogleDrive-smrak@bu.edu/My Drive/SuperDARN_SECS/20130117/'

t0 = datetime(2013,1,17,21,50)
t1 = datetime(2013,1,17,22,10)
D1 = SD.read_superDARN(folder+'data/v3_grid/', start_time=t0, datatype="v3_grid", end_time=t1)

T = datetime(2013,1,17,22,0)

# D1
D1_time = D1['times_start'].values.astype('datetime64[s]').astype(datetime)
idt1 = np.isin(D1_time, T)
D1_los_east = D1['velocity_east'][idt1].values
D1_los_north = D1['velocity_north'][idt1].values
D1_los = np.vstack((D1_los_east, D1_los_north, np.zeros(D1_los_north.size))).T
D1_los_xy = np.vstack((D1['lat'][idt1].values, D1['lon'][idt1].values)).T


xg1, yg1, rad1, prediction_grid = SD.discretize([40,90], [-180,180], 2, 5, velocity_latlon = D1_los_xy, debugging=1, density_function='gauss', density_min=1, density_max=4)

epsilon=0.05
N = 7
secs_velocity = np.nan * np.ones((N, prediction_grid.shape[0], 3))
for i in range(N):
    poles = SD.discretize([40,90], [-180, 180], 2, 5, velocity_latlon = D1_los_xy, density_function='gauss')
    secs_velocity[i] = SD.predict_with_SECS(D1_los, D1_los_xy, prediction_grid, poles, epsilon=epsilon)
prediction_velocity = np.nanmedian(secs_velocity, axis=0)
prediction_std = np.nanstd(secs_velocity, axis=0)

vel_close = SD.velocity_isclose(prediction_grid, D1_los_xy)

fig = plt.figure(figsize=[8,5])
ax = fig.add_subplot(111)
ax.set_title(f"SECS output at {T}")
Q = ax.quiver(prediction_grid[vel_close,1], prediction_grid[vel_close,0], 
                    prediction_velocity[vel_close,0], prediction_velocity[vel_close,1],
                    color = 'b', scale = 10000, width=0.0035)
ax.quiver(prediction_grid[~vel_close,1], prediction_grid[~vel_close,0], 
          prediction_velocity[~vel_close,0], prediction_velocity[~vel_close,1],
          color = 'k', scale = 10000, width=0.0015)
qk = plt.quiverkey(Q, 0.85, 1.025, 500,
                '500 m s$^{-1}$)',
                labelpos='E',
                color='b')