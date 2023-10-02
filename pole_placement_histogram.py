# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 12:47:15 2023

@author: ryanj
"""

import os

from datetime import datetime
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import cartopy.crs as ccrs

# import superDARN SECS 
import SECSSD as SD



import numpy as np
from scipy.io import savemat

# input directory containing one day superDARN data (only one day currently, and as many files as there are radars)
superDARN_data_directory = "northern_2015_march_1" + os.sep

# directory to save the data
save = 0
savedir = "output_figures" + os.sep


def compute_percent_change(all):
    '''
    computes the percent difference from the mean for each of the ROWS,
    and then it combines the percent differences into one volumn vector
    '''
    
    mean_all = np.mean(all, axis=1)[:, np.newaxis]
    percent_change_all = 100 * ((all - mean_all) / mean_all)
    percent_change_return = percent_change_all.reshape(-1, 1)
    return percent_change_return

# define the starting time and ending time.
# specify a select range to reduce the amount of for loop iterations, and therefore computation time
# can, instead, only select a single date by inputting only a start time, and the superdarn file reader function will return every time in the file
#start_date = datetime(2015, 3, 1, 4, 0)
start_date = datetime(2015, 3, 1, 5, 0)

# read the data
# specify the grid as "v3_grid" because this is the option selected in the download data webpage located at https://superdarn.jhuapl.edu/download/
#(all_data, all_time) = SD.read_superDARN(superDARN_data_directory, "v3_grid", start_date, end_date)
(all_data, all_time) = SD.read_superDARN(superDARN_data_directory, "v3_grid", start_date)

# generate the list of prediction points. this is currently a grid, reshaped into a 2D list
#prediction_latlim = [46, 80]
#prediction_lonlim = [-160, -30]

prediction_latlim = [35, 80]
prediction_lonlim = [-155, -40]

prediction_lat_step = 1.5
prediction_lon_step = 1.8
# obtain prediction latitude and longitude locations
prediction_latlon = SD.place_prediction(prediction_latlim, prediction_lonlim, prediction_lat_step, prediction_lon_step)

poles_latlim = [30, 85]
poles_lonlim = [-220, -20]

# the number of times to repeat placing the poles and computing the SECS velocity fit
num_iterations = 50

# SVD truncation parameter
epsilon = 0.1

'''
uniform placement poles
'''

# select the data to input into SECS
select_velocity = all_data[0][0]
select_velocity_latlon= all_data[1][0]
select_radar_latlon = all_data[2][0]
select_radar_index = all_data[3][0]
select_time = all_time[0][0] # get out of list and into datetime format

# cut the velocity data to only the areas inside the prediction locations
#(select_velocity, select_velocity_latlon) = SD.return_interior_input_velocities(select_velocity, select_velocity_latlon, prediction_latlim, prediction_lonlim)

print("Running " + datetime.strftime(select_time, "%Y-%m-%d %H:%M") + ":")

# iterate through the same input data and compute the output
# change the poles though because they are randomly generated
# observe the variance in SECS model velocity outputs
for k in range(0, num_iterations):
    # generate the variably spaced SECS poles utilizing the input measurements
    print("\tIteration (uniform pole spacing) "+ str(k+1) + "...", end = " ")
    uni_radius = 0.92
    # poles_latlim = [44, 82]
    # poles_lonlim = [-162, -28]
    poles_lat_step = uni_radius
    poles_lon_step = uni_radius
    t1 = time.time()
    (num_poles, poles_latlon) = SD.place_poles(poles_latlim, poles_lonlim, poles_lat_step, poles_lon_step,
                                   select_velocity_latlon, density_curvature=4, max_density=0.5, close_tolerance=0) #0.5
    t2 = time.time()
    print(str(num_poles) + " poles computed in " + "{:.2f}".format(t2-t1) + " seconds...", end = " ")

    # run SECS!
    t1 = time.time()
    prediction_velocity_frame_pr = SD.predict_with_SECS(select_velocity, select_velocity_latlon, prediction_latlon, poles_latlon, epsilon=epsilon)
    t2 = time.time()
    print("SECS prediction computed in " + "{:.2f}".format(t2-t1) + " seconds")

    # compute the close and far vectors
    (prediction_velocity_close, prediction_velocity_far, prediction_latlon_close, prediction_latlon_far) = \
        SD.compute_close_and_far(prediction_velocity_frame_pr, prediction_latlon, select_velocity_latlon)
    
    # compute the magnitude of the velocities and the angles of the vectors
    # angle is defined as if a cartesian grid, so angle counterclosewise from x-axis. this definition is not critical -- we only care about relative angle from the mean
    # only do the first two dimensions because there is no z component (the z componet is numerically computed to be zero to within machine precision)
    if k == 0:
        mag_close_uni =  np.sqrt(prediction_velocity_close[:, [0]]**2 + prediction_velocity_close[:, [1]]**2)
        mag_far_uni = np.sqrt(prediction_velocity_far[:, [0]]**2 + prediction_velocity_far[:, [1]]**2)
        angle_close_uni = np.arctan2(prediction_velocity_close[:, [1]], prediction_velocity_close[:, [0]]) * 180/np.pi # degrees
        angle_far_uni = np.arctan2(prediction_velocity_far[:, [1]], prediction_velocity_far[:, [0]]) * 180/np.pi # degrees
    else:
        mag_close_uni = np.hstack((mag_close_uni, np.sqrt(prediction_velocity_close[:, [0]]**2 + prediction_velocity_close[:, [1]]**2)))
        mag_far_uni = np.hstack((mag_far_uni, np.sqrt(prediction_velocity_far[:, [0]]**2 + prediction_velocity_far[:, [1]]**2)))
        angle_close_uni = np.hstack((angle_close_uni, np.arctan2(prediction_velocity_close[:, [1]], prediction_velocity_close[:, [0]]) * 180/np.pi)) # degrees
        angle_far_uni = np.hstack((angle_far_uni, np.arctan2(prediction_velocity_far[:, [1]], prediction_velocity_far[:, [0]]) * 180/np.pi)) # degrees


'''
variable-placement poles
'''

# select the data to input into SECS
select_velocity = all_data[0][0]
select_velocity_latlon= all_data[1][0]
select_radar_latlon = all_data[2][0]
select_radar_index = all_data[3][0]
select_time = all_time[0][0] # get out of list and into datetime format

# cut the velocity data to only the areas inside the prediction locations
#(select_velocity, select_velocity_latlon) = SD.return_interior_input_velocities(select_velocity, select_velocity_latlon, prediction_latlim, prediction_lonlim)

print("Running " + datetime.strftime(select_time, "%Y-%m-%d %H:%M") + ":")

# iterate through the same input data and compute the output
# change the poles though because they are randomly generated
# observe the variance in SECS model velocity outputs
for k in range(0, num_iterations):
    # generate the variably spaced SECS poles utilizing the input measurements
    print("\tIteration (variable pole spacing) "+ str(k+1) + "...", end = " ")
    # poles_latlim = [44, 82]
    # poles_lonlim = [-162, -28]
    poles_lat_step = 4
    poles_lon_step = 5
    t1 = time.time()
    (num_poles, poles_latlon) = SD.place_poles(poles_latlim, poles_lonlim, poles_lat_step/2, poles_lon_step/2,
                                           select_velocity_latlon, density_curvature=4, max_density=0.5, close_tolerance=1.5)
    t2 = time.time()
    print(str(num_poles) + " poles computed in " + "{:.2f}".format(t2-t1) + " seconds...", end = " ")
    savemat("poles.mat", {'poles': poles_latlon, 'velocity_latlon': select_velocity_latlon})

    # run SECS!
    t1 = time.time()
    prediction_velocity_frame_pr = SD.predict_with_SECS(select_velocity, select_velocity_latlon, prediction_latlon, poles_latlon, epsilon=epsilon)
    t2 = time.time()
    print("SECS prediction computed in " + "{:.2f}".format(t2-t1) + " seconds")

    # compute the close and far vectors
    (prediction_velocity_close, prediction_velocity_far, prediction_latlon_close, prediction_latlon_far) = \
        SD.compute_close_and_far(prediction_velocity_frame_pr, prediction_latlon, select_velocity_latlon)
    
    # compute the magnitude of the velocities and the angles of the vectors
    # angle is defined as if a cartesian grid, so angle counterclosewise from x-axis. this definition is not critical -- we only care about relative angle from the mean
    # only do the first two dimensions because there is no z component (the z componet is numerically computed to be zero to within machine precision)
    if k == 0:
        mag_close_vary =  np.sqrt(prediction_velocity_close[:, [0]]**2 + prediction_velocity_close[:, [1]]**2)
        mag_far_vary = np.sqrt(prediction_velocity_far[:, [0]]**2 + prediction_velocity_far[:, [1]]**2)
        angle_close_vary = np.arctan2(prediction_velocity_close[:, [1]], prediction_velocity_close[:, [0]]) * 180/np.pi
        angle_far_vary = np.arctan2(prediction_velocity_far[:, [1]], prediction_velocity_far[:, [0]]) * 180/np.pi
    else:
        mag_close_vary = np.hstack((mag_close_vary, np.sqrt(prediction_velocity_close[:, [0]]**2 + prediction_velocity_close[:, [1]]**2)))
        mag_far_vary = np.hstack((mag_far_vary, np.sqrt(prediction_velocity_far[:, [0]]**2 + prediction_velocity_far[:, [1]]**2)))
        angle_close_vary = np.hstack((angle_close_vary, np.arctan2(prediction_velocity_close[:, [1]], prediction_velocity_close[:, [0]]) * 180/np.pi))
        angle_far_vary = np.hstack((angle_far_vary, np.arctan2(prediction_velocity_far[:, [1]], prediction_velocity_far[:, [0]]) * 180/np.pi))



'''
compute statistics
'''
# compute percent difference of all the magnitudes compared to the mean of EACH PRED_LATLON POINT
mag_percent_close_vary = compute_percent_change(mag_close_vary)
mag_percent_far_vary = compute_percent_change(mag_far_vary)
mag_percent_close_uni = compute_percent_change(mag_close_uni)
mag_percent_far_uni = compute_percent_change(mag_far_uni)

# compute the percent difference of all the angles compared to the mean of EACH PRED_LATLON POINT
ang_percent_close_vary = compute_percent_change(angle_close_vary)
ang_percent_far_vary = compute_percent_change(angle_far_vary)
ang_percent_close_uni = compute_percent_change(angle_close_uni)
ang_percent_far_uni = compute_percent_change(angle_far_uni)



'''
plotting section
'''   
f_size_title_sup = 62
f_size_title = 56
f_size_axis = 52
f_size_ticks = 44
hfont_sup = {'fontname':'Times New Roman', 'size': f_size_title_sup}
hfont = {'fontname':'Times New Roman', 'size': f_size_title}
hfont_axis = {'fontname':'Times New Roman', 'size': f_size_axis}
hfont_ticks = {'fontname':'Times New Roman', 'size': f_size_ticks}
fig_size = [30, 15]
num_bins = 250
color_str = "#00b2b5"
lb = 75 # labelpad

# plot histograms!

###########
###### NEW (better use of subplot)

fig = plt.figure(figsize=fig_size)
gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
(ax1, ax2) = gs.subplots(sharey='row')
fig.suptitle("Stability of Nearby SECS Prediction Locations, Magnitude of Velocity", fontweight = "bold", **hfont_sup)
ax1.hist(mag_percent_close_uni, num_bins, range=[-100, 100], color=color_str)
ax2.hist(mag_percent_close_vary, num_bins, range=[-100, 100], color=color_str)
ax1.set_title("Uniform SECS", **hfont)
ax2.set_title("Adaptive SECS", **hfont)

plt.setp(ax1.get_xticklabels(), **hfont_ticks)
plt.setp(ax1.get_yticklabels(), **hfont_ticks)
plt.setp(ax2.get_xticklabels(), **hfont_ticks)
plt.setp(ax2.get_yticklabels(), **hfont_ticks)

fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Percent Change", labelpad=40, **hfont_axis)
plt.ylabel("Counts", labelpad=lb, **hfont_axis)


# plot the magnitude of FAR vectors, VARIABLE poles
fig = plt.figure(figsize=fig_size)
gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
(ax1, ax2) = gs.subplots(sharey='row')
fig.suptitle("Stability of Distant SECS Prediction Locations, Magnitude of Velocity", fontweight = "bold", **hfont_sup)
ax1.hist(mag_percent_far_uni, num_bins, range=[-100, 100], color=color_str)
ax2.hist(mag_percent_far_vary, num_bins, range=[-100, 100], color=color_str)
ax1.set_title("Uniform SECS", **hfont)
ax2.set_title("Adaptive SECS", **hfont)

plt.setp(ax1.get_xticklabels(), **hfont_ticks)
plt.setp(ax1.get_yticklabels(), **hfont_ticks)
plt.setp(ax2.get_xticklabels(), **hfont_ticks)
plt.setp(ax2.get_yticklabels(), **hfont_ticks)

fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Percent Change", labelpad=40, **hfont_axis)
plt.ylabel("Counts", labelpad=lb, **hfont_axis)


# plot the magnitude of CLOSE vectors, UNIFORM poles    
fig = plt.figure(figsize=fig_size)
gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
(ax1, ax2) = gs.subplots(sharey='row')
fig.suptitle("Stability of Nearby SECS Prediction Locations, Angle of Velocity", fontweight = "bold", **hfont_sup)
ax1.hist(ang_percent_close_uni, num_bins, range=[-100, 100], color=color_str)
ax2.hist(ang_percent_close_vary, num_bins, range=[-100, 100], color=color_str)
ax1.set_title("Uniform SECS", **hfont)
ax2.set_title("Adaptive SECS", **hfont)

plt.setp(ax1.get_xticklabels(), **hfont_ticks)
plt.setp(ax1.get_yticklabels(), **hfont_ticks)
plt.setp(ax2.get_xticklabels(), **hfont_ticks)
plt.setp(ax2.get_yticklabels(), **hfont_ticks)

fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Percent Change", labelpad=40, **hfont_axis)
plt.ylabel("Counts", labelpad=lb, **hfont_axis)


# plot the magnitude of FAR vectors, UNIFORM poles
fig = plt.figure(figsize=fig_size)
gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
(ax1, ax2) = gs.subplots(sharey='row')
fig.suptitle("Stability of Distant SECS Prediction Locations, Angle of Velocity", fontweight = "bold", **hfont_sup)
ax1.hist(ang_percent_far_uni, num_bins, range=[-100, 100], color=color_str)
ax2.hist(ang_percent_far_vary, num_bins, range=[-100, 100], color=color_str)
ax1.set_title("Uniform SECS", **hfont)
ax2.set_title("Adaptive SECS", **hfont)

plt.setp(ax1.get_xticklabels(), **hfont_ticks)
plt.setp(ax1.get_yticklabels(), **hfont_ticks)
plt.setp(ax2.get_xticklabels(), **hfont_ticks)
plt.setp(ax2.get_yticklabels(), **hfont_ticks)

fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Percent Change", labelpad=40, **hfont_axis)
plt.ylabel("Counts", labelpad=lb, **hfont_axis)





#############
##### OLD (poor use of subplots)
# plot the magnitude of CLOSE vectors, VARIABLE poles
# fig = plt.figure(figsize=fig_size)
# gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
# (ax1, ax2) = gs.subplots(sharey='row')
# fig.suptitle("Stability of Nearby SECS Prediction Locations, Adaptive Poles", fontweight = "bold", **hfont_sup)
# ax1.hist(mag_percent_close_vary, num_bins, range=[-100, 100], color=color_str)
# ax2.hist(ang_percent_close_vary, num_bins, range=[-100, 100], color=color_str)
# ax1.set_title("Magnitude of Vectors", **hfont)
# ax2.set_title("Angle of Vectors", **hfont)

# plt.setp(ax1.get_xticklabels(), **hfont_ticks)
# plt.setp(ax1.get_yticklabels(), **hfont_ticks)
# plt.setp(ax2.get_xticklabels(), **hfont_ticks)
# plt.setp(ax2.get_yticklabels(), **hfont_ticks)

# fig.add_subplot(111, frameon=False)
# # hide tick and tick label of the big axis
# plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
# plt.xlabel("Percent Change", labelpad=40, **hfont_axis)
# plt.ylabel("Counts", labelpad=50, **hfont_axis)


# # plot the magnitude of FAR vectors, VARIABLE poles
# fig = plt.figure(figsize=fig_size)
# gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
# (ax1, ax2) = gs.subplots(sharey='row')
# fig.suptitle("Stability of Distant SECS Prediction Locations, Adaptive Poles", fontweight = "bold", **hfont_sup)
# ax1.hist(mag_percent_far_vary, num_bins, range=[-100, 100], color=color_str)
# ax2.hist(ang_percent_far_vary, num_bins, range=[-100, 100], color=color_str)
# ax1.set_title("Magnitude of Vectors", **hfont)
# ax2.set_title("Angle of Vectors", **hfont)

# plt.setp(ax1.get_xticklabels(), **hfont_ticks)
# plt.setp(ax1.get_yticklabels(), **hfont_ticks)
# plt.setp(ax2.get_xticklabels(), **hfont_ticks)
# plt.setp(ax2.get_yticklabels(), **hfont_ticks)

# fig.add_subplot(111, frameon=False)
# # hide tick and tick label of the big axis
# plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
# plt.xlabel("Percent Change", labelpad=40, **hfont_axis)
# plt.ylabel("Counts", labelpad=50, **hfont_axis)


# # plot the magnitude of CLOSE vectors, UNIFORM poles    
# fig = plt.figure(figsize=fig_size)
# gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
# (ax1, ax2) = gs.subplots(sharey='row')
# fig.suptitle("Stability of Nearby SECS Prediction Locations, Uniform Poles", fontweight = "bold", **hfont_sup)
# ax1.hist(mag_percent_close_uni, num_bins, range=[-100, 100], color=color_str)
# ax2.hist(ang_percent_close_uni, num_bins, range=[-100, 100], color=color_str)
# ax1.set_title("Magnitude of Vectors", **hfont)
# ax2.set_title("Angle of Vectors", **hfont)

# plt.setp(ax1.get_xticklabels(), **hfont_ticks)
# plt.setp(ax1.get_yticklabels(), **hfont_ticks)
# plt.setp(ax2.get_xticklabels(), **hfont_ticks)
# plt.setp(ax2.get_yticklabels(), **hfont_ticks)

# fig.add_subplot(111, frameon=False)
# # hide tick and tick label of the big axis
# plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
# plt.xlabel("Percent Change", labelpad=40, **hfont_axis)
# plt.ylabel("Counts", labelpad=50, **hfont_axis)


# # plot the magnitude of FAR vectors, UNIFORM poles
# fig = plt.figure(figsize=fig_size)
# gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
# (ax1, ax2) = gs.subplots(sharey='row')
# fig.suptitle("Stability of Distant SECS Prediction Locations, Uniform Poles", fontweight = "bold", **hfont_sup)
# ax1.hist(mag_percent_far_uni, num_bins, range=[-100, 100], color=color_str)
# ax2.hist(ang_percent_far_uni, num_bins, range=[-100, 100], color=color_str)
# ax1.set_title("Magnitude of Vectors", **hfont)
# ax2.set_title("Angle of Vectors", **hfont)

# plt.setp(ax1.get_xticklabels(), **hfont_ticks)
# plt.setp(ax1.get_yticklabels(), **hfont_ticks)
# plt.setp(ax2.get_xticklabels(), **hfont_ticks)
# plt.setp(ax2.get_yticklabels(), **hfont_ticks)

# fig.add_subplot(111, frameon=False)
# # hide tick and tick label of the big axis
# plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
# plt.xlabel("Percent Change", labelpad=40, **hfont_axis)
# plt.ylabel("Counts", labelpad=50, **hfont_axis)
