# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:36:37 2023

@author: ryanj
"""

import os

from datetime import datetime
import time
import matplotlib.pyplot as plt

# import superDARN SECS 
import SECSSD as SD


# input directory containing one day superDARN data (only one day currently, and as many files as there are radars)
superDARN_data_directory = "superDARN_data_input_directory" + os.sep

# directory to save the data
save = 0
savedir = "output_figures" + os.sep


# define the starting time and ending time.
# specify a select range to reduce the amount of for loop iterations, and therefore computation time
# can, instead, only select a single date by inputting only a start time, and the superdarn file reader function will return every time in the file
start_date = datetime(2015, 3, 1, 4, 0)
end_date = datetime(2015, 3, 1, 4, 10)

# read the data
# specify the grid as "v3_grid" because this is the option selected in the download data webpage located at https://superdarn.jhuapl.edu/download/
(all_data, all_time) = SD.read_superDARN(superDARN_data_directory, "v3_grid", start_date, end_date)


# generate the list of prediction points. this is currently a grid, reshaped into a 2D list
prediction_latlim = [45, 75]
prediction_lonlim = [-160, -30]
prediction_lat_step = 2
prediction_lon_step = 3
# obtain prediction latitude and longitude locations
prediction_latlon = SD.place_prediction(prediction_latlim, prediction_lonlim, prediction_lat_step, prediction_lon_step)


# generate the list of poles. this is currently a grid, reshaped in to a 2D list
poles_latlim = [44, 77]
poles_lonlim = [-162, -28]
poles_lat_step = 2
poles_lon_step = 3

# eventually, use a for loop to iterate thorugh each set of input velocity measurements
# SEE --~-- MOVE TO FOR LOOP --~--
poles_latlon = SD.place_poles(poles_latlim, poles_lonlim, poles_lat_step, poles_lon_step)

# run a for loop that goes through each time and computes the SECS model for each time
# inside the for loop, it also plots and saves the figure to an output save directory
for i, select_time in enumerate(all_time):
    # select the data to input into SECS
    select_velocity = all_data[0][i]
    select_velocity_latlon= all_data[1][i]
    select_radar_latlon = all_data[2][i]
    select_radar_index = all_data[3][i]
    select_time = select_time[0] # get out of list and into datetime format
    
    # run SECS!
    t1 = time.time()
    prediction_velocity_frame_pr = SD.predict_with_SECS(select_velocity, select_velocity_latlon, select_radar_latlon, select_radar_index, prediction_latlon, poles_latlon)
    t2 = time.time()
    print("Ran SECS Model for " + datetime.strftime(select_time, "%Y-%m-%d %H:%M") + " in " + "{:.2f}".format(t2-t1) + " seconds")
    
    # compute the close and far vectors
    (prediction_velocity_close, prediction_velocity_far, prediction_latlon_close, prediction_latlon_far) = \
    SD.compute_close_and_far(prediction_velocity_frame_pr, prediction_latlon, select_velocity_latlon)

    '''
    plotting section
    '''    
    skip = 1
    fig_size = 16
    fig_latlim = prediction_latlim
    fig_lonlim = prediction_lonlim
    scale = 30000
    
    fig = plt.figure(figsize=[8,8])
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    # plot the input velocity measurements from SuperDARN
#    fig, ax1 = gm.plotCartoMap(figsize=[fig_size, fig_size], projection='lambert',
#                              parallels=np.arange(0,90.1, 10), meridians=np.arange(-130,-30,20),
#                              latlim=fig_latlim, lonlim=fig_lonlim,
#                              resolution='50m',
#                              states=False,
#                              title=select_time,
#                              background_color='w',
#                              border_color='k'
#                              )
    # plot input measurements
#    ax1.quiver(select_velocity_latlon[::skip, 1], select_velocity_latlon[::skip, 0], select_velocity[::skip, 1], select_velocity[::skip, 0],
#              transform=ccrs.PlateCarree(), color="b", 
#              width=0.001, scale=scale)
    ax1.quiver(select_velocity_latlon[::skip, 1], select_velocity_latlon[::skip, 0], select_velocity[::skip, 1], select_velocity[::skip, 0],
              color="b", width=0.001, scale=scale)
    ax1.set_xlim(prediction_lonlim)
    ax1.set_ylim(prediction_latlim)
    ax1.grid(axis='both')
    # save the figure
#    if not os.path.exists(savedir):
#        import subprocess
#        subprocess.call('mkdir "{}"'.format(savedir), shell=True)
#    fig.savefig(savedir + "_input" + "{}.png".format(select_time.strftime("%Y%m%d_%H%M")), dpi=200)
#    plt.close(fig)
    
    # plot the predicted plasma convection from SECS model
#    fig, ax2 = gm.plotCartoMap(figsize=[fig_size, fig_size], projection='lambert',
#                              parallels=np.arange(0,90.1, 10), meridians=np.arange(-130,-30,20),
#                              latlim=fig_latlim, lonlim=fig_lonlim,
#                              resolution='50m',
#                              states=False,
#                              title=select_time,
#                              background_color='w',
#                              border_color='k'
#                              )
    
    # plot close vectors
#    ax2.quiver(prediction_latlon_close[::skip, 1], prediction_latlon_close[::skip, 0], prediction_velocity_close[::skip, 1], prediction_velocity_close[::skip, 0], 
#              transform=ccrs.PlateCarree(), color="b", 
#              width=0.002, scale=scale)
    # plot far vectors
#    ax2.quiver(prediction_latlon_far[::skip, 1], prediction_latlon_far[::skip, 0], prediction_velocity_far[::skip, 1], prediction_velocity_far[::skip, 0], 
#              transform=ccrs.PlateCarree(), color="k", 
#              width=0.001, scale=scale)
    
    ax2.quiver(prediction_latlon_close[::skip, 1], prediction_latlon_close[::skip, 0], prediction_velocity_close[::skip, 1], prediction_velocity_close[::skip, 0], 
              color="b", width=0.002, scale=scale)
    ax2.quiver(prediction_latlon_far[::skip, 1], prediction_latlon_far[::skip, 0], prediction_velocity_far[::skip, 1], prediction_velocity_far[::skip, 0], 
              color="k", width=0.001, scale=scale)
    ax2.set_xlim(prediction_lonlim)
    ax2.set_ylim(prediction_latlim)
    ax2.grid(axis='both')
    # save the SECS model prediction
    if save:
        if not os.path.exists(savedir):
            import subprocess
            subprocess.call('mkdir "{}"'.format(savedir), shell=True)
        fig.savefig(savedir + "_SECS" + "{}.png".format(select_time.strftime("%Y%m%d_%H%M")), dpi=200)
        plt.close(fig)
