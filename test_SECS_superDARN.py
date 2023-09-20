# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:36:37 2023

@author: ryanj
"""

import os
#os.add_dll_directory("C://Users/ryanj/OneDrive/Documents/MATLAB/superDARN/pyvirtual/Lib/site-packages/igrf12fort/.libs")

from datetime import datetime
import time
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# import superDARN SECS 
import SECSSD as SD


# temp
import numpy as np
from scipy.io import savemat
# end temp

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
#(all_data, all_time) = SD.read_superDARN(superDARN_data_directory, "v3_grid", start_date, end_date)
(all_data, all_time) = SD.read_superDARN(superDARN_data_directory, "v3_grid", start_date)

# generate the list of prediction points. this is currently a grid, reshaped into a 2D list
prediction_latlim = [46, 80]
prediction_lonlim = [-160, -30]
prediction_lat_step = 1.5
prediction_lon_step = 1.8
# obtain prediction latitude and longitude locations
prediction_latlon = SD.place_prediction(prediction_latlim, prediction_lonlim, prediction_lat_step, prediction_lon_step)


#run a for loop that goes through each time and computes the SECS model for each time
#inside the for loop, it also plots and saves the figure to an output save directory
for i, select_time in enumerate(all_time):
    # select the data to input into SECS
    select_velocity = all_data[0][i]
    select_velocity_latlon= all_data[1][i]
    select_radar_latlon = all_data[2][i]
    select_radar_index = all_data[3][i]
    select_time = select_time[0] # get out of list and into datetime format
    
    print("Running " + datetime.strftime(select_time, "%Y-%m-%d %H:%M") + "..." , end = " ")
    
    # cut the velocity data to only the areas inside the prediction locations
    (select_velocity, select_velocity_latlon) = SD.return_interior_input_velocities(select_velocity, select_velocity_latlon, prediction_latlim, prediction_lonlim)
    
    # generate the variably spaced SECS poles utilizing the input measurements
    
    poles_latlim = [44, 82]
    poles_lonlim = [-162, -28]
    poles_lat_step = 4
    poles_lon_step = 5
    t1 = time.time()
    #(num_poles, poles_latlon) = SD.place_poles(poles_latlim, poles_lonlim, poles_lat_step, poles_lon_step,
    #                                           select_velocity_latlon, density_curvature=1, max_density=0.7, close_tolerance=2)
    (num_poles, poles_latlon) = SD.place_poles(poles_latlim, poles_lonlim, poles_lat_step, poles_lon_step,
                                               select_velocity_latlon, density_curvature=4, max_density=0.5, close_tolerance=1.5)
    t2 = time.time()
    print(str(num_poles) + " poles computed in " + "{:.2f}".format(t2-t1) + " seconds...", end = " ")
    savemat("poles.mat", {'poles': poles_latlon, 'velocity_latlon': select_velocity_latlon})
    
    # run SECS!
    t1 = time.time()
    epsilon = 0.1
    prediction_velocity_frame_pr = SD.predict_with_SECS(select_velocity, select_velocity_latlon, select_radar_latlon, select_radar_index, prediction_latlon, poles_latlon, epsilon=epsilon)
    t2 = time.time()
    print("SECS prediction computed in " + "{:.2f}".format(t2-t1) + " seconds")
    
    # compute the close and far vectors
    (prediction_velocity_close, prediction_velocity_far, prediction_latlon_close, prediction_latlon_far) = \
    SD.compute_close_and_far(prediction_velocity_frame_pr, prediction_latlon, select_velocity_latlon)


    '''
    compute 
    '''

    '''
    plotting section
    '''   
    
    central_lat = np.mean(prediction_latlim)
    central_lon = np.mean(prediction_lonlim)
    
    hfont = {'fontname':'Times New Roman', 'size': 40}
    hfont_axis = {'fontname':'Times New Roman', 'size': 24, 'rotation': 0}
    skip = 1
    fig_size_poles = [30, 40]
    fig_latlim = prediction_latlim
    fig_lonlim = prediction_lonlim
    scale = 30000
    width = 0.0025
    
    pj = ccrs.EquidistantConic(central_lon, central_lat)
    fig = plt.figure(figsize=fig_size_poles)
    ax = plt.axes(projection=pj)
    ax.coastlines()
    ax.set_extent((prediction_lonlim + prediction_latlim), crs=ccrs.PlateCarree())

    # necessary \/*\/
    lat = select_velocity_latlon[::skip, 0]
    lon = select_velocity_latlon[::skip, 1]
    u = select_velocity[::skip, 1]
    v = select_velocity[::skip, 0]
    u_src_crs = u / np.cos(lat / 180 * np.pi)
    v_src_crs = v
    magnitude = np.sqrt(u**2 + v**2)
    magn_src_crs = np.sqrt(u_src_crs**2 + v_src_crs**2)
    # necessary /\./\

    ax.quiver(lon, lat, u_src_crs * magnitude / magn_src_crs, v_src_crs * magnitude / magn_src_crs,
              color="b", width=width, scale=scale, transform=ccrs.PlateCarree(), angles='xy')

    ax.plot(poles_latlon[:, 1], poles_latlon[:, 0], linestyle = "None", marker = ".", ms = 6, mfc = 'r', mec = 'r', transform=ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=1.5, color='gray', alpha=1, linestyle='--')
    gl.top_labels = False
    gl.left_labels = False
    gl.xlabel_style = hfont_axis
    
    
    
    #plot input vectors and SECS output vectors

    fig_size_output = [40, 30]
    pj = ccrs.EquidistantConic(central_lon, central_lat)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size_output, subplot_kw={"projection": pj})
    
    # plot input vectors
    ax1.coastlines()
    ax1.set_extent((prediction_lonlim + prediction_latlim), crs=ccrs.PlateCarree())
    
    # necessary \/*\/
    lat = select_velocity_latlon[::skip, 0]
    lon = select_velocity_latlon[::skip, 1]
    u = select_velocity[::skip, 1]
    v = select_velocity[::skip, 0]
    u_src_crs = u / np.cos(lat / 180 * np.pi)
    v_src_crs = v
    magnitude = np.sqrt(u**2 + v**2)
    magn_src_crs = np.sqrt(u_src_crs**2 + v_src_crs**2)
    # necessary /\./\

    ax1.quiver(lon, lat, u_src_crs * magnitude / magn_src_crs, v_src_crs * magnitude / magn_src_crs,
              color="b", width=width, scale=scale, transform=ccrs.PlateCarree(), angles='xy')
    
    gl_1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=1.5, color='gray', alpha=1, linestyle='--')
    gl_1.top_labels = False
    gl_1.left_labels = False
    gl_1.xlabel_style = hfont_axis
    
    # plot SECS output vectors
    ax2.coastlines()
    ax2.set_extent((prediction_lonlim + prediction_latlim), crs=ccrs.PlateCarree())
    
    # necessary \/*\/
    lat = prediction_latlon_close[::skip, 0]
    lon = prediction_latlon_close[::skip, 1]
    u = prediction_velocity_close[::skip, 1]
    v = prediction_velocity_close[::skip, 0]
    u_src_crs = u / np.cos(lat / 180 * np.pi)
    v_src_crs = v
    magnitude = np.sqrt(u**2 + v**2)
    magn_src_crs = np.sqrt(u_src_crs**2 + v_src_crs**2)
    # necessary /\./\

    ax2.quiver(lon, lat, u_src_crs * magnitude / magn_src_crs, v_src_crs * magnitude / magn_src_crs,
              color="b", width=width, scale=scale, transform=ccrs.PlateCarree(), angles='xy')
    
    # necessary \/*\/
    lat = prediction_latlon_far[::skip, 0]
    lon = prediction_latlon_far[::skip, 1]
    u = prediction_velocity_far[::skip, 1]
    v = prediction_velocity_far[::skip, 0]
    u_src_crs = u / np.cos(lat / 180 * np.pi)
    v_src_crs = v
    magnitude = np.sqrt(u**2 + v**2)
    magn_src_crs = np.sqrt(u_src_crs**2 + v_src_crs**2)
    # necessary /\./\

    ax2.quiver(lon, lat, u_src_crs * magnitude / magn_src_crs, v_src_crs * magnitude / magn_src_crs,
              color="k", width=width/2, scale=scale, transform=ccrs.PlateCarree(), angles='xy')

    gl_2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=1.5, color='gray', alpha=1, linestyle='--')
    gl_2.top_labels = False
    gl_2.left_labels = False
    gl_2.xlabel_style = hfont_axis
    
    ###################


#     # save the figure
# #    if not os.path.exists(savedir):
# #        import subprocess
# #        subprocess.call('mkdir "{}"'.format(savedir), shell=True)
# #    fig.savefig(savedir + "_input" + "{}.png".format(select_time.strftime("%Y%m%d_%H%M")), dpi=200)
# #    plt.close(fig)
    
    
#     # save the SECS model prediction
#     if save:
#         if not os.path.exists(savedir):
#             import subprocess
#             subprocess.call('mkdir "{}"'.format(savedir), shell=True)
#         fig.savefig(savedir + "_SECS" + "{}.png".format(select_time.strftime("%Y%m%d_%H%M")), dpi=200)
#         plt.close(fig)