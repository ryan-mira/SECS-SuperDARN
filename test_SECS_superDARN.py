# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:36:37 2023

@author: ryanj
"""

import os
from datetime import datetime
import time
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# import superDARN SECS 
import SECSSD as SD

import matplotlib as mpl
# temp
import numpy as np
from scipy.io import savemat
# end temp

# input directory containing one day superDARN data (only one day currently, and as many files as there are radars)
superDARN_data_directory = "northern_2015_march_1" + os.sep

# directory to save the data
save = 0
savedir = "output_figures" + os.sep


# define the starting time and ending time.
# specify a select range to reduce the amount of for loop iterations, and therefore computation time
# can, instead, only select a single date by inputting only a start time, and the superdarn file reader function will return every time in the file
start_date = datetime(2015, 3, 1, 5, 0)
end_date = datetime(2015, 3, 1, 4, 10)

# read the data
# specify the grid as "v3_grid" because this is the option selected in the download data webpage located at https://superdarn.jhuapl.edu/download/
#(all_data, all_time) = SD.read_superDARN(superDARN_data_directory, "v3_grid", start_date, end_date)
(all_data, all_time) = SD.read_superDARN(superDARN_data_directory, "v3_grid", start_date)

# generate the list of prediction points. this is currently a grid, reshaped into a 2D list
prediction_latlim = [25, 89]
#prediction_lonlim = [-170, -10]
prediction_lonlim = [-220, -10]
prediction_lat_step = 1.5
prediction_lon_step = 1.8

# obtain prediction latitude and longitude locations
#prediction_latlon = SD.place_prediction(prediction_latlim, prediction_lonlim, prediction_lat_step, prediction_lon_step)


#run a for loop that goes through each time and computes the SECS model for each time
#inside the for loop, it also plots and saves the figure to an output save directory
for i, select_time in enumerate(all_time):
    # select the data to input into SECS
    select_velocity = all_data[0][i]
    select_velocity_latlon = all_data[1][i]
    select_time = select_time[0] # get out of list and into datetime format
    
    print("Running " + datetime.strftime(select_time, "%Y-%m-%d %H:%M") + "..." , end = " ")
    
    # cut the velocity data to only the areas inside the prediction locations
    #(select_velocity, select_velocity_latlon) = SD.return_interior_input_velocities(select_velocity, select_velocity_latlon, prediction_latlim, prediction_lonlim)
    
    # generate the variably spaced SECS poles utilizing the input measurements
    
    #poles_latlim = [30, 80]
    poles_latlim = [30, 89]
    poles_lonlim = [-220, -20]
    poles_lat_step = 4
    poles_lon_step = 5
    t1 = time.time()
    #(num_poles, poles_latlon) = SD.place_poles(poles_latlim, poles_lonlim, poles_lat_step/1.7, poles_lon_step/1.7)
    #(num_poles, poles_latlon) = SD.place_poles(poles_latlim, poles_lonlim, 1.25, 1.25,
    #                                           select_velocity_latlon, density_curvature=1, max_density=0.5, close_tolerance=0)

    t1 = time.time()
    #(num_poles, poles_latlon) = SD.place_poles(poles_latlim, poles_lonlim, poles_lat_step/1.5, poles_lon_step/1.5,
    #                                           select_velocity_latlon, density_curvature=4, max_density=0.7, close_tolerance=1.0)
    (num_poles, poles_latlon) = SD.place_poles(poles_latlim, poles_lonlim, poles_lat_step/2, poles_lon_step/2,
                                               select_velocity_latlon, density_curvature=4, max_density=0.5, close_tolerance=1.5)
    t2 = time.time()
    print(t2-t1)

    
    t2 = time.time()
    print(str(num_poles) + " poles computed in " + "{:.2f}".format(t2-t1) + " seconds...", end = " ")
    #savemat("poles.mat", {'poles': poles_latlon, 'velocity_latlon': select_velocity_latlon})
    
    (_, prediction_latlon) = SD.place_poles(prediction_latlim, prediction_lonlim, prediction_lat_step/1.7, prediction_lon_step/1.7,
                                            select_velocity_latlon, density_curvature=4, max_density=0.5, close_tolerance=0)
    
    # run SECS!
    t1 = time.time()
    epsilon = 0.1
    prediction_velocity_frame_pr = SD.predict_with_SECS(select_velocity, select_velocity_latlon, prediction_latlon, poles_latlon, epsilon=epsilon)
    t2 = time.time()
    print("SECS prediction computed in " + "{:.2f}".format(t2-t1) + " seconds")
   
    # no data in select_velocity
    if np.all(prediction_velocity_frame_pr == 0):
        continue
    
    # compute the close and far vectors
    (prediction_velocity_close, prediction_velocity_far, prediction_latlon_close, prediction_latlon_far) = \
    SD.compute_close_and_far(prediction_velocity_frame_pr, prediction_latlon, select_velocity_latlon)




    '''
    plotting section
    '''   
    mpl.rcParams['axes.linewidth'] = 2.5
    central_lat = np.mean(prediction_latlim)
    central_lon = np.mean(prediction_lonlim)
    legend_font = {"family":"Times New Roman", "size": 46}
    
    hfont = {'fontname':'Times New Roman', 'size': 64}
    hfont_axis = {'fontname':'Times New Roman', 'size': 24, 'rotation': 0}
    skip = 2
    fig_size_poles = [25, 25]
    fig_size_output = [25, 25]
    fig_latlim = prediction_latlim
    fig_lonlim = prediction_lonlim
    scale = 16000
    width = 0.0035
    
    p_latlim = [40, 90]
    p_lonlim = [-158, -68]
    
    #pj = ccrs.EquidistantConic(central_lon, central_lat)
    pj = ccrs.NearsidePerspective(satellite_height=20500000.0, central_longitude=central_lon, central_latitude=central_lat)
    
    ##########################
    ############## PLOT INPUT
    
    
    fig = plt.figure(figsize=fig_size_output)
    ax = plt.axes(projection=pj)
    ax.coastlines("50m", linewidth=1.5)
    ax.set_extent((p_lonlim + p_latlim), crs=ccrs.PlateCarree())
    #ax.set_extent(([-140, -60] + [34, 70]), crs=ccrs.PlateCarree())

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

    ax.set_title("SuperDARN LOS Velocity Measurements", **hfont)
    
    plt.text(.99, .01, datetime.strftime(select_time, "%Y-%m-%d %H:%M" + " UTC"), ha='right', va='bottom', transform=ax.transAxes, **legend_font)
    
    ##########################
    ############## PLOT OUTPUT
    
    fig = plt.figure(figsize=fig_size_output)
    ax = plt.axes(projection=pj)
    ax.coastlines("50m", linewidth=1.5)
    ax.set_extent((p_lonlim + p_latlim), crs=ccrs.PlateCarree())

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

    ax.quiver(lon, lat, u_src_crs * magnitude / magn_src_crs, v_src_crs * magnitude / magn_src_crs,
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

    ax.quiver(lon, lat, u_src_crs * magnitude / magn_src_crs, v_src_crs * magnitude / magn_src_crs,
              color="#404040", width=width/2, scale=scale, transform=ccrs.PlateCarree(), angles='xy')
    ax.set_title("SECS Reconstruction, Epsilon = 0.5", **hfont)
    
    plt.text(.99, .01, datetime.strftime(select_time, "%Y-%m-%d %H:%M" + " UTC"), ha='right', va='bottom', transform=ax.transAxes, **legend_font)


    ##########################
    ############## PLOT POLES

    ms = 16
    fig = plt.figure(figsize=fig_size_output)
    ax = plt.axes(projection=pj)
    ax.coastlines("50m", linewidth=1.5)
    ax.set_extent((p_lonlim + p_latlim), crs=ccrs.PlateCarree())

    # plot poles and velocity locations, respectively
    ax.plot(poles_latlon[:, 1], poles_latlon[:, 0], linestyle='None', marker='.', mec = 'r', mfc = 'r', markersize=ms, transform=ccrs.PlateCarree())
    ax.plot(select_velocity_latlon[:, 1], select_velocity_latlon[:, 0], linestyle='None', marker='.', mec = 'k', mfc = 'k', markersize=ms, transform=ccrs.PlateCarree())

    ax.plot(0, 0, linestyle='None', marker='.', mec = 'r', mfc = 'r', markersize=ms*2, label="Pole Locations", transform=ccrs.PlateCarree())
    ax.plot(0, 0, linestyle='None', marker='.', mec = 'k', mfc = 'k', markersize=ms*2, label="Input Velocity Locations", transform=ccrs.PlateCarree())

    ax.legend(prop=legend_font)

    ax.set_title("Adaptive Pole Placement", **hfont)
    
    plt.text(.99, .01, datetime.strftime(select_time, "%Y-%m-%d %H:%M" + " UTC"), ha='right', va='bottom', transform=ax.transAxes, **legend_font)










# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size_output, subplot_kw={"projection": pj})

# # plot input vectors
# ax1.coastlines("50m")
# ax1.set_extent((prediction_lonlim + prediction_latlim), crs=ccrs.PlateCarree())

# # necessary \/*\/
# lat = select_velocity_latlon[::skip, 0]
# lon = select_velocity_latlon[::skip, 1]
# u = select_velocity[::skip, 1]
# v = select_velocity[::skip, 0]
# u_src_crs = u / np.cos(lat / 180 * np.pi)
# v_src_crs = v
# magnitude = np.sqrt(u**2 + v**2)
# magn_src_crs = np.sqrt(u_src_crs**2 + v_src_crs**2)
# # necessary /\./\

# ax1.quiver(lon, lat, u_src_crs * magnitude / magn_src_crs, v_src_crs * magnitude / magn_src_crs,
#           color="b", width=width, scale=scale, transform=ccrs.PlateCarree(), angles='xy')

# gl_1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
#               linewidth=1.5, color='gray', alpha=1, linestyle='--')
# gl_1.top_labels = False
# gl_1.left_labels = False
# gl_1.xlabel_style = hfont_axis

# # plot SECS output vectors
# ax2.coastlines("50m")
# ax2.set_extent((prediction_lonlim + prediction_latlim), crs=ccrs.PlateCarree())

# # necessary \/*\/
# lat = prediction_latlon_close[::skip, 0]
# lon = prediction_latlon_close[::skip, 1]
# u = prediction_velocity_close[::skip, 1]
# v = prediction_velocity_close[::skip, 0]
# u_src_crs = u / np.cos(lat / 180 * np.pi)
# v_src_crs = v
# magnitude = np.sqrt(u**2 + v**2)
# magn_src_crs = np.sqrt(u_src_crs**2 + v_src_crs**2)
# # necessary /\./\

# ax2.quiver(lon, lat, u_src_crs * magnitude / magn_src_crs, v_src_crs * magnitude / magn_src_crs,
#           color="b", width=width, scale=scale, transform=ccrs.PlateCarree(), angles='xy')

# # necessary \/*\/
# lat = prediction_latlon_far[::skip, 0]
# lon = prediction_latlon_far[::skip, 1]
# u = prediction_velocity_far[::skip, 1]
# v = prediction_velocity_far[::skip, 0]
# u_src_crs = u / np.cos(lat / 180 * np.pi)
# v_src_crs = v
# magnitude = np.sqrt(u**2 + v**2)
# magn_src_crs = np.sqrt(u_src_crs**2 + v_src_crs**2)
# # necessary /\./\

# ax2.quiver(lon, lat, u_src_crs * magnitude / magn_src_crs, v_src_crs * magnitude / magn_src_crs,
#           color="k", width=width/2, scale=scale, transform=ccrs.PlateCarree(), angles='xy')

# gl_2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
#               linewidth=1.5, color='gray', alpha=1, linestyle='--')
# gl_2.top_labels = False
# gl_2.left_labels = False
# gl_2.xlabel_style = hfont_axis

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