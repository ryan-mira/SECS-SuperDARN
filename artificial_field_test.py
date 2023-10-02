# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 16:12:18 2023

@author: ryanj
"""

'''
generate an artificial vectorfield that the SECS system must solve, and then see how accurate the solution is
define TWO sets of input vectors
    1. input only part of the true vectorfield, and see how well SECS reproduces between the gaps
    2. input only part of the true vectorfield THAT THE RADAR WOULD SEE, and see how well SECS reproduces
    
'''

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import time
import matplotlib as mpl
# import SECS
import SECSSD as SD

def compute_vector_components(x, y):
    x = 2 * x
    y = 2 * y
    u = x * np.cos(x + y)
    v = -x * np.cos(x + y) - np.sin(x + y)
    # #u = -x**2 + y**2 - 1
    # #v = -2 * x * y
    # u = 100 * u
    # v = 100 * v
    
    # x = 1/2*x
    # y = 1/2*y
    # u = (np.pi/2) - x
    # v = (np.pi/2) - y
    
    u = 150 * u
    v = 150 * v
    return (u, v)

lat_offset = 55
lon_offset = -30
prediction_latlim = [30 - lat_offset, 80 - lat_offset]
prediction_lonlim = [-130 - lon_offset, -70 - lon_offset]
prediction_lat_step = 1.2#1.5
prediction_lon_step = 2.5#3

# get all prediction locations
prediction_latlon = SD.place_prediction(prediction_latlim, prediction_lonlim, prediction_lat_step, prediction_lon_step)

# get input velocities
# place input velocities using the pole placement algorithm (but we are just placing velocity_latlon points)
# place these input velocities near certain points
np.random.seed(17) # choose seed 17 for good distribution
num_select_points = 10

# initialize
coords_to_nearby = np.zeros((num_select_points, 2))

# generate points that the velocity_latlon will be grouped around.
for i in range(num_select_points):
    coords_to_nearby[i, 0] = np.random.random() * (max(prediction_latlim) - min(prediction_latlim)) + min(prediction_latlim) # latitude
    coords_to_nearby[i, 1] = np.random.random() * (max(prediction_lonlim) - min(prediction_lonlim)) + min(prediction_lonlim) # longitude

# ''hack'' the density function by placing duplicate points on top of eacah other to circumvent the minus 1 in the exponent of the logistic equation
coords_to_nearby = np.vstack((coords_to_nearby, coords_to_nearby))

    
(num_vel, velocity_exact_latlon) = SD.place_poles(prediction_latlim, prediction_lonlim, 5, 5,
                                           coords_to_nearby, density_curvature=100, max_density=1, close_tolerance=5)

# initialize
velocity_exact = np.zeros((num_vel, 3))

# compute the velocity EXACT at the velocity_latlon points according to the vectorfield defined in the function above
for i in range(num_vel):
    (velocity_exact[i, 0], velocity_exact[i, 1]) = compute_vector_components(velocity_exact_latlon[i, 0] * np.pi/180, velocity_exact_latlon[i, 1] * np.pi/180)

# delete points that are all alone
num_close = SD.compute_num_closeto(velocity_exact_latlon, velocity_exact_latlon, angular_tolerance=3)
bool_keep = num_close > 1

velocity_exact = velocity_exact[bool_keep, :]
velocity_exact_latlon = velocity_exact_latlon[bool_keep, :]


# get all prediction locations
prediction_latlon = SD.place_prediction(prediction_latlim, prediction_lonlim, prediction_lat_step, prediction_lon_step)
p_velocity_truth_latlon = prediction_latlon

size_p_velocity_truth = np.size(p_velocity_truth_latlon, 0)
p_velocity_truth = np.zeros((size_p_velocity_truth, 2))
for i in range(size_p_velocity_truth):
    (p_velocity_truth[i, 0], p_velocity_truth[i, 1]) = compute_vector_components(p_velocity_truth_latlon[i, 0] * np.pi/180, p_velocity_truth_latlon[i, 1] * np.pi/180)

np.random.seed() # choose seed 17 for good distribution
'''
SECS section
'''
poles_latlim = [20 - lat_offset, 85 - lat_offset]
poles_lonlim = [-150 - lon_offset, -50 - lon_offset]
poles_lat_step = 4
poles_lon_step = 5

# compute poles!
print("Running " + "..." , end = " ")
t1 = time.time()
#(num_poles, poles_latlon) = SD.place_poles(poles_latlim, poles_lonlim, poles_lat_step/2, poles_lon_step/2)
#(num_poles, poles_latlon) = SD.place_poles(poles_latlim, poles_lonlim, poles_lat_step/2, poles_lon_step/2,
#                                           velocity_exact_latlon, density_curvature=4, max_density=0.5, close_tolerance=0)
(num_poles, poles_latlon) = SD.place_poles(poles_latlim, poles_lonlim, poles_lat_step, poles_lon_step,
                                           velocity_exact_latlon, density_curvature=4, max_density=0.5, close_tolerance=1.5)
t2 = time.time()
print(str(num_poles) + " poles computed in " + "{:.2f}".format(t2-t1) + " seconds...", end = " ")

# run SECS!
t1 = time.time()
epsilon = 0.001
prediction_velocity_frame_pr = SD.predict_with_SECS(velocity_exact, velocity_exact_latlon, prediction_latlon, poles_latlon, epsilon=epsilon)
t2 = time.time()
print("SECS prediction computed in " + "{:.2f}".format(t2-t1) + " seconds")

# compute the close and far vectors
(prediction_velocity_close, prediction_velocity_far, prediction_latlon_close, prediction_latlon_far) = \
SD.compute_close_and_far(prediction_velocity_frame_pr, prediction_latlon, velocity_exact_latlon)

'''
plotting
'''
mpl.rcParams['axes.linewidth'] = 1.75
central_lat = np.mean(prediction_latlim)
central_lon = np.mean(prediction_lonlim)

hfont = {'fontname':'Times New Roman', 'size': 40}
hfont_axis = {'fontname':'Times New Roman', 'size': 24, 'rotation': 0}
skip = 3
fig_size_input = [15, 30]
fig_latlim = prediction_latlim
fig_lonlim = prediction_lonlim
scale = 3500
width = 0.0035
pj = ccrs.EquidistantConic(central_lon, central_lat)


fig = plt.figure(figsize=fig_size_input)
ax = plt.axes(projection=pj)
ax.coastlines("50m", linewidth=1.5)
ax.set_extent((prediction_lonlim + prediction_latlim), crs=ccrs.PlateCarree())


# necessary \/*\/
lat = velocity_exact_latlon[::skip, 0]
lon = velocity_exact_latlon[::skip, 1]
u = velocity_exact[::skip, 1]
v = velocity_exact[::skip, 0]
u_src_crs = u / np.cos(lat / 180 * np.pi)
v_src_crs = v
magnitude = np.sqrt(u**2 + v**2)
magn_src_crs = np.sqrt(u_src_crs**2 + v_src_crs**2)
# necessary /\./\

ax.quiver(lon, lat, u_src_crs * magnitude / magn_src_crs, v_src_crs * magnitude / magn_src_crs,
          color="b", width=width, scale=scale, transform=ccrs.PlateCarree(), angles='xy')
ax.set_title("SECS Input Velocities", **hfont)



####################


fig = plt.figure(figsize=fig_size_input)
ax = plt.axes(projection=pj)
ax.coastlines("50m", linewidth=1.5)
ax.set_extent((prediction_lonlim + prediction_latlim), crs=ccrs.PlateCarree())


# necessary \/*\/
lat = p_velocity_truth_latlon[::skip, 0]
lon = p_velocity_truth_latlon[::skip, 1]
u = p_velocity_truth[::skip, 1]
v = p_velocity_truth[::skip, 0]
u_src_crs = u / np.cos(lat / 180 * np.pi)
v_src_crs = v
magnitude = np.sqrt(u**2 + v**2)
magn_src_crs = np.sqrt(u_src_crs**2 + v_src_crs**2)
# necessary /\./\

ax.quiver(lon, lat, u_src_crs * magnitude / magn_src_crs, v_src_crs * magnitude / magn_src_crs,
          color="b", width=width, scale=scale, transform=ccrs.PlateCarree(), angles='xy')
ax.set_title("Test Vectorfield, Truth Data", **hfont)


####################


fig = plt.figure(figsize=fig_size_input)
ax = plt.axes(projection=pj)
ax.coastlines("50m", linewidth=1.5)
ax.set_extent((prediction_lonlim + prediction_latlim), crs=ccrs.PlateCarree())

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
          color="k", width=width/2, scale=scale, transform=ccrs.PlateCarree(), angles='xy')
ax.set_title("SECS Reconstruction", **hfont)

































# #pj = ccrs.Orthographic(central_lon, central_lat)
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size_input, subplot_kw={"projection": pj})

# # plot truth vectorfield
# ax1.coastlines()
# ax1.set_extent((prediction_lonlim + prediction_latlim), crs=ccrs.PlateCarree())

# # necessary \/*\/
# lat = p_velocity_truth_latlon[:, 0]
# lon = p_velocity_truth_latlon[:, 1]
# u = p_velocity_truth[:, 1]
# v = p_velocity_truth[:, 0]
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

# ax2.coastlines()
# ax2.set_extent((prediction_lonlim + prediction_latlim), crs=ccrs.PlateCarree())
# # necessary \/*\/
# lat = velocity_exact_latlon[:, 0]
# lon = velocity_exact_latlon[:, 1]
# u = velocity_exact[:, 1]
# v = velocity_exact[:, 0]
# u_src_crs = u / np.cos(lat / 180 * np.pi)
# v_src_crs = v
# magnitude = np.sqrt(u**2 + v**2)
# magn_src_crs = np.sqrt(u_src_crs**2 + v_src_crs**2)
# # necessary /\./\
    
# ax2.quiver(lon, lat, u_src_crs * magnitude / magn_src_crs, v_src_crs * magnitude / magn_src_crs,
#           color="b", width=width, scale=scale, transform=ccrs.PlateCarree(), angles='xy')
# ax2.plot(poles_latlon[:, 1], poles_latlon[:, 0], linestyle = "None", marker = ".", ms = 6, mfc = 'r', mec = 'r', transform=ccrs.PlateCarree())
# gl_2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
#               linewidth=1.5, color='gray', alpha=1, linestyle='--')
# gl_2.top_labels = False
# gl_2.left_labels = False
# gl_2.xlabel_style = hfont_axis
    


# ############################

# #plot input vectors and SECS output vectors

# fig_size_output = [40, 30]
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size_output, subplot_kw={"projection": pj})

# # plot input vectors
# ax1.coastlines()
# ax1.set_extent((prediction_lonlim + prediction_latlim), crs=ccrs.PlateCarree())

# # necessary \/*\/
# lat = velocity_exact_latlon[::skip, 0]
# lon = velocity_exact_latlon[::skip, 1]
# u = velocity_exact[::skip, 1]
# v = velocity_exact[::skip, 0]
# u_src_crs = u / np.cos(lat / 180 * np.pi)
# v_src_crs = v
# magnitude = np.sqrt(u**2 + v**2)
# magn_src_crs = np.sqrt(u_src_crs**2 + v_src_crs**2)
# # necessary /\./\

# ax1.quiver(lon, lat, u_src_crs * magnitude / magn_src_crs, v_src_crs * magnitude / magn_src_crs,
#           color="b", width=width, scale=scale, transform=ccrs.PlateCarree(), angles='xy')
# ax1.plot(coords_to_nearby[:, 1], coords_to_nearby[:, 0], linestyle = "None", marker = ".", ms = 6, mfc = 'r', mec = 'r', transform=ccrs.PlateCarree())

# gl_1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
#               linewidth=1.5, color='gray', alpha=1, linestyle='--')
# gl_1.top_labels = False
# gl_1.left_labels = False
# gl_1.xlabel_style = hfont_axis

# # plot SECS output vectors
# ax2.coastlines()
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
    

