# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 18:50:26 2023

@author: ryanj
"""


'''
This is the file that contains the functions for performing the spherical elementary current system (SECS) model.
It contains functions for reading in SuperDARN data and the various sub-tasks that must be accomplished in order to properly run SECS.
'''

import numpy as np
import xarray as xr
import os
from datetime import datetime, timedelta
import time

from .perform_SECS import run_secs


def geographic_azimuth_to_radarframe(vector_magnitude, azimuth, direction):
    
    '''
    this function converts the velocity magnitude and radar boresight direction and computes
    the velocity vector in the frame of the radar
    '''
    
    # azimuth is angle from north, going clockwise (negative)
    # convert azimuth to radians
    azimuth = azimuth * np.pi/180
    
    # FRAME: body frame of radar; (north - east - down)
    R = np.hstack([ (np.cos(azimuth))[:, np.newaxis],
                  (np.sin(azimuth))[:, np.newaxis],
                  np.zeros([np.size(azimuth), 1]) ])
    
    # multiply by the magnitude and then the sign of the direction
    radar_velocity_radarframe = vector_magnitude[:, np.newaxis] * direction[:, np.newaxis] * R
    
    return radar_velocity_radarframe



def read_superDARN(directory, datatype, start_time = "none", end_time = "none", time_step = 2):
    '''
    This function reads in the superdarn data, and returns an xarray object that contains the relevant information
    '''
    
    # get the array of times computed
    if isinstance(start_time, str):
        # if there is no user input, then default to the entire day
        start_file = os.listdir(directory)[0]
        start_time = datetime.strptime(start_file[0:8], '%Y%m%d')
        end_time = start_time + timedelta(days=1)
        
    elif not(isinstance(start_time, str)) and isinstance(end_time, str):
        # if the user requests a start date, but provides no end date, then return only one date
        start_file = os.listdir(directory)[0]
        end_time = start_time + timedelta(minutes=time_step)
    
    # create a list of all times to provide velocity outputs
    all_time = np.empty([0, 1])
    all_time = np.vstack((all_time, start_time))
    while all_time[-1] < end_time:
        all_time = np.vstack((all_time, all_time[-1] + timedelta(minutes=time_step)))
        
    # chop off the end of all_time because it is always one larger. if we want to 4:30, we want all_time to end at 4:28
    all_time = all_time[0:-1]
    
    
    # initialize the variables to return
    vel_return = []
    velocity_latlon_return = []
    radar_latlon_return = []
    radar_latlon_index_return = []
    
    if datatype.lower() == "2.5" or datatype.lower() == "3.0":
        # loop through each file in the directory, opening each one and appending the relevant data to the output
        for radar_iteration, filename in enumerate(os.listdir(directory)):
            print("Reading file " + filename)
            t1 = time.time()
            # open the file
            container = xr.load_dataset(os.path.join(directory, filename))
        
            # handle the dates by adding an offset so it can be converted to Python's datetime format
            mjd_offset = np.datetime64('1858-11-17') # an offset parameter to convert the julian date to datetime properly
            radar_time = (container["mjd"].values + mjd_offset).astype('datetime64[s]').astype(datetime) # datetime format of the radar beams
    
            # compute the time difference between the time of radar scan compared to a standard clock interval
            timediff = radar_time - all_time
    
            # generate a 2D logical array that corresponds to the bits of data that go with a specified time
            bool_check = np.logical_and(timediff < timedelta(minutes=time_step), timediff >= timedelta(minutes=0))
    
            # loop over each time and arrange properly
            for i in range(0, np.size(bool_check, 0)):
                # get the logical array for indexing
                bool_select = bool_check[i, :]
                
                # obtain velocity measurements and the beam angles
                vel_select = container["v"][bool_select].values
                beam_number_select = container["beam"][bool_select].values
                bearing_select = container.attrs["brng_at_15deg_el"]
            
                # obtain latitude and longitude of velocity measurements
                lat_select = container["lat"][bool_select].values[:, np.newaxis] # deg
                lon_select = container["lon"][bool_select].values[:, np.newaxis] # deg
            
                # get rid of ground scatter measurements
                bool_notgroundscatter = container["gflg"][bool_select].values == 0
                lat_select = lat_select[bool_notgroundscatter]
                lon_select = lon_select[bool_notgroundscatter]
                vel_select = vel_select[bool_notgroundscatter]
                beam_number_select = beam_number_select[bool_notgroundscatter]
            
                # get azimuth angle
                g_azimuth_angle = bearing_select[beam_number_select]
            
                # compute velocity in N-E-D frame
                vel_radar = geographic_azimuth_to_radarframe(vel_select, g_azimuth_angle, -1 * np.ones(np.shape(vel_select))) # negative one is necessary to maintain proper direction
                
                # get coordinates
                velocity_latlon_select = np.hstack((lat_select, lon_select))
                radar_latlon_select = (container.attrs["lat"], container.attrs["lon"])
                
                # label each radar_latlon point with an index that is the same as the iteration number of the inner for loop
                len_vel = np.size(vel_radar, 0)
                radar_latlon_index_select = np.tile(radar_iteration, (len_vel, 1))
                
                # arrange selected velocities into one structure
                if i >= len(vel_return):
                    vel_return.append(vel_radar)
                    velocity_latlon_return.append(velocity_latlon_select)
                    radar_latlon_return.append(radar_latlon_select)
                    radar_latlon_index_return.append(radar_latlon_index_select)
                else:
                    vel_return[i] = np.vstack((vel_return[i], vel_radar))
                    velocity_latlon_return[i] = np.vstack((velocity_latlon_return[i], velocity_latlon_select))
                    radar_latlon_return[i] = np.vstack((radar_latlon_return[i], radar_latlon_select))
                    radar_latlon_index_return[i] = np.vstack((radar_latlon_index_return[i], radar_latlon_index_select))
                t2 = time.time()
            print("File " + filename + " read in " + "{:.2f}".format(t2-t1) + " seconds.")
        
    elif datatype.lower() == "v3_grid":
        # loop through each file in the directory, opening each one and appending the relevant data to the output
        for radar_iteration, filename in enumerate(os.listdir(directory)):
            print("Reading file " + filename)
            t1 = time.time()
            # open the file
            container = xr.load_dataset(os.path.join(directory, filename))
            
            # handle the dates by adding an offset so it can be converted to Python's datetime format
            mjd_offset = np.datetime64('1858-11-17') # an offset parameter to convert the julian date to datetime properly
            radar_time = (container["mjd_start"].values + mjd_offset).astype('datetime64[s]').astype(datetime) # datetime format of the radar beams
    
            # compute the time difference between the time of radar scan compared to a standard clock interval
            timediff = radar_time - all_time
    
            # generate a 2D logical array that corresponds to the bits of data that go with a specified time
            bool_check = np.logical_and(timediff < timedelta(minutes=time_step), timediff >= timedelta(minutes=0))
    
            # loop over each time and arrange properly
            for i in range(0, np.size(bool_check, 0)):
                # get the logical array for indexing
                bool_select = bool_check[i, :]
                
                lat_select = container["vector.glat"][bool_select].values[:, np.newaxis] # deg
                lon_select = container["vector.glon"][bool_select].values[:, np.newaxis] # deg
                g_azimuth_angle = container["vector.g_kvect"][bool_select].values # azimuth angle [deg]
                weighted_mean_velocity = container["vector.vel.median"][bool_select].values # m/s
                vel_direction_plusminus = container["vector.vel.dirn"][bool_select].values # direction of velocity. +1 away from radar, -1 towards
                
                # compute velocity in N-E-D frame
                vel_select = geographic_azimuth_to_radarframe(weighted_mean_velocity, g_azimuth_angle, vel_direction_plusminus)
                
                # get coordinates
                velocity_latlon_select = np.hstack((lat_select, lon_select))
                radar_latlon_select = (container.attrs["lat"], container.attrs["lon"])
                
                # label each radar_latlon point with an index that is the same as the iteration number of the inner for loop
                len_vel = np.size(vel_select, 0)
                radar_latlon_index_select = np.tile(radar_iteration, (len_vel, 1))
                
                # arrange selected velocities into one structure
                if i >= len(vel_return):
                    vel_return.append(vel_select)
                    velocity_latlon_return.append(velocity_latlon_select)
                    radar_latlon_return.append(radar_latlon_select)
                    radar_latlon_index_return.append(radar_latlon_index_select)
                else:
                    vel_return[i] = np.vstack((vel_return[i], vel_select))
                    velocity_latlon_return[i] = np.vstack((velocity_latlon_return[i], velocity_latlon_select))
                    radar_latlon_return[i] = np.vstack((radar_latlon_return[i], radar_latlon_select))
                    radar_latlon_index_return[i] = np.vstack((radar_latlon_index_return[i], radar_latlon_index_select))
                t2 = time.time()
            print("File " + filename + " read in " + "{:.2f}".format(t2-t1) + " seconds.")
    
    else:
        print("Unknown SuperDARN data format")
        
    # collect the output, and return!
    collected_output = [vel_return, velocity_latlon_return, radar_latlon_return, radar_latlon_index_return]
    
    return (collected_output, all_time)



def place_poles(latlim, lonlim, lat_step, lon_step, velocity_latlon = "none"):
    '''
    this function places poles given the input latitude and longitude limits
    it is encouraged to place the poles larger than the limits of the velocity_latlon locations
    '''
    
    # extract variables
    lat_min = latlim[0]
    lat_max = latlim[1]
    
    lon_min = lonlim[0]
    lon_max = lonlim[1]

    # generate grid of points
    [lat_s, lon_s] = np.meshgrid(np.arange(lat_min, lat_max+0.1, lat_step), np.arange(lon_min, lon_max+0.1, lon_step))

    # this offset is here to prevent poles from being located on top of prediction locations
    offset = 0.3
    lat_s = lat_s - offset
    lon_s = lon_s - offset

    # generate the list of poles (lat, lon)
    poles_latlon = np.hstack((lat_s.reshape([-1, 1]),
                                  lon_s.reshape([-1, 1])))
    return poles_latlon
    


def place_prediction(latlim, lonlim, lat_step, lon_step):
    '''
    this function places prediction points on the ionosphere given input latitude and longitude limits
    it is recommended to place these within a region fully bounded by the input observations
    '''
    
    # extract variables
    lat_min = latlim[0]
    lat_max = latlim[1]
    
    lon_min = lonlim[0]
    lon_max = lonlim[1]

    # generate grid of points
    [lat_pred, lon_pred] = np.meshgrid(np.arange(lat_min, lat_max+0.1, lat_step), np.arange(lon_min, lon_max+0.1, lon_step))

    # generate the list of poles (lat, lon)
    pred_latlon = np.hstack((lat_pred.reshape([-1, 1]),
                                  lon_pred.reshape([-1, 1])))
    
    return pred_latlon



def predict_with_SECS(radar_velocity_radarframe, velocity_latlon, radar_latlon, radar_index, pred_latlon, poles_latlon, select_time, epsilon=0.05):
    '''
    this function serves as the starter function to run the SECS algorithm to predict the plasma flow at various points in the ionosphere
    '''
    
    t1 = time.time()
    # check that the poles do not coincide with a velocity measurement or a prediciton location
    bool_1 = abs(poles_latlon[:, [0]] - velocity_latlon[:, [0]].transpose()) < 0.01
    bool_2 = abs(poles_latlon[:, [1]] - velocity_latlon[:, [1]].transpose()) < 0.01
    
    bool_3 = abs(poles_latlon[:, [0]] - pred_latlon[:, [0]].transpose()) < 0.01
    bool_4 = abs(poles_latlon[:, [1]] - pred_latlon[:, [1]].transpose()) < 0.01
    
    bool_overlap = np.logical_or(np.any(np.logical_and(bool_1, bool_2), 1), np.any(np.logical_and(bool_3, bool_4), 1))
    
    # cut the poles that overlap with either a velocity location or prediction location
    poles_latlon = poles_latlon[np.logical_not(bool_overlap), :]
    
    # run SECS!
    pred_vel_frame_pr = run_secs(radar_velocity_radarframe, velocity_latlon, radar_latlon, radar_index, pred_latlon, poles_latlon, epsilon)
    
    # reduce last dimension of length 1
    pred_vel_frame_pr = pred_vel_frame_pr.squeeze()
    
    t2 = time.time()
    print("Ran SECS Model for " + datetime.strftime(select_time, "%Y-%m-%d %H:%M") + " in " + "{:.2f}".format(t2-t1) + " seconds")
    return pred_vel_frame_pr



def compute_bool_closeto(pred_latlon, closeto_latlon, angular_tolerance=1):
    '''
    this function computes if the positions in pred_latlon are within a tolerance (in degrees) to a list of points given in closeto_latlon.
    returns a boolean matrix that identified TRUE (is close) or FALSE (is not close) corresponding to the entries in pred_latlon
    '''
    # find SECS pred outputs that are within a tolerance to a velocity measurement
    
    closeto_rotate = np.transpose(closeto_latlon)
    phi1 = pred_latlon[:, [0]] * np.pi/180
    phi2 = closeto_rotate[[0], :] * np.pi/180
    delta1 = pred_latlon[:, [1]] * np.pi/180
    delta2 = closeto_rotate[[1], :] * np.pi/180

    angular_separation = np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(delta1 - delta2)

    bool_isclose = np.any(angular_separation > np.cos(angular_tolerance * np.pi/180), 1)
    
    return bool_isclose



def compute_close_and_far(velocity_vectors, pred_latlon, closeto_latlon, angular_tolerance=1):
    '''
    this function computes if the positions in pred_latlon are within a tolerance (in degrees) to a list of points given in closeto_latlon.
    the velocity vectors correspond to pred_latlon
    this function calls compute_closeto, yet it has additional functionality
    returns the list of velocity_vectors that are close, and the list of velocity_vectors that are far
    '''
    # compute the boolean if a prediction velocity_vector location is close to a input velocity 
    bool_isclose = compute_bool_closeto(pred_latlon, closeto_latlon, angular_tolerance)

    # initialize
    velocity_vector_close = np.zeros((np.count_nonzero(bool_isclose), 3))
    velocity_vector_far = np.zeros(((np.count_nonzero(np.logical_not(bool_isclose)), 3)))
    pred_latlon_close = np.zeros((np.count_nonzero(bool_isclose), 2))
    pred_latlon_far = np.zeros(((np.count_nonzero(np.logical_not(bool_isclose)), 2)))

    # set close and far output vectors
    velocity_vector_close[:, 0] = velocity_vectors[bool_isclose, 0]
    velocity_vector_close[:, 1] = velocity_vectors[bool_isclose, 1]
    velocity_vector_close[:, 2] = velocity_vectors[bool_isclose, 2]
    pred_latlon_close[:, 0] = pred_latlon[bool_isclose, 0]
    pred_latlon_close[:, 1] = pred_latlon[bool_isclose, 1]

    velocity_vector_far[:, 0] = velocity_vectors[np.logical_not(bool_isclose), 0]
    velocity_vector_far[:, 1] = velocity_vectors[np.logical_not(bool_isclose), 1]
    velocity_vector_far[:, 2] = velocity_vectors[np.logical_not(bool_isclose), 2]
    pred_latlon_far[:, 0] = pred_latlon[np.logical_not(bool_isclose), 0]
    pred_latlon_far[:, 1] = pred_latlon[np.logical_not(bool_isclose), 1]
    
    return (velocity_vector_close, velocity_vector_far, pred_latlon_close, pred_latlon_far)