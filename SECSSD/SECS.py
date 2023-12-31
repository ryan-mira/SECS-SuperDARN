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
import glob
from dateutil import parser
from .perform_SECS import run_secs
from .bridsonVariableRadius import poissonDiskSampling

def mjd2datetime(mjd):
    mjd_offset = np.datetime64('1858-11-17') # an offset parameter to convert the julian date to datetime properly
    dt = (mjd + mjd_offset).astype('datetime64[s]').astype(datetime) # datetime format of the radar beams
    return dt

def round2minute(dt, roundTo=60):
    """Round a datetime object to any time lapse in seconds
    dt : datetime.datetime object, default now.
    roundTo : Closest number of seconds to round to, default 1 minute.
    """
    def _convert(t):
        seconds = (t.replace(tzinfo=None) - t.min).seconds
        rounding = (seconds+roundTo/2) // roundTo * roundTo
        return t + timedelta(0,rounding-seconds,-t.microsecond)
    if isinstance(dt, datetime):
        dtt = _convert(dt)
    else:
        dtt = np.array([_convert(t) for t in dt])

    return dtt  

def geographic_azimuth_to_velframe(vector_magnitude, azimuth, direction):
    
    '''
    this function converts the velocity magnitude and radar boresight direction and computes
    the velocity vector in the frame of the radar
    radar frame is defined to be (N)orth - (E)ast - (D)own -- NED
    
    this function is internal to SECS.py, and it does not need to be called outside it
    
    azimuth is RELATIVE TO THE VELOCITY LAT/LON POINT!
    the bearing of a geodetic line will be different on point A compared to point B...
    this has already been accounted for in the superDARN data files
    '''
    # FRAME: body frame of radar; (north - east - down)
    R = np.hstack([ (np.cos(np.radians(azimuth)))[:, np.newaxis],
                  (np.sin(np.radians(azimuth)))[:, np.newaxis],
                  np.zeros([azimuth.shape[0], 1]) ])
    
    # multiply by the magnitude and then the sign of the direction
    radar_velocity_velframe = vector_magnitude[:, np.newaxis] * direction[:, np.newaxis] * R
    
    return radar_velocity_velframe

def read_superDARN(directory, start_time=None, end_time = None, datatype='v3_grid', time_step = 2):
    '''
    This function reads in the superdarn data, and returns an farray object that contains the relevant information
    output is collected_output and collected_time
    
    INPUTS:
        directory - must contain only superDARN files. these files MUST be for a single day only.
        datatype - a string that is "2.5", "3.0", or "v3_grid". Match this to the selected option on the download page
        start_time - python datetime format of the selected time. if no input, then defaults to the beginning on the day located in the input files
        end_time - python datetime format of the end time. if no input, then defaults to the end of the day located in the input files
        time_step - amount of time in minutes that separates each radar scan. can be used to collect more radar sweeps into one agglomerated "time"
        
        if no start or end time is given, then function will return entire day
        if only a start time is given, then function will return a 2 minute slice of all the files starting (not centered) at that start time
        if both a start and end time is given, then function will return slices of all files starting and ending at these times. the last time END at the end time.
        
    OUTPUTS:
        collected_output - list of 4 long:
            collected_output[0] - all the superDARN measured velocities. this itself is a list that organized by times. each list entry corresponds to a single time
                collected_output[0][0] is all the velocities at the first time numpy array of floats. dims [num velocity f 3 components (f, y, z)]
                    all velocities are in the radar frame, North-East-Down (NED) frame

            collected_output[1] - all the lat/lon locations of the velocity returns. length [number_of_times]
                collected_output[1][0] is all the lat/lon locations at the first time. numpy array of floats. dims [num velocity f 2 (lat, lon in degrees)]
            
            collected_output[2] - lat/lon of all the radars that measure velocity. length [number_of_times]
                collected_output[2][0] is all the lat/lon of the radars that measured velocity in the first time. numpy array of floats. dims [num radar f 2 (lat, lon in degrees)]

            collected_output[3] - an indef that contains the radar number, corresponding to collected_output[2], for a velocity measurement. it tells which radars measured which velocity measurement. length [number_of_times]
                collected_output[3][0] are the indices that correspond to the first time. dims [num velocity f 1]

        all_time is list that a list of length number_of_times. this contains the datetime.datetime of each selected time. length [number_of_times]
            the length of this list is equal to the length of collected_output[0], collected_output[1], collected_output[2], and collected_output[3]            
    '''
    
    # get the array of times computed
    if start_time is None:
        # if there is no user input, then default to the entire day of the first file in the directory
        start_file = os.listdir(directory)[0]
        start_time = datetime.strptime(start_file[0:8], '%Y%m%d')
        end_time = start_time + timedelta(days=1)
    if isinstance(start_time, str):
        # If the start_time is provided as string, try the conversion to datetime
        start_time = parser.parse(start_time)
    if isinstance(end_time, str):
        # If end)time is provided as str, try the conversion to datetime
        end_time = parser.parse(end_time)
        
    if end_time is None:
        # if the user requests a start date, but provides no end date, then return only one date
        end_time = start_time + timedelta(minutes=time_step)
    
    assert isinstance(start_time, datetime) and isinstance(end_time, datetime)
    
    # Filter the available files by the dates
    start_date = datetime(start_time.year, start_time.month, start_time.day)
    end_date = datetime(start_time.year, end_time.month, end_time.day)
    #Read all files in directory
    files = np.array(sorted(glob.glob(directory + '*.nc')))
    #Get the corresponding dates for each file
    file_dates = np.array([datetime.strptime(os.path.split(f)[1][:8], "%Y%m%d") for f in files])
    #What are good files?
    idd = (file_dates >= start_date) & (file_dates <= end_date)
    #Filte the file names
    good_files = files[idd]
    
    # create a list of all times to provide velocity outputs
    all_time = np.expand_dims(np.arange(start_time, end_time, timedelta(minutes=time_step)).astype('datetime64[s]').astype(datetime), axis=1)
    # return

    # initialize the variables to return
    vel_return = []
    velocity_latlon_return = []
    radar_latlon_return = []
    radar_latlon_indef_return = []
    
    if datatype.lower() == "2.5" or datatype.lower() == "3.0":
        # loop through each file in the directory, opening each one and appending the relevant data to the output
        for ir, f in enumerate(good_files):
            print("Reading file " + f)
            t1 = time.time()
            # open the file
            container = xr.load_dataset(f)
        
            # handle the dates by adding an offset so it can be converted to Python's datetime format
            # mjd_offset = np.datetime64('1858-11-17') # an offset parameter to convert the julian date to datetime properly
            # radar_time = (container["mjd"].values + mjd_offset).astype('datetime64[s]').astype(datetime) # datetime format of the radar beams
            radar_time = round2minute(mjd2datetime(container["mjd"].values))
            # idtt = (radar_time >= start_time) & (radar_time <= end_time)
            # radar_time = radar_time[idtt]
            print (radar_time.shape, all_time.shape)
            # compute the time difference between the time of radar scan compared to a standard clock interval
            timediff = radar_time - all_time
    
            # generate a 2D logical array that corresponds to the bits of data that go with a specified time
            bool_check = np.logical_and(timediff < timedelta(minutes=time_step), timediff >= timedelta(minutes=0))
    
            # loop over each time and arrange properly
            for i in range(0, np.size(bool_check, 0)):
                # get the logical array for indefing
                bool_select = bool_check[i, :]
                
                # obtain velocity measurements and the beam angles
                vel_select = container["v"][bool_select].values
                beam_number_select = container["beam"][bool_select].values
                bearing_select = container.attrs["brng_at_15deg_el"]
            
                # obtain latitude and longitude of velocity measurements
                lat_select = container["lat"][bool_select].values[:, np.newafis] # deg
                lon_select = container["lon"][bool_select].values[:, np.newafis] # deg
            
                # get rid of ground scatter measurements
                bool_notgroundscatter = container["gflg"][bool_select].values == 0
                lat_select = lat_select[bool_notgroundscatter]
                lon_select = lon_select[bool_notgroundscatter]
                vel_select = vel_select[bool_notgroundscatter]
                beam_number_select = beam_number_select[bool_notgroundscatter]
            
                # get azimuth angle
                g_azimuth_angle = bearing_select[beam_number_select]
            
                # compute velocity in N-E-D frame
                vel_radar = geographic_azimuth_to_velframe(vel_select, g_azimuth_angle, -1 * np.ones(np.shape(vel_select))) # negative one is necessary to maintain proper direction
                
                # get coordinates
                velocity_latlon_select = np.hstack((lat_select, lon_select))
                radar_latlon_select = (container.attrs["lat"], container.attrs["lon"])
                
                # label each radar_latlon point with an indef that is the same as the iteration number of the inner for loop
                len_vel = np.size(vel_radar, 0)
                radar_latlon_indef_select = np.tile(ir, (len_vel, 1))
                
                # arrange selected velocities into one structure
                if i >= len(vel_return):
                    vel_return.append(vel_radar)
                    velocity_latlon_return.append(velocity_latlon_select)
                    radar_latlon_return.append(radar_latlon_select)
                    radar_latlon_indef_return.append(radar_latlon_indef_select)
                else:
                    vel_return[i] = np.vstack((vel_return[i], vel_radar))
                    velocity_latlon_return[i] = np.vstack((velocity_latlon_return[i], velocity_latlon_select))
                    radar_latlon_return[i] = np.vstack((radar_latlon_return[i], radar_latlon_select))
                    radar_latlon_indef_return[i] = np.vstack((radar_latlon_indef_return[i], radar_latlon_indef_select))
                t2 = time.time()
            print("\tFile " + f + " read in " + "{:.2f}".format(t2-t1) + " seconds.")
        
    elif datatype.lower() == "v3_grid":
        # loop through each file in the directory, opening each one and appending the relevant data to the output
        for ir, f in enumerate(good_files):
            print("Reading file " + f)
            t1 = time.time()
            # open the file
            X = xr.load_dataset(f)
            
            # handle the dates by adding an offset so it can be converted to Python's datetime format
            radar_time = round2minute(mjd2datetime(X["mjd_start"].values))
            # compute the time difference between the time of radar scan compared to a standard clock interval
            # mjd_offset = np.datetime64('1858-11-17') # an offset parameter to convert the julian date to datetime properly
            # radar_time = (X["mjd_start"].values + mjd_offset).astype('datetime64[s]').astype(datetime) # datetime format of the radar beams
            timediff = radar_time - all_time
            # imediff1 = radar_time - all_time
            # print(imediff)
            # return
            # generate a 2D logical array that corresponds to the bits of data that go with a specified time
            bool_check = np.logical_and(timediff < timedelta(minutes=time_step), timediff >= timedelta(minutes=0))
    
            # loop over each time and arrange properly
            for i in range(0, np.size(bool_check, 0)):
                # get the logical array for indefing
                idt = bool_check[i, :]
                
                lat_select = X["vector.glat"][idt].values[:, np.newaxis] # deg
                lon_select = X["vector.glon"][idt].values[:, np.newaxis] # deg
                azimuth = X["vector.g_kvect"][idt].values # azimuth angle [deg]
                velocity_magnitude = X["vector.vel.median"][idt].values # m/s
                velocity_direction = X["vector.vel.dirn"][idt].values # direction of velocity. +1 away from radar, -1 towards
                
                # get coordinates
                velocity_latlon_select = np.hstack((lat_select, lon_select))
                radar_latlon_select = (X.attrs["lat"], X.attrs["lon"])
                
                # compute velocity in N-E-D frame
                vel_ned = geographic_azimuth_to_velframe(velocity_magnitude, azimuth, velocity_direction)
                
                # label each radar_latlon point with an indef that is the same as the iteration number of the inner for loop
                # len_vel = vel_ned.shape[0]#np.size(vel_select, 0)
                radar_latlon_indef_select = np.tile(ir, (vel_ned.shape[0], 1))
                
                # arrange selected velocities into one structure
                if i >= len(vel_return):
                    vel_return.append(vel_ned)
                    velocity_latlon_return.append(velocity_latlon_select)
                    radar_latlon_return.append(np.array(radar_latlon_select)[np.newaxis, :])
                    radar_latlon_indef_return.append(radar_latlon_indef_select)
                else:
                    vel_return[i] = np.vstack((vel_return[i], vel_ned))
                    velocity_latlon_return[i] = np.vstack((velocity_latlon_return[i], velocity_latlon_select))
                    radar_latlon_return[i] = np.vstack((radar_latlon_return[i], radar_latlon_select))
                    radar_latlon_indef_return[i] = np.vstack((radar_latlon_indef_return[i], radar_latlon_indef_select))
                t2 = time.time()
            print("\tFile " + f + " read in " + "{:.2f}".format(t2-t1) + " seconds.")
    
    else:
        print("Unknown SuperDARN data format")
        
    # collect the output, and return!
    collected_output = [vel_return, velocity_latlon_return, radar_latlon_return, radar_latlon_indef_return]
    
    return (collected_output, all_time)



def place_poles(latlim, lonlim, lat_step, lon_step, velocity_latlon = "none", density_curvature=1, max_density=0.5, close_tolerance=2):
    '''
    this function places poles given the input latitude and longitude limits
    it is encouraged to place the poles larger than the limits of the velocity_latlon locations
    
    currently, this function computes a meshgrid and outputs a 2D list
    
    INPUTS:
        latlim - latitude limits [lower, upper]
        lonlim - longitude limits [lower, upper]
        lat_step - the spacing in degrees of the meshgrid in the latitude direction
        lon_step - the spacing in degrees of the meshgrid in the longitude direction
        velocity_latlon - used to inform the placement of poles in a variable density placement scheme. dims [number of velocity f 2 (lat, lon)]
            default: "none", which means the poles are just a meshgrid of points put into a 2D list
            IF velocity_latlon is a list of locations, the poles will be variably spaced, clustering around these lat/lon points.
                the density far from the points is determined by the AVERAGE of lat_step and lon_step
        density_strength - how much eftra density the profimity to input measuremeents causes
            IF velocity_latlon is "none", this does nothing
        maf_density - the lowest radius (most dense) the poles can be spaced
            IF velocity_latlon is "none", this does nothing
        close_tolerance - the radius of effect that an input measurement has
            IF velocity_latlon is "none", this does nothing
    
    OUTPUTS:
        poles_latlon - a 2D array containing all the lat/lon points. this list is 2D because the lat and lon meshgrids are reshaped to a 1D column vector and placed pair-wise
        dims: [number of pole points f 2 (lat, lon in degrees)]
    '''
    
    # eftract variables
    lat_min = np.floor(latlim[0])
    lat_max = np.floor(latlim[1])
    
    lon_min = np.floor(lonlim[0])
    lon_max = np.floor(lonlim[1])

    # if this is TRUE, then run a simple meshgrid of points
    if isinstance(velocity_latlon, str):
        # generate grid of points according to the spacing inputted above
        [lat_s, lon_s] = np.meshgrid(np.arange(lat_min, lat_max+0.1, lat_step), np.arange(lon_min, lon_max+0.1, lon_step))
        
        # this offset is here to prevent poles from being located on top of prediction locations
        # it isn't the greatest nor the most robust, but it works decently enough
        offset = 0.3
        lat_s = lat_s - offset
        lon_s = lon_s - offset
        # reshape the above meshgrid into a 2D list of poles (lat, lon)
        poles_latlon = np.hstack((lat_s.reshape([-1, 1]),
                                      lon_s.reshape([-1, 1])))
        # efit function
        return (np.size(poles_latlon, 0), poles_latlon)
    
    # otherwise, continue efecuting the code
    # velocity_latlon is given as a list, and place the SECS poles in a variable-spacing, variable-density manner

    # generate grid of points. this is a meshgrid of 1f1 degree spacing in (lat f lon)
    [lat_s, lon_s] = np.meshgrid(np.arange(lat_min, lat_max+0.1, 1), np.arange(lon_min, lon_max+0.1, 1))

    # reshape the above meshgrid into a 2D list of poles (lat, lon)
    poles_latlon_initial = np.hstack((lat_s.reshape([-1, 1]),
                                  lon_s.reshape([-1, 1])))

    # for each entry in the list, compute the number of pole stations within a certain angular tolerance
    # this informs the (discrete) density function of the input measurements,
    # which is used to determine the (discrete) output of variably-spaced, variably-dense SECS poles
    num_close = compute_num_closeto(poles_latlon_initial, velocity_latlon, angular_tolerance=close_tolerance)[:, np.newaxis]

    # input discrete density function
    # units are radius in degrees of each point
    # each entry specifies the approfimate number of degrees the resulting poles would like to be from each other at the specific corresponding lat/lon
    # this varies for each lat/lon, and the poles will distribute themselves accordingly
    global_step = (lat_step + lon_step) / 2 # the global density step
    density_input = np.zeros((np.size(poles_latlon_initial, 0), 1)) # initialize
    

    # set the density input using a mask and a mathematical function
    # using a slightly modified logistic function to generate nicely spaced density gradients
    density_input[num_close == 0] = global_step # radius spacing of poles far from input measurements
    density_input[num_close != 0] = 1.5*(global_step - max_density) / (1 + np.e**(0.4*density_curvature * (num_close[num_close != 0] - 1))) + max_density # radius spacing for poles near input measurements


    # run the poisson disk sampling algorithm on a shell surface to place the SECS poles
    num_iterations = 40 # this is the number of attempts to place a point at a new location. don't change this; it does not make things more "accurate". 

    # perform the poisson disk sampling algorithm.
    # THANK YOU ADRIAN BITTNER
    (num_poles, poles_latlon) = poissonDiskSampling(poles_latlon_initial, lat_min, lat_max, lon_min, lon_max, density_input, num_iterations)
    
    return (num_poles, poles_latlon)
    


def place_prediction(latlim, lonlim, lat_step, lon_step):
    '''
    this function places prediction points on the ionosphere given input latitude and longitude limits
    it is recommended to place these within a region fully bounded by the input observations
    
    currently, this function computes a meshgrid and outputs a 2D list
    
    INPUTS:
        latlim - latitude limits [lower, upper]
        lonlim - longitude limits [lower, upper]
        lat_step - the spacing in degrees of the meshgrid in the latitude direction
        lon_step - the spacing in degrees of the meshgrid in the longitude direction
    
    OUTPUTS:
        pred_latlon - a 2D array containing all the lat/lon points. this list is 2D because the lat and lon meshgrids are reshaped to a 1D column vector and placed pair-wise
        dims: [number of prediction points x 2 (lat, lon in degrees)]
    '''
    
    # eftract inputs
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



def predict_with_SECS(radar_velocity_frame_i, velocity_latlon, pred_latlon, poles_latlon, epsilon=0.05):
    '''
    this function serves as the starter function to run the SECS algorithm to predict the plasma flow at various points in the ionosphere
    
    INPUTS:
        radar_velocity_frame_i - the measured velocity in velocity lat/lon NED frame which serve as inputs.
            dims: [number of velocity measurements x 3 (x, y, z)]
            
        velocity_latlon - the lat/lon positions of the corresponding measured velocities
            dims: [number of velocity measurements x 3 (x, y, z)]
            
        pred_latlon - prediction lat/lon positions
            dims: [number of prediction positions x 2 (lat, lon in degrees)]
            
        poles_latlon - pole lat/lon positions
            dims: [number of pole positions x 2 (lat, lon in degrees)]
        
        epsilon - the eigenspace truncation parameter. zero defines NO truncation, and 1 defines 100% truncation. a value of 0.05 is a good amount of truncation.
            dims: a single number. this number should not need to be changed much, if at all.
            
        OUTPUTS:
            pred_vel_frame_fr - spherical elementary current system predicted velocity at the prediction locations.
                this velocity is in the coordinate frame of the prediction location (N-E-D)
            dims: [number of output velocity x 3 (x, y, z)]
    '''
    # check that there are inputs
    if np.size(radar_velocity_frame_i) == 0:
        print("No radar velocity data, skipping SECS algorithm...")
        return 0
    
    # check that the poles do not coincide with a velocity measurement or a prediciton location
    overlap_tolerance = 0.05 # degrees
    bool_overlap1 = compute_num_closeto(poles_latlon, velocity_latlon, angular_tolerance=overlap_tolerance).astype(bool) # check velocity locations
    bool_overlap2 =  compute_num_closeto(poles_latlon, pred_latlon, angular_tolerance=overlap_tolerance).astype(bool) # check prediction locations
    bool_overlap = np.logical_or(bool_overlap1, bool_overlap2)

    # delete the poles that are too close to either the velociy OR prediction locations
    poles_latlon = poles_latlon[np.logical_not(bool_overlap), :]

    # run SECS!
    pred_vel_frame_pr = run_secs(radar_velocity_frame_i, velocity_latlon, pred_latlon, poles_latlon, epsilon)
    
    # if there was an error in the solving of the SVD, return a zero
    if np.all(pred_vel_frame_pr == 0):
        return 0
    
    # reduce last dimension of length 1
    pred_vel_frame_pr = pred_vel_frame_pr.squeeze()
    

    return pred_vel_frame_pr



def compute_num_closeto(interest_latlon, closeto_latlon, angular_tolerance=1):
    '''
    this function computes if the positions in pred_latlon are within a tolerance (in degrees) to a list of points given in closeto_latlon.
    returns a matrif that identifies TRUE (is close) or FALSE (is not close) corresponding to the entries in pred_latlon
    TRUE - entry is positive integer. the specific number identifies the number of CLOSE_TO that is within angular_tolerance
    FALSE - entry is zero
    
    INPUTS:
        interest_latlon - the 2D list of lat/lon positions that we are interested in checking if they are close to ""something""
            dims: [number of positions of interest x 2 (lat, lon in degrees)]
            
        closeto_latlon - the 2D list of lat/lon positions that we define as the ""something"" which we are checking the interest_latlon with respect to
            dims: [number of positions of interest x 2 (lat, lon in degrees)]
        
        angular_tolerance - the tolerance in degrees that defines the maximum separation before flagging as FALSE that an interest_latlon point can be from ANY closeto_latlon (i.e., the closest closeto_latlon)
            dims: a single number
            
    OUTPUTS:
        num_isclose - a list of integers, the bool of which identifies TRUE/FALSE.
            TRUE - interest_latlon is within tolerance to at least one closeto_latlon point.
            xALSE - the selected interest_latlon point is not within tolerance to ANY closeto_latlon
            dims: [number of interest lat/lon locations f nothing]
    '''
    # find SECS pred outputs that are within a tolerance to a velocity measurement
    
    closeto_rotate = np.transpose(closeto_latlon)
    phi1 = interest_latlon[:, [0]] * np.pi/180
    phi2 = closeto_rotate[[0], :] * np.pi/180
    delta1 = interest_latlon[:, [1]] * np.pi/180
    delta2 = closeto_rotate[[1], :] * np.pi/180

    angular_separation = np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(delta1 - delta2)
    num_isclose = np.sum(angular_separation > np.cos(angular_tolerance * np.pi/180), 1)
    return num_isclose



def compute_close_and_far(velocity_vectors, vel_latlon, closeto_latlon, angular_tolerance=1):
    '''
    this function computes if the positions in pred_latlon are within a tolerance (in degrees) to a list of points given in closeto_latlon.
    the velocity vectors correspond to pred_latlon
    this function calls compute_closeto, yet it has additional functionality
    returns the list of velocity_vectors that are close, and the list of velocity_vectors that are far
    
    INPUTS:
        velocity_vectors - the list of velocity vectors that we which to compute if they are close or far from a list of selected points
            dims: [number of velocity vectors x 3 (x, y, z)]
        
        vel_latlon - the list of lat/lon positions that correspond to the input velocity_vectors
            dims: [number of velocity vectors x 2 (lat, lon in degrees)]
            
        closeto_latlon - the list of lat/lon positions that we are comparing if a different lat/lon point is close to
            dims: [number of desired comparison lat/lon points x 2 (lat, lon in degrees)]
        
        angular_tolerance - optional parameter to define the mafimum angular separation to define the velocity_vector being close or far from any closeto_latlon points
        
    OUTPUTS:
        velocity_vector_close - the set of vectors which are within the angular tolerance of ANY closeto_latlon
            dims: [number of close velocity vectors x 3 (x, y, z)]
            
        velocity_vector_far - the set of vectors which are NOT within the angular tolerance of ANY closeto_latlon
            dims: [number of far velocity vectors x 3 (x, y, z)]
        
        vel_latlon_close - the list of lat/lon positions which correspond to velocity_vector_close
            dims: [number of close velocity vectors x 3 (x, y, z)]
        
        vel_latlon_far - the list of lat/lon positions which corerspond to velocity_vector_far
            dims: [number of far velocity vectors x 3 (x, y, z)]
    '''
    # compute the boolean if a prediction velocity_vector location is close to a input velocity 
    bool_isclose = compute_num_closeto(vel_latlon, closeto_latlon, angular_tolerance).astype(bool)
    # initialize
    velocity_vector_close = np.zeros((np.count_nonzero(bool_isclose), 3))
    velocity_vector_far = np.zeros(((np.count_nonzero(np.logical_not(bool_isclose)), 3)))
    vel_latlon_close = np.zeros((np.count_nonzero(bool_isclose), 2))
    vel_latlon_far = np.zeros(((np.count_nonzero(np.logical_not(bool_isclose)), 2)))

    # set close output vectors
    velocity_vector_close[:, 0] = velocity_vectors[bool_isclose, 0]
    velocity_vector_close[:, 1] = velocity_vectors[bool_isclose, 1]
    velocity_vector_close[:, 2] = velocity_vectors[bool_isclose, 2]
    vel_latlon_close[:, 0] = vel_latlon[bool_isclose, 0]
    vel_latlon_close[:, 1] = vel_latlon[bool_isclose, 1]

    # set the far output vectors
    velocity_vector_far[:, 0] = velocity_vectors[np.logical_not(bool_isclose), 0]
    velocity_vector_far[:, 1] = velocity_vectors[np.logical_not(bool_isclose), 1]
    velocity_vector_far[:, 2] = velocity_vectors[np.logical_not(bool_isclose), 2]
    vel_latlon_far[:, 0] = vel_latlon[np.logical_not(bool_isclose), 0]
    vel_latlon_far[:, 1] = vel_latlon[np.logical_not(bool_isclose), 1]
    
    return (velocity_vector_close, velocity_vector_far, vel_latlon_close, vel_latlon_far)


def return_interior_input_velocities(velocity, velocity_latlon, latlim, lonlim):
    '''
    this function returns only the velocities that are located within the latitude and longitude limits
    note this function will not work properly if the north or south pole is involved, although if these poles are within the domain of interest,
    there is likely not a reason to cut the velocities
    
    INPUTS:
        velocity- the list of velocity vectors that we which to compute if they are close or far from a list of selected points
            dims: [number of velocity vectors x 3 (x, y, z)]
           
        vel_latlon - the list of lat/lon positions that correspond to the input velocity_vectors
            dims: [number of velocity vectors x 2 (lat, lon in degrees)]
               
        latlim - latitude limits, lower and upper
            dims: [(lower, upper) x 1]
            
        lonlim - longitude limits, lower and upper
            dims: [(lower, upper) x 1]
            
    OUTPUTS:
        velocity_cut - the list of velocity vectors that are located within the lat/lon boundaries
            dims: [number of velocity vectors x 3 (x, y, z)]
            
        velocity_latlon_cut - the corresponding list of lat/lon coordinates that define the locations of the velocity_cut vectors
            dims: [number of velocity vectors x 2 (lat, lon in degrees)]
    '''
    
    # boolean vector, TRUE means within the domain, FALSE is outside it
    bool_keep = (velocity_latlon[:, 0] >= latlim[0]) & (velocity_latlon[:, 0] <= latlim[1]) & (velocity_latlon[:, 1] >= lonlim[0]) \
                    & (velocity_latlon[:, 1] <= lonlim[1])
    
    # eftract the velocity vectors and velocity vector locations that we desire
    velocity_cut = velocity[bool_keep, :]
    velocity_latlon_cut = velocity_latlon[bool_keep, :]
    
    return (velocity_cut, velocity_latlon_cut)