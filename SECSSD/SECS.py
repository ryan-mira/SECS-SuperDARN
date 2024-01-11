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
import glob
from typing import Union
from .perform_SECS import run_secs
from .bridsonVariableRadius import poissonDiskSampling

def eu_distance_angle(coord1, coord2):
    lat1, lon1 = coord1[0], coord1[1]
    if len(coord2.shape) == 2:
        lat2, lon2 = coord2[:,0], coord2[:,1]
    else:
        lat2, lon2 = coord2[0], coord2[1]
    x = ((lon2 - lon1) * np.cos(0.5 * (np.radians(lat2) + np.radians(lat1))))
    y = (lat2 - lat1)
    distance = np.sqrt(x**2 + y**2)
    return distance

def eu_distance_km(coord1, coord2):
    lat1, lon1 = coord1[0], coord1[1]
    if len(coord2.shape) == 2:
        lat2, lon2 = coord2[:,0], coord2[:,1]
    else:
        lat2, lon2 = coord2[0], coord2[1]
    x = ((lon2 - lon1) * np.cos(0.5 * (np.radians(lat2) + np.radians(lat1)))) * 111.3
    y = (lat2 - lat1) * 111.3
    distance = np.sqrt(x**2 + y**2)
    return distance

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

def read_superDARN(directory:str, 
                   start_time:datetime, end_time:datetime = None, 
                   datatype:str ='v3_grid', 
                   time_step:int = 2):
    
    if end_time is None:
        # if the user requests a start date, but provides no end date, then return only one date
        end_time = start_time + timedelta(minutes=time_step)
    if datatype.lower() == 'v3_grid':
        pattern = '*v3.0.grid.nc'
    elif datatype == '2.5':
        pattern = '*v2.5.nc'
    elif datatype == '3.0':
        pattern = '*v3.0.*.nc'
    else:
        raise ("Wrong datatype, choose between 'v3_grid', '2.5', and '3.0'")
    # Filter the available files by the dates
    start_date = datetime(start_time.year, start_time.month, start_time.day)
    end_date = datetime(start_time.year, end_time.month, end_time.day)
    #Read all files in directory
    files = np.array(sorted(glob.glob(directory + pattern)))
    #Get the corresponding dates for each file
    file_dates = np.array([datetime.strptime(os.path.split(f)[1][:8], "%Y%m%d") for f in files])
    #What are good files?
    idd = (file_dates >= start_date) & (file_dates <= end_date)
    #Filte the file names
    good_files = files[idd]
    
    # create a list of all times to provide velocity outputs
    times0 = np.arange(start_time, end_time, timedelta(minutes=time_step)).astype('datetime64[s]').astype(datetime)
    # times1 = np.arange(start_time+timedelta(minutes=time_step), end_time+timedelta(minutes=time_step), timedelta(minutes=time_step)).astype('datetime64[s]').astype(datetime)
    
    D = xr.Dataset()
    for ir, f in enumerate(good_files):
        print("Reading file " + f)
        # t1 = time.time()
        # open the file
        X = xr.load_dataset(f)
        if datatype.lower() == "v3_grid":
            # handle the dates by adding an offset so it can be converted to Python's datetime format
            obstimes0 = round2minute(mjd2datetime(X["mjd_start"].values))
            idt = np.logical_and(obstimes0 >= start_time, obstimes0 <= end_time)
            # Resample radar observation times to a common time-series defined with times0
            obstimes0_rounded_to_times0 = np.array([times0[abs(times0-t).argmin()] for t in obstimes0[idt]])
            obstimes1_rounded_to_times1 = obstimes0_rounded_to_times0 + timedelta(minutes=time_step)
            
            lat_select = X["vector.glat"][idt].values # deg
            lon_select = X["vector.glon"][idt].values # deg
            azimuth_select = X["vector.g_kvect"][idt].values # azimuth angle N -> E [deg]
            velocity_magnitude_select = X["vector.vel.median"][idt].values # m/s
            velocity_direction_select = X["vector.vel.dirn"][idt].values # +1=towards radar
            velocity_east = velocity_magnitude_select * velocity_direction_select * np.sin(np.radians(azimuth_select))
            velocity_north = velocity_magnitude_select * velocity_direction_select * np.cos(np.radians(azimuth_select))
            velocity_std_select = X["vector.vel.sd"][idt].values 
            
        elif datatype.lower() in ('2.5', '3.0'):
            
            # handle the dates by adding an offset so it can be converted to Python's datetime format
            obstimes0 = round2minute(mjd2datetime(X["mjd"].values))
            idt = np.logical_and(obstimes0 >= start_time, obstimes0 <= end_time)
            iground = np.logical_not(X['gflg'].values.astype(bool))
            idx = np.logical_and(idt, iground)
            # Resample radar observation times to a common time-series defined with times0
            obstimes0_rounded_to_times0 = np.array([times0[abs(times0-t).argmin()] for t in obstimes0[idx]])
            obstimes1_rounded_to_times1 = obstimes0_rounded_to_times0 + timedelta(minutes=time_step)
            
            beam_directions = X.attrs["brng_at_15deg_el"]
            
            beam_select = X['beam'][idx].astype(int)
            azimuth_select = np.array([beam_directions[b] for b in beam_select])
            velocity_direction_select = np.ones(obstimes0[idx].size)
            lat_select = X["lat"][idx].values # deg
            lon_select = X["lon"][idx].values # deg
            velocity_magnitude_select = abs(X["v"][idx].values) # m/s
            away = (X["v"][idx].values > 0)
            velocity_direction_select[away] = -1
            velocity_east = velocity_magnitude_select * velocity_direction_select * np.sin(np.radians(azimuth_select))
            velocity_north = velocity_magnitude_select * velocity_direction_select * np.cos(np.radians(azimuth_select))
            velocity_std_select = X["v_e"][idx].values # 
            del iground, beam_directions, beam_select
        else:
            raise ("")
        
        if ir == 0:
            D['times_start'] = obstimes0_rounded_to_times0
            D['times_end'] = obstimes1_rounded_to_times1
            D['lat'] = lat_select
            D['lon'] = lon_select
            D['velocity_azimuth'] = azimuth_select
            D['velocity_magnitude'] = velocity_magnitude_select
            D['velocity_direction'] = velocity_direction_select
            D['velocity_east'] = velocity_east
            D['velocity_north'] = velocity_north
            D['velocity_std'] = velocity_std_select
            D['radar_lat'] = np.tile(X.attrs["lat"], lat_select.size)
            D['radar_lon'] = np.tile(X.attrs["lon"], lon_select.size)
        
        else:
            D['times_start'] = np.hstack((D['times_start'].values.astype('datetime64[s]').astype(datetime), obstimes0_rounded_to_times0))
            D['times_end'] = np.hstack((D['times_end'].values.astype('datetime64[s]').astype(datetime), obstimes1_rounded_to_times1))
            D['lat'] = np.hstack((D['lat'], lat_select))
            D['lon'] = np.hstack((D['lon'], lon_select))
            D['velocity_azimuth'] = np.hstack((D['velocity_azimuth'], azimuth_select))
            D['velocity_magnitude'] = np.hstack((D['velocity_magnitude'], velocity_magnitude_select))
            D['velocity_direction'] = np.hstack((D['velocity_direction'], velocity_direction_select))
            D['velocity_east'] = np.hstack((D['velocity_east'], velocity_east))
            D['velocity_north'] = np.hstack((D['velocity_north'], velocity_north))
            D['velocity_std'] = np.hstack((D['velocity_std'], velocity_std_select))
            D['radar_lat'] = np.hstack((D['radar_lat'], np.tile(X.attrs["lat"], lat_select.size)))
            D['radar_lon'] = np.hstack((D['radar_lon'], np.tile(X.attrs["lon"], lon_select.size)))    
        X.close()
        
        del lat_select, lon_select, obstimes0, obstimes0_rounded_to_times0, obstimes1_rounded_to_times1, azimuth_select, velocity_magnitude_select, velocity_direction_select, velocity_east, velocity_north, velocity_std_select
    
    return D

def discretize(latlim:Union[list, np.ndarray], lonlim:Union[list, np.ndarray], 
                dlat:Union[float, int], dlon:Union[float, int], 
                velocity_latlon:np.ndarray = None, 
                density_function:str = 'gauss', 
                density_max:Union[float,int] = 10,
                density_min:Union[float,int] = 1,
                debugging:bool = False):
 
    t1 = datetime.now()
    xgrid, ygrid = np.meshgrid(np.arange(lonlim[0], lonlim[1]+.1, dlon), 
                               np.arange(latlim[0], latlim[1]+.1, dlat))
    if velocity_latlon is not None:
        velocity_latlon = np.vstack((velocity_latlon, [90,10]))
        rad = np.ones(xgrid.shape)
        for i in range(xgrid.shape[0]):
            for j in range(xgrid.shape[1]):
                rad[i,j] = np.nanmin(np.array([eu_distance_angle((ygrid[i,j], xgrid[i,j]), velocity_latlon[k,:]) for k in range(velocity_latlon.shape[0])]))
        print (datetime.now()-t1)
        #rad[rad>=1000] = 1000 km
        if density_function == None:
            pass
        elif density_function == 'gauss':
            
            sigma = density_min * 2 #200 km
            A = 10 * (density_max - density_min) / 2 #4.5e1 #1e3 km
            mu = 0
            rad = density_max - (A / np.sqrt(2*np.pi*sigma**2) * np.e**-((rad-mu)/(2*sigma))**2)
            #rad =  1 / (A / np.sqrt(2*np.pi*sigma**2) * np.e**-((rad-mu)/(2*sigma))**2) + 1
        
        else:
            pass
        # TODO Implement more scaling functions, and convert them to KM instead of degrees
       
        samples_latlon = poissonDiskSampling(xgrid, ygrid, latlim[0], latlim[1], lonlim[0], lonlim[1], radius=rad, k=30)
    else:
        samples_latlon = np.hstack((ygrid.reshape([-1,1]), xgrid.reshape([-1,1])))
        rad = np.nan * np.ones(samples_latlon.shape)
        
    if debugging:
        return xgrid, ygrid, rad, samples_latlon
    
    else:
        return samples_latlon

def predict_with_SECS(radar_los:np.ndarray, 
                      velocity_latlon:np.ndarray, 
                      pred_latlon:np.ndarray, 
                      poles_latlon:np.ndarray, 
                      epsilon:Union[float,int] = 0.05):
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
    if np.size(radar_los) == 0:
        print("No radar velocity data, skipping SECS algorithm...")
        return 0
    
    # check that the poles do not coincide with a velocity measurement or a prediciton location
    overlap_tolerance = 0.05 # degrees
    # WE don't really need to remove prediction_points colocated with radar_los????
    bool_overlap = compute_num_closeto(poles_latlon, velocity_latlon, angular_tolerance=overlap_tolerance).astype(bool) # check velocity locations
    # delete the poles that are too close to either the velociy OR prediction locations
    poles_latlon = poles_latlon[~bool_overlap, :]
    
    # run SECS!
    prediction_velocity = run_secs(radar_los, velocity_latlon, pred_latlon, poles_latlon, epsilon)
    # if there was an error in the solving of the SVD, return a zero
    if np.all(prediction_velocity == 0):
        return 0
    else:
        prediction_velocity = prediction_velocity.squeeze()
    
    return prediction_velocity



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

def velocity_isclose(secs_latlon, velocity_latlon, tolerance=1, units:str = 'angle'):
    assert units in ("angle", "km")
    
    idx = np.zeros(secs_latlon.shape[0], dtype=bool)
    if units == "angle":
        #Get an array with the closest observation to each SECS_VEL point
        distance = np.array([np.nanmin(eu_distance_angle(secs_latlon[i], velocity_latlon)) for i in range(secs_latlon.shape[0])])
    else:
        #Get an array with the closest observation to each SECS_VEL point
        distance = np.array([np.nanmin(eu_distance_km(secs_latlon[i], velocity_latlon)) for i in range(secs_latlon.shape[0])])
    idx[distance<=tolerance] = True
    
    return idx

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