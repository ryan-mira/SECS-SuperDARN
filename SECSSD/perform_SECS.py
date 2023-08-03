# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:13:42 2023

@author: ryanj
"""
import numpy as np


def R_from_inertial_to_coord(lat, lon):
    '''
    This function computes the DCM rotation matrix from the inertial frame to a body frame, located at a specific
    lat/lon point. The body frame is assumed to be (north - east - down).
    lat/lon are in radians

    this function is internal to perform_SECS.py
    '''
    # define the R2 rotation matrix (about 2 axis)
    R2 = lambda angle : np.array([ (np.cos(angle), 0, -np.sin(angle)),
                                 (0, 1, 0),
                                 (np.sin(angle), 0, np.cos(angle))
        ])
    # define the R3 rotation matrix (about 3 axis)
    R3 = lambda angle : np.array([ (np.cos(angle), np.sin(angle), 0),
                                  (-np.sin(angle), np.cos(angle), 0),
                                  (0, 0, 1)
        ])
    
    # compute the DCM from inertial to body, [Body <-- iNertial]
    BN = R2(-90 * np.pi/180) @ R2(-lat) @ R3(lon)
    return BN


def cartesian_from_latlon(lat, lon):
    '''
    function takes in lat, lon, and a radius (assummed to be one to ensure answers are in unit vectors) and computes the
    cartesian coordinates with this, expressed in inertial coordinates.
    (lat) x (lon) are in radians

    this function is internal to perform_SECS.py
    '''
    # get inertial components
    x = np.cos(lon) * np.sin(np.pi/2 - lat)
    y = np.sin(lon) * np.sin(np.pi/2 - lat)
    z = np.cos(np.pi/2 - lat)
    r_frame_N = np.hstack((x, y, z))
    
    return r_frame_N


def compute_theta_star(r1_N, r2_N):
    '''
    computes the angular separation between two unit vectors, r1 and r2 (given in inertial coordinates)
    r1 and r2 MUST be unit vectors. keeping them unit improves computation time
    VECTORIZED
    
    this function is internal to perform_SECS.py
    '''
    r2 = r2_N.transpose()
    dot_product = np.dot(r1_N, r2)
    
    # compute the latitude of vector 2 assuming vector 1 defines the north pole
    # mathematically, this is equivalent to the angular separation between r1 and r2
    theta_star = np.arccos(dot_product)
    
    return (theta_star, dot_product)


def compute_ehat_phi_frame_N(r1_N, r_p_N, dot_product):
    '''
    computes the ehat_phi unit vector. this is the unit vector in the PHI direction (LONGITUDE...not latitude) of a sphere assuming the +z pole is located
    at r_p_N (given in inertial coordinates, with inertial frame defined using the earth, NOT the pole sphere)
    r1_N is the location to compute ehat_phi at.
    ehat_phi is computed and returned in inertial coordinates, relative to the earth, NOT the pole sphere.
    mind those coordinate frames...
    VECTORIZED
    
    this function is internal to perform_SECS.py
    '''
    
    size_pos = np.size(r1_N, 0)
    size_p = np.size(r_p_N, 0)

    # prepare the arrays for vectorized division
    r1_N_reshaped = r1_N.reshape(-1, 1, 3)
    r1_prepared = np.tile(r1_N_reshaped, [1, size_p, 1]);
    r_p_N_reshaped = r_p_N.reshape(1, -1, 3)
    r_p_prepared = np.tile(r_p_N_reshaped, [size_pos, 1, 1])
    
    # cross the vectors to find a right-handed direction that is perpendicular to both vectors
    cross_result = np.cross(r1_prepared, r_p_prepared)
    
    # divide by is the magnitude. this comes from the dot product instead of the cross product vectors themselves.
    # mathematically, it is identical, but using the dot product saves computation time because it has already been carried out
    divide_by = np.sqrt(1 - dot_product**2).reshape([size_pos, size_p, -1])

    # compute the ehat vector of SECS pole
    ehat_phi_frame_N = cross_result / divide_by
    return ehat_phi_frame_N

def run_secs(radar_velocity_radarframe, velocity_latlon, radar_latlon, radar_index, pred_latlon, poles_latlon, epsilon):
    '''
    this is the main function to call from any script to run the spherical elementary current system method.
    the SECS method places many divergence free poles (in this implementation), and it attempts to find a least-effort solution to the overdetermined or underdetermined
    problem of fitting scaling factors to these poles to reproduce a set of input vectors. it accomplishes this solution by utilizing a truncated
    singular value decomposition, with the cutoff parameter defined by epsilon, and using this to solve for the scaling factors. once the scaling factors are computed,
    the reconstructed divergence free velocity field can be computed and outputted.
    
    this is a mathematical model. there is no physics, nor is there anything "smart" about it. it is just math, eigenvalues, and fancy linear algebra. therefore, care
    must be taken to ensure that the SECS model does not invent false, spurrious details in the reconstructed vectorfield. the SECS model will solve just about any
    linear system you give it, so make sure to give it a good one.
    
    if, for example, you give it input velocities containing ground scatter, it WILL fit large flow velocities next to nearly stationaly "flow (in actuality, ground scatter)",
    and the resulting reconstructed vectorfield will look terrible. there will be large gradients in flow velocity over short distances because that is what
    the model was told to fit. there is no ability for this model, as it is coded here, to filter out "bad" input observations, nor does it weigh any inputs.
    
    INPUTS:
        radar_velocity_radarframe: These are the velocities that the radars see, in the frame of the radar (North - East - Down).
            Size: (num of radar velocities) x (components: 3)
            
        velocity_latlonR: Coordinates of the velocity points that the radars are sampling
            Size: (num of radar velocities) x (coord, 2: lat, lon)
        
        radar_latlonR: radar locations
            Size: (num of radars) x (coord, 2: lat, lon)
        
        radar_index: the indicies that correspond to radar_velocity_radarframe to tell which radar. Since radar_velocity_radarframe is relative
        to the radar frame, the specific radar must be known to know the coordinate frame
            Size: (num of radar velocities) x (1)
            
        pred_latlonR: prediction locations
            Size: (num of prediction locations) x (coord, 2: lat, lon)
        
        poles_latlonR: SECS pole locatoins
            Size: (num of SECS poles) x (coord, 2: lat, lon)
            
    OUTPUTS:
        pred_data_frame_pr - the list of velocity vectors at the prediction locations that are computed from the SECS method
            dims: [number of prediction velocity x 3 (x, y, z)]
    '''

    # convert all coordinates to radians
    velocity_latlon = velocity_latlon * np.pi/180
    radar_latlon = radar_latlon * np.pi/180
    pred_latlon = pred_latlon * np.pi/180
    poles_latlon = poles_latlon * np.pi/180
    
    # radius_I is radius of the velocity_latlon points. this is assumed to be CONSTANT (change??)
    R_earth = 6371e3 # radius of earth in meter
    radius_I = R_earth
    
    # pre-compute the cartesian vectors for all the lat/lon positions
    r_i_frame_N = cartesian_from_latlon(velocity_latlon[:, [0]], velocity_latlon[:, [1]])
    r_p_frame_N = cartesian_from_latlon(poles_latlon[:, [0]], poles_latlon[:, [1]])
    r_pr_frame_N = cartesian_from_latlon(pred_latlon[:, [0]], pred_latlon[:, [1]])
    
    # size_i is the number of velocity points
    # size_j is the number of SECS poles
    size_i = np.size(radar_velocity_radarframe, 0)
    size_j = np.size(poles_latlon, 0)
    size_k = np.size(radar_latlon, 0)
    
    # initialize the transfer matrix, T
    T = np.zeros((size_i, size_j))
    
    # comute theta_star, which is the latitude of the velocity point relative to a SECS pole.
    # this is geometrically equivalent to the angular separation between a vector pointing to the velocity locatoin and the SECS pole
    # order matters because of the order of reshaping that is done
    (theta_star, dot_product) = compute_theta_star(r_i_frame_N, r_p_frame_N)
    
    # compute the ehat_phi (longitude) vector in inertial frame of a SECS sphere with +z pole located at SECS_latlon at
    # location specified by r1_N, also in inertial coordinates
    ehat_phi_frame_N = compute_ehat_phi_frame_N(r_i_frame_N, r_p_frame_N, dot_product)

    # determine what velocity points line up with which radars. use radar_index to do this.
    index_change = np.where(np.not_equal(np.diff(radar_index, axis=0), 0))[0]+1 # add one?? YES! CONFIRMED
    index_change = np.insert(index_change, 0, 0)
    index_change = np.append(index_change, size_i)
    
    # compute [NR], which is the DCM that transforms from the radar frame to inertial frame
    # [iNertial <-- Radar]
    NR = np.zeros([size_i, 3, 3])
    counter = 0
    for k in range(size_k):
        # if the radar is not contained in radar_index
        if not(np.any(k == radar_index)):
            continue
        
        # set the DCM for radar to inertial
        NR[index_change[counter]:index_change[counter+1], :, :] = R_from_inertial_to_coord(radar_latlon[k, 0], radar_latlon[k, 1]).transpose()
        counter = counter + 1
    
    # compute the unit vector of the radar-measured velocity, in inertial coordinates.
    # care must be taken to ensure the correct frame. since the radar measures velocity
    # in its own frame (at radar_latlon), the frame of velocity is actually in the radar frame
    radar_velocity_magnitude = np.sqrt((radar_velocity_radarframe * radar_velocity_radarframe).sum(axis=1)).reshape([-1, 1])
    ehat_vel_frame_R = radar_velocity_radarframe / radar_velocity_magnitude
    
    # reshape the array to enable vectorized matrix multiplication with a stack of DCMs, [NR]
    ehat_vel_frame_R = ehat_vel_frame_R.reshape(-1, 3, 1)
    
    # rotate velocity vectors from radar to inertial frame
    ehat_vel_frame_N = NR @ ehat_vel_frame_R
    
    # change shape and multiply to fill out the appropriate spaces
    ehat_vel_frame_N = np.tile(ehat_vel_frame_N.reshape([size_i, 1, 3]), [1, size_j, 1])

    # compute the dot product
    prod1 = 1 / (4 * np.pi * radius_I * np.tan(theta_star / 2))
    prod1 = prod1.reshape([size_i, size_j, 1])
    to_dot1_frame_N = prod1 * ehat_phi_frame_N
    
    # compute the transfer matrix
    T = np.einsum('ijk, ijk->ij', to_dot1_frame_N, ehat_vel_frame_N)

    # get Z matrix, which is the list of velocity magnitudes (speeds) observed by all the radars
    # could simplify by not creating a duplicate variable
    Z = radar_velocity_magnitude
    
    # compute the singular value decomposition of the transfer matrix, T. This breaks the matrix up into a list of eigenvalues (S) and other stuff (U, V')
    (U, S_vec, V_t) = np.linalg.svd(T,full_matrices=False)

    # define the cutoff point to separate the poorly conditioned from the well conditioned part of the transfer matrix
    S_cutoff = epsilon * np.max(S_vec)
    
    #svd = scipy.linalg.lapack.dgejsv(T, joba=1)

    # logical index to set the poorly conditioned (sqrt) eigenvalues to zero
    S_vec[S_vec < S_cutoff] = 0
    S_vec[S_vec!=0] = 1 / S_vec[S_vec!=0]
    
    S = np.diag(S_vec)
    
    # compute the scaling factors using fancy matrix math
    I = V_t.transpose() @ S @ U.transpose() @ Z
    
    
    '''
    PREDICTION PART
    This is the part that computes the predicted velocity at any desired location
    '''

    # number of prediction points
    size_p = np.size(pred_latlon, 0)
    
    # use a for loop to compute the DCMs. this is slow and should be optimized
    # DCM from inertial frame to prediction frame [PRedict <-- iNetial]
    PRN = np.zeros([size_p, 3, 3])
    for p in range(size_p):
        PRN[p, :, :] = R_from_inertial_to_coord(pred_latlon[p, 0], pred_latlon[p, 1])
    
    # compute theta star, same angle as above but with different points (pred crossed with poles)
    (theta_star, dot_product) = compute_theta_star(r_pr_frame_N, r_p_frame_N)
    
    # compute ehat_phi_frame_N, same unit vector as above but with different points (pred crossed with poles)
    # PHI is in direction of LONGITUDE (of the SECS pole, NOT of globe, yet this unit vector is expressed in inertial coordinates)
    ehat_phi_frame_N = compute_ehat_phi_frame_N(r_pr_frame_N, r_p_frame_N, dot_product)
    
    # compute the intermediate sum
    intermediate_scalar = I.transpose() * 1 / (4 * np.pi * radius_I * np.tan(theta_star / 2))
    intermediate = intermediate_scalar[:, :, np.newaxis] * ehat_phi_frame_N
    
    # compute a summation of all the scaling factors multiplied by respective basis functions
    sum_V = np.sum(intermediate, axis=1)
    
    # matrix multiplication to get into prediction data frame
    pred_data_frame_pr = PRN @ sum_V[:, :, np.newaxis]
    
    return pred_data_frame_pr
