"""
Implementation of the fast Poisson Disk Sampling algorithm of 
Bridson (2007) adapted to support spatially varying sampling radii. 

Adrian Bittner, 2021
Published under MIT license. 
"""

import numpy as np


def compute_bool_shouldcompare(interest_latlon, seed_point, angular_tolerance):
    '''
    this function computes if the positions in pred_latlon are within a tolerance (in degrees) to a list of points given in closeto_latlon.
    returns a matrix that identifies TRUE (is close) or FALSE (is not close) corresponding to the entries in pred_latlon
    
    INPUTS:
        interest_latlon - the 2D list of lat/lon positions that we are interested in checking if they are close to ""something""
            dims: [number of positions of interest x 2 (lat, lon in degrees)]
            
        seed_point - the lat/lon of the point that is used to generate the random sammples. used for speeding up computation
            dims: [a single lat/lon point (lat, lon in degrees) x nothing]
        
        angular_tolerance - the tolerance in degrees that defines the maximum separation before flagging as FALSE that an interest_latlon point can be from ANY closeto_latlon (i.e., the closest closeto_latlon)
            dims: a single number
            
    OUTPUTS:
        bool_isclose - TRUE - interest_latlon is within tolerance to at least one closeto_latlon point. FALSE - the selected interest_latlon point is not within tolerance to ANY closeto_latlon
            dims: [number of interest lat/lon locations x nothing]
    '''
    # find SECS pred outputs that are within a tolerance to a velocity measurement
    phi1 = interest_latlon[:, [0]] * np.pi/180 # lat 1
    phi2 = seed_point[[0]] * np.pi/180 # lat 2 # yes, there MUST be two square brackets for some reason
    delta1 = interest_latlon[:, [1]] * np.pi/180 # lon 1
    delta2 = seed_point[[1]] * np.pi/180 # lon 2

    # compute the cosine of the angular separation. avoid taking the inverse cosine to save computation
    angular_separation = np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(delta1 - delta2)

    # TRUE if close
    bool_isclose = np.any(angular_separation > np.cos(angular_tolerance * np.pi/180), 1)
    return bool_isclose

def compute_bool_closeto(interest_latlon, closeto_latlon, seed_point, angular_tolerance):
    '''
    this function computes if the positions in pred_latlon are within a tolerance (in degrees) to a list of points given in closeto_latlon.
    returns a matrix that identifies TRUE (is close) or FALSE (is not close) corresponding to the entries in pred_latlon
    
    INPUTS:
        interest_latlon - the 2D list of lat/lon positions that we are interested in checking if they are close to ""something""
            dims: [number of positions of interest x 2 (lat, lon in degrees)]
            
        closeto_latlon - the 2D list of lat/lon positions that we define as the ""something"" which we are checking the interest_latlon with respect to
            dims: [number of positions of interest x 2 (lat, lon in degrees)]
            
        seed_point - the lat/lon of the point that is used to generate the random sammples. used for speeding up computation
            dims: [a single lat/lon point (lat, lon in degrees) x nothing]
        
        angular_tolerance - the tolerance in degrees that defines the maximum separation before flagging as FALSE that an interest_latlon point can be from ANY closeto_latlon (i.e., the closest closeto_latlon)
            dims: a single number
            
    OUTPUTS:
        bool_isclose - TRUE - interest_latlon is within tolerance to at least one closeto_latlon point. FALSE - the selected interest_latlon point is not within tolerance to ANY closeto_latlon
            dims: [number of interest lat/lon locations x nothing]
    '''
    # find SECS pred outputs that are within a tolerance to a velocity measurement
    # pre-compute which points are DEFINITELY too far away, and not worth comparing to
    # do this using only a single point (the seed), and not the multitude of random samples
    # the angular tolerance is 3x the radius because that is what it must be
    bool_isnear = compute_bool_shouldcompare(closeto_latlon, seed_point, 3*angular_tolerance)
    
    # delete the points that are too far away and not worth considering
    closeto_latlon = closeto_latlon[bool_isnear.squeeze(), :]
    
    closeto_rotate = np.transpose(closeto_latlon)
    phi1 = interest_latlon[:, [0]] * np.pi/180 # lat 1
    phi2 = closeto_rotate[[0], :] * np.pi/180 # lat 2
    delta1 = interest_latlon[:, [1]] * np.pi/180 # lon 1
    delta2 = closeto_rotate[[1], :] * np.pi/180 # lon 2

    # compute the cosine of the angular separation. avoid taking the inverse cosine to save computation
    angular_separation = np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(delta1 - delta2)

    # TRUE if close
    bool_isclose = np.any(angular_separation > np.cos(angular_tolerance * np.pi/180), 1)
    return bool_isclose


def poissonDiskSampling(latlon, lat_min, lat_max, lon_min, lon_max, radius, k=30):
    """
    Implementation of the Poisson Disk Sampling algorithm.

    :param radius: 2d array specifying the minimum sampling radius for each spatial position in the sampling box. The
                   size of the sampling box is given by the size of the radius array.
    :param k: Number of iterations to find a new particle in an annulus between radius r and 2r from a sample particle.
    :param radiusType: Method to determine the distance to newly spawned particles. 'default' follows the algorithm of
                       Bridson (2007) and generates particles uniformly in the annulus between radius r and 2r.
                       'normDist' instead creates new particles at distances drawn from a normal distribution centered
                       around 1.5r with a dispersion of 0.2r.
    """
    
    '''
    this code is written by Adrian Bittner at https://gitlab.com/abittner, and this code is itself based on Fast Poisson Disk Sampling in Arbitrary Dimensions,
    written by Robert Bridson. doi 
    
    ADDITIONS:
        new inputs:
            latlon - a grid of latitude/longitude points in degrees that are used as inputs to start the poisson disk algorithm
            lat_min - minimum latitude in degrees
            lat_max - maximum latitude in degrees
            lon_min - minimum longitude in degrees
            lon_max - maximum longitude in degrees
    '''
    
    # Pick initial (active) point
    np.random.seed(2) # TEMPORARY!!!!
    # linearly interpolate between the latitude and longitude limits, respectively
    coords = np.zeros((2, 1))
    coords[0] = np.random.random() * (lat_max - lat_min) + lat_min # latitude
    coords[1] = np.random.random() * (lon_max - lon_min) + lon_min # longitude
    nParticle = 1

    # Initialise active queue
    queue = np.empty([0, 2])
    queue = np.vstack((queue, coords.transpose())) # Appending to list is much quicker than to numpy array, if you do it very often
    particleCoordinates = np.empty([0, 2])
    particleCoordinates = np.vstack((particleCoordinates, coords.transpose())) # List containing the exact positions of the final particles

    # Continue iteration while there is still points in active list
    while np.size(queue):
        # Pick random element in active queue
        idx = np.random.randint(np.size(queue, 0))
        activeCoords = queue[idx]
        nearest_latlon = np.floor(activeCoords).astype('int')    
        nearest_latlon_index = np.logical_and(nearest_latlon[0] == latlon[:, 0], nearest_latlon[1] == latlon[:, 1])

        # Pick radius for new sample particle ranging between 1 and 2 times the local radius
        radius_pt = radius[nearest_latlon_index]
        newRadius_d = radius_pt * (np.random.random(k) + 1)[:, np.newaxis]
        newRadius = newRadius_d * np.pi/180 # convert to radians
        
        # Pick the angle to the sample particle and determine its coordinates
        angle = 2 * np.pi * np.random.random(k)[:, np.newaxis]
        newCoords = np.zeros((k, 2))

        # find new coordinates utilizing spherical coordinates. the randomly selected radius and angle
        # define the angular separation and angle, repsectively, from the point
        lat1 = activeCoords[0] * np.pi/180
        lon1 = activeCoords[1] * np.pi/180
        
        newCoords[:, [0]] = np.arcsin(np.sin(lat1) * np.cos(newRadius) + np.cos(lat1) * np.sin(newRadius) * np.cos(angle))
        newCoords[:, [1]] = lon1 + np.arctan2(np.sin(angle) * np.sin(newRadius) * np.cos(lat1), np.cos(newRadius) - np.sin(lat1) * np.sin(newCoords[:, [0]]))
        
        # convert back to degrees
        newCoords = newCoords * 180/np.pi
        
        # Prevent that the new particle is outside of the grid
        bool_inclusive = (newCoords[:, [0]] >= lat_min) & (newCoords[:, [0]] <= lat_max) & (newCoords[:, [1]] >= lon_min) & (newCoords[:, [1]] <= lon_max)
        newCoords = newCoords[bool_inclusive.squeeze(), :]
        newRadius_d = newRadius_d[bool_inclusive.squeeze(), :]
        
        # perform the logical check to find if there are any close points to the k-amount of generated points. if so, IGNORE THESE POINTS
        bool_isclose = compute_bool_closeto(newCoords, particleCoordinates, activeCoords, radius_pt)
        check = np.where(np.logical_not(bool_isclose))[0]
        if (np.size(check)):
            # No conflicts detected. Create a new particle at this position!
            index_isclose = check[0]

            queue = np.vstack((queue, newCoords[[index_isclose], :]))
            particleCoordinates = np.vstack((particleCoordinates, newCoords[[index_isclose], :]))
            nParticle += 1

        else:
            # There is a conflict. Do NOT create a new particle at this position!
            queue = np.delete(queue, idx ,0)
            continue

    return(nParticle, particleCoordinates)