from datetime import datetime, timezone
import csv
import subprocess
import os

import matplotlib.pyplot as plt

import SECSSD as SD
import numpy as np
import pymap3d

from density import Density
import pnpwrapper


def test_density(velocity, velocity_latlon, radar_latlon, radar_index, time, prediction_lonlim, prediction_latlim, poles_lonlim, poles_latlim, density_poles: Density, density_predictions: Density, outdir):
    
    filename_base = str(time) + "_" + density_poles.description()
    scale = 30000

    # Filter velocities locations that are the same
    # Filter also based on poles_lonlim and poles_latlim?
    velocity_latlon_unique = filter_unique(velocity_latlon)

    fig = plt.figure(figsize=[16, 16])
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(326)
    ax6 = fig.add_subplot(325)

    # 1. Plot measured velocity field

    ax1.quiver(velocity_latlon[:, 1], velocity_latlon[:, 0], velocity[:, 1], velocity[:, 0], color="b", width=0.001, scale=scale)
    ax1.set_xlim(poles_lonlim)
    ax1.set_ylim(poles_latlim)
    ax1.grid(axis='both')
    ax1.set_title("Input velocities, time = %s" % time)

    # 2. Place poles uniformly to determine density plot

    poles_latlon = SD.place_poles(poles_latlim, poles_lonlim, 0.3, 0.5)
    poles_density = np.array(list(map(lambda p: density_poles.density(p, velocity_latlon_unique), poles_latlon)))

    ax2.scatter(poles_latlon[:, 1], poles_latlon[:, 0], s=1, c=poles_density, cmap="jet")
    ax2.scatter(velocity_latlon[:, 1], velocity_latlon[:, 0], marker='.', c="blue", s=1) 
    ax2.set_xlim(poles_lonlim)
    ax2.set_ylim(poles_latlim)
    ax2.grid(axis='both')
    ax2.set_title(density_poles.description())

    # 3. Run fill algorithm for poles with the provided density function and plot the result

    inside_pole = lambda pole: poles_latlim[0] <= pole[0] <= poles_latlim[1] and poles_lonlim[0] <= pole[1] <= poles_lonlim[1]
    h_pole = lambda pole: density_poles.h(pole, velocity_latlon_unique)

    poles_latlon = np.ndarray((1, 2))
    poles_latlon[0, 0] = np.average(poles_latlim)
    poles_latlon[0, 1] = np.average(poles_lonlim)
    poles_latlon = np.array(pnpwrapper.fill(inside_pole, h_pole, poles_latlon, 12, 50000))
    print("Number of poles: %d" % len(poles_latlon))
    if len(poles_latlon) >= 50000:
        print("Upper limit on number of poles reached!")

    inside_prediction = lambda p: prediction_latlim[0] <= p[0] <= prediction_latlim[1] and prediction_lonlim[0] <= p[1] <= prediction_lonlim[1]
    h_prediction = lambda p: density_predictions.h(p, velocity_latlon_unique)
    # Some prediction points might be outside of prediction bounds, because we start with velocity_latlon_unique; use an additional filter for this
    prediction_latlon = np.array(list(filter(inside_prediction, pnpwrapper.fill(inside_prediction, h_prediction, velocity_latlon_unique))))
    print("Number of prediction points: %d" % len(prediction_latlon))

    ax3.scatter(velocity_latlon[:, 1], velocity_latlon[:, 0], marker='.', c="blue", s=1) 
    ax3.scatter(poles_latlon[:, 1], poles_latlon[:, 0], marker='.', c="red", s=1)
    ax3.set_xlim(poles_lonlim)
    ax3.set_ylim(poles_latlim)
    ax3.grid(axis='both')
    ax3.set_title("%d velocities (blue) and %d poles (red)" % (len(velocity_latlon), len(poles_latlon)))

    # 4. Run SECS and plot result for prediction locations

    print("Running SECS model")
    t1 = datetime.utcnow()
    prediction_velocity_frame_pr = SD.predict_with_SECS(velocity, velocity_latlon, radar_latlon, radar_index, prediction_latlon, poles_latlon)
    t2 = datetime.utcnow()
    print("Ran SECS Model in " + "{:.2f}".format((t2 - t1).total_seconds()) + " seconds")
    (prediction_velocity_close, prediction_velocity_far, prediction_latlon_close, prediction_latlon_far) = SD.compute_close_and_far(prediction_velocity_frame_pr, prediction_latlon, velocity_latlon)

    ax4.quiver(prediction_latlon_close[:, 1], prediction_latlon_close[:, 0], prediction_velocity_close[:, 1], prediction_velocity_close[:, 0], color="b", width=0.002, scale=scale)
    ax4.quiver(prediction_latlon_far[:, 1], prediction_latlon_far[:, 0], prediction_velocity_far[:, 1], prediction_velocity_far[:, 0], color="k", width=0.001, scale=scale)
    ax4.set_xlim(poles_lonlim)
    ax4.set_ylim(poles_latlim)
    ax4.grid(axis='both')
    ax4.set_title("Predicted velocity field, %d points" % len(prediction_latlon))

    # 5. Convert to geocentric Earth-Centered Earth-Fixed (ECEF) cartesian coordinate system and calculate divergence using Medusa
    # Consider Earth as a sphere of radius 1

    prediction_lat_rad = np.deg2rad(prediction_latlon[:, 0])
    prediction_lon_rad = np.deg2rad(prediction_latlon[:, 1])
    prediction_locations_ecef = np.array(pymap3d.geodetic2ecef(prediction_lat_rad, prediction_lon_rad, 0, ell=pymap3d.Ellipsoid(6371e3, 6371e3), deg=False)).transpose()

    def ned2ecef_matrix(lat, lon):
        return np.array([[-np.sin(lat) * np.cos(lon), -np.sin(lon), -np.cos(lat) * np.cos(lon)],
                         [-np.sin(lat) * np.sin(lon),  np.cos(lon), -np.cos(lat) * np.sin(lon)],
                         [ np.cos(lat),                0,           -np.sin(lat)              ]])

    prediction_velocity_ecef = np.array(list(map(lambda vellatlon: np.matmul(ned2ecef_matrix(vellatlon[1], vellatlon[2]), vellatlon[0]), zip(prediction_velocity_frame_pr, prediction_lat_rad, prediction_lon_rad))))

    #fig2 = plt.figure()
    #ax = fig2.add_subplot(111, projection='3d')
    #ax.quiver(prediction_locations_ecef[:, 0], prediction_locations_ecef[:, 1], prediction_locations_ecef[:, 2], prediction_velocity_ecef[:, 0], prediction_velocity_ecef[:, 1], prediction_velocity_ecef[:, 2], color="b", length=0.01, normalize=True)
    #plt.show()

    file_velocity_field = outdir + filename_base + "_field.csv"
    with open(file_velocity_field, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in zip(prediction_locations_ecef, prediction_velocity_ecef):
            writer.writerow(np.append(row[0], row[1]))

    file_velocity_field_div = outdir + filename_base + "_field_div.csv"
    cmd = ["./adaptive/divergence", file_velocity_field, file_velocity_field_div]
    subprocess.run(cmd, capture_output=True)
    velocity_divergence = np.loadtxt(file_velocity_field_div)

    max_div = np.max(np.abs(velocity_divergence))
    ax5.scatter(prediction_latlon[:, 1], prediction_latlon[:, 0], s=np.abs(velocity_divergence)/max_div, c=np.abs(velocity_divergence), cmap="jet")
    ax5.set_xlim(poles_lonlim)
    ax5.set_ylim(poles_latlim)
    ax5.grid(axis='both')
    ax5.set_title("Divergence")

    # 6. Plot histogram of divergence

    ax6.hist(velocity_divergence, bins=30, log=True)
    ax6.set_title("Histogram of divergence, std = %.3f, max = %.3f" % (np.std(velocity_divergence), np.max(np.abs(velocity_divergence))))

    # 7. Save figure

    fig.savefig(outdir + filename_base + ".png", dpi=200)
    plt.close(fig)

    os.remove(file_velocity_field)
    os.remove(file_velocity_field_div)

    return (prediction_velocity_close, prediction_velocity_far, prediction_latlon_close, prediction_latlon_far, velocity_divergence)


# Returns locations that are at least epsilon apart
def filter_unique(locations_latlon, epsilon = 0.5):
    unique = [locations_latlon[0]]
    for i in range(1, np.size(locations_latlon, 0)):
        location = locations_latlon[i]
        d_min = np.min(np.linalg.norm(unique - location, ord=2, axis=1))
        if d_min > epsilon:
            unique.append(location)
    return np.array(unique)
