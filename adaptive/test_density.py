import os
from datetime import datetime

import SECSSD as SD
from test_density_impl import test_density

from density import DensityConstant, DensityNormal, DensityLinear, DensityNearest

# Input directory
indir = "superDARN_data_input_directory" + os.sep
start_date = datetime(2015, 3, 1, 4, 0)
end_date = datetime(2015, 3, 1, 4, 10)
(all_data, all_time) = SD.read_superDARN(indir, "v3_grid", start_date, end_date)

# Output directory
outdir = "adaptive" + os.sep + "out" + os.sep
if not os.path.exists(outdir):
    import subprocess
    subprocess.call('mkdir "{}"'.format(outdir), shell=True)

# Limit parameters
prediction_latlim = [45, 75]
prediction_lonlim = [-160, -30]
poles_latlim = [44, 77]
poles_lonlim = [-162, -28]

# List of density functions to test (for the poles)
densities = [
    DensityNearest(a=1.25,b=0,h_min=0.75,h_max=2.5),
    #DensityNormal(sigma=2.4, h_min=0.75, h_max=2.5),
    #DensityNearest(a=1,b=0,h_min=0.75,h_max=2.5),
    #DensityNearest(a=1,b=0,h_min=0.6,h_max=2.5),
    #DensityConstant(rho=1.2),
    #DensityLinear(b=2, a=1, h_min=0.75, h_max=2.5)
]

density_prediction = DensityNormal(sigma=4.0, h_min=0.8, h_max=2.0)

# run a for loop that goes through each time and computes the SECS model for each time
# inside the for loop, it also plots and saves the figure to an output save directory
for density in densities:
    for i, select_time in enumerate(all_time):

        # select the data to input into SECS
        select_velocity = all_data[0][i]
        select_velocity_latlon= all_data[1][i]
        select_radar_latlon = all_data[2][i]
        select_radar_index = all_data[3][i]
        select_time = select_time[0] # get out of list and into datetime format

        test_density(select_velocity, select_velocity_latlon, select_radar_latlon, select_radar_index, select_time, prediction_lonlim, prediction_latlim, poles_lonlim, poles_latlim, density, density_prediction, outdir)
