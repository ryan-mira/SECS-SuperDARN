# SECS-SuperDARN

SECS-SuerDARN is a Python program that computes a divergence-free ion drift vector field based on line-of-sight SuperDARN velocities.  The work is
sponsored by a NASA contract 80GSFC22CA011.

The program currently reads SuperDARN files available from the JHUAPL website https://superdarn.jhuapl.edu/. Best results are obtained from the "V3_grid" 
data format which remove most of the ground clutter.

The SECS coordinate system (Spherical Elementary Current Systems) has been applied to magnetic field measurements; however, the method can be applied to general vectorfields on a spherical shell. The SECS algorithm in this repository is the genearl implementation.
Further, it provides a method for adapting to the density of input measurements, and this gives significantly better numerical stability to the SECS output fit, and it overall improves the output substantially. The variable-spaced poles can be accessed either with a python function (slower) or a cpp file located in 'adaptive.' Both methods are coded independently of each other, yet they achieve the same result. The Python algorithm to place poles is suggested for its ability to work ``out of the box.''

The program is under development as we speak.

Install the program in a development mode as
```
python setup.py develop
```
