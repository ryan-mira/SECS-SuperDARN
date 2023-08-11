# Adaptive pole and prediction locations

There are 3 components in this directory:

1. Standalone fill algorithm in `fill/`, contained in `PNP.hpp`, which uses `nanoflann.hpp` for k-d trees, together with a C++ wrapper for Python using pybind11 in `pnpwrapper.cpp`. The fill algorithm needs a domain function, a spacing function and at least one starting point. In essence, the algorithm takes a point, calculates the spacing at that point and tries placing points on a circle of that radius. This goes on until no more points can be placed. Check the [Filling domain interior section of this wiki link](https://e6.ijs.si/medusa/wiki/index.php/Positioning_of_computational_nodes).
   
2. Python files: `test_density_impl.py` runs the SECS algorithm and handles plotting, `test_density.py` reads input files and gets everyhing else ready for `test_density_impl.py`. There are some predefined density functions in `density.py`. The `h` function is used for the spacing function in the fill algorithm and the `density` function is used for plotting the density.

3. Divergence calculator using [Medusa](https://gitlab.com/e62Lab/medusa/) in `divergence.cpp`.

## Building

To compile the standalone fill algorithm, you need at least Eigen and pybind11. From this directory, run the following command (tested on Linux):

`g++ -I/usr/include/Eigen/ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) -o pnpwrapper$(python3-config --extension-suffix) fill/pnpwrapper.cpp`

This generates a `.so` file in this directory that Python can use as `import pnpwrapper`.

To compile the divergence calculator, you need Medusa source in the `medusa/` directory at the root of this project. Then, configure the CMake project in the root directory using `cmake .` and compile the `divergence.cpp` using `make divergence`. This should place a file named `divergence` in this directory, `adaptive/`.

## Running

After everything is compiled, edit `test_density.py` as needed and run using Python.
