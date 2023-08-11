/**
 * PNP algorithm wrapper for pybind11.
 * 
 */

#include <chrono>

#include <Eigen/Core>
#include "PNP.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

using Vector2 = Eigen::Matrix<double, 2, 1>;

/**
 * Fill algorithm wrapper for 2D vectors.
 *
 * @param inside Characteristic function of the domain, returns `true` if and only if given point is inside the domain.
 * @param h Spacing function for a given point.
 * @param starting_points List of starting points for the algorithm (at least 1).
 * @param num_samples Controls the number of generated candidates from each point.
 * @param max_points Maximal number of points. The algorithm terminates once this threshold has been
 * reached. May generate up to `num_samples` more points than `max_points`.
 * 
 * Using references for inside and h function parameters results in a small speedup.
 */
std::vector<Vector2, Eigen::aligned_allocator<Vector2>> fill(
    const std::function<bool(Eigen::Ref<Vector2>)> &inside, 
    const std::function<double(Eigen::Ref<Vector2>)> &h, 
    const std::vector<Vector2, Eigen::aligned_allocator<Vector2>> &points,
    int num_samples,
    size_t max_points
) {
    // Random number generator
    std::mt19937 gen(1337);

    // Run PNP algorithm
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    auto result = pnp<double, 2>(inside, h, points, num_samples, max_points, gen);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "C++ pnp time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

    return result;
}

// The first argument is the module name for Python
PYBIND11_MODULE(pnpwrapper, m) {
    m.def("fill", &fill, "Fill algorithm for 2D domain", 
        py::arg("inside"), py::arg("h"), py::arg("points"), 
        py::arg("num_samples") = 12, 
        py::arg("max_points") = 1000000
    );
}
