#ifndef FILLALGORITMS_PNP_HPP
#define FILLALGORITMS_PNP_HPP

#include <random>
#include <cassert>
#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/StdVector>
#include "nanoflann.hpp"

/**
 * Standalone implementation of the fill algorithm presented in the paper.
 * @tparam scalar_t Floating point number type, eg. `float` or `double`.
 * @tparam dim Dimension of the domain.
 * @param is_inside Characteristic function of the domain, which takes a point and returns `true` if
 * and only if given point is inside the domain.
 * @param h Function prescribing local nodal spacing, which takes a point and returns a positive
 * floating point number.
 * @param starting_points List of starting points for the algorithm.
 * @param num_samples Controls the number of generated candidates from each point. For 2-D it is the
 * actual number of candidates, for 3-D it is the number of candidates on the great circle. Its value
 * is ignored in 1-D.
 * @param max_points Maximal number of points. The algorithm terminates once this threshold has been
 * reached. May generate up to `num_samples` more points than `max_points`.
 * @param generator (Pseudo) random generator to be used for randomization.
 */
template <typename scalar_t, int dim, typename spacing_fn_t, typename domain_fn_t, typename generator_t>
std::vector<Eigen::Matrix<scalar_t, dim, 1>,
            Eigen::aligned_allocator<Eigen::Matrix<scalar_t, dim, 1>>> pnp(
        domain_fn_t is_inside, spacing_fn_t h,
        const std::vector<Eigen::Matrix<scalar_t, dim, 1>,
                Eigen::aligned_allocator<Eigen::Matrix<scalar_t, dim, 1>>>& starting_points,
        int num_samples,
        size_t max_points, generator_t& generator);

/// Implementation details of PNP.
namespace pnp_impl {

template <typename T>
struct Pi {
    static constexpr T value = T(3.1415926535897932385L);
};

/**
 * Discretizes a sphere with given radius uniformly with `num_points` points on the great circle.
 * This is in a class because C++ does not allow partial specializations of template functions.
 * @tparam dim Dimension of the sphere.
 * @tparam scalar_t Data type for numeric computations, e.g. `double` or `float`.
 * @param radius Radius of the sphere.
 * @param num_samples Number of points on the equator, implies nodal spacing `dp = 2*pi*r/n`.
 * @param generator A random generator.
 * @return A vector of discretization points.
 */
template <typename scalar_t, int dim>
struct Sphere {
    template <typename generator_t>
    static std::vector<Eigen::Matrix<scalar_t, dim, 1>,
                       Eigen::aligned_allocator<Eigen::Matrix<scalar_t, dim, 1>>> discretize(
            scalar_t radius, int num_samples, generator_t& generator) {
        scalar_t dphi = 2 * Pi<scalar_t>::value / num_samples;
        std::uniform_real_distribution<scalar_t> distribution(0, Pi<scalar_t>::value);
        scalar_t offset = distribution(generator);
        std::vector<Eigen::Matrix<scalar_t, dim, 1>,
                Eigen::aligned_allocator<Eigen::Matrix<scalar_t, dim, 1>>> result;
        for (int i = 0; i < num_samples / 2; ++i) {
            scalar_t phi = i * dphi + offset;
            if (phi > Pi<scalar_t>::value) phi -= Pi<scalar_t>::value;
            int slice_n = static_cast<int>(std::ceil(num_samples * std::sin(phi)));
            if (slice_n == 0) continue;
            auto slice = Sphere<scalar_t, dim - 1>::discretize(
                    radius * std::sin(phi), slice_n, generator);
            Eigen::Matrix<scalar_t, dim, 1> v;
            for (const auto& p : slice) {
                v[0] = radius * std::cos(phi);
                v.template tail<dim - 1>() = p;
                result.push_back(v);
            }
        }
        return result;
    }
};

/// Two-dimensional base case of the discretisation.
template <typename scalar_t>
struct Sphere<scalar_t, 2> {
    template <typename generator_t>
    static std::vector<Eigen::Matrix<scalar_t, 2, 1>,
            Eigen::aligned_allocator<Eigen::Matrix<scalar_t, 2, 1>>> discretize(
            scalar_t radius, int num_samples, generator_t& generator) {
        scalar_t dphi = 2 * Pi<scalar_t>::value / num_samples;
        std::uniform_real_distribution<scalar_t> distribution(0, 2 * Pi<scalar_t>::value);
        scalar_t offset = distribution(generator);
        std::vector<Eigen::Matrix<scalar_t, 2, 1>,
                Eigen::aligned_allocator<Eigen::Matrix<scalar_t, 2, 1>>> result;
        for (int i = 0; i < num_samples; ++i) {
            scalar_t phi = i * dphi + offset;
            result.emplace_back(radius * std::cos(phi), radius * std::sin(phi));
        }
        return result;
    }
};

/// One-dimensional base case of the discretisation.
template <typename scalar_t>
struct Sphere<scalar_t, 1> {
    template <typename generator_t>
    static std::vector<Eigen::Matrix<scalar_t, 1, 1>,
            Eigen::aligned_allocator<Eigen::Matrix<scalar_t, 1, 1>>> discretize(
            scalar_t radius, int, generator_t&) {
        return {Eigen::Matrix<scalar_t, 1, 1>(-radius), Eigen::Matrix<scalar_t, 1, 1>(radius)};
    }
};


/// Helper class for KDTree with appropriate accessors containing a set of points.
template <typename scalar_t, int dim>
struct PointCloud {
    typedef typename std::vector<Eigen::Matrix<scalar_t, dim, 1>,
                                 Eigen::aligned_allocator<Eigen::Matrix<scalar_t, dim, 1>>>
                                 container_t;
    container_t data;  ///< Points contained in the tree.

    /// Construct an empty point set.
    PointCloud() = default;

    /// Construct from an array of points.
    PointCloud(container_t pts) : data(std::move(pts)) {}

    /// Interface requirement: returns number of data points.
    inline int kdtree_get_point_count() const { return static_cast<int>(data.size()); }

    /// Interface requirement: returns `d`-th coordinate of `idx`-th point.
    inline scalar_t kdtree_get_pt(const size_t idx, int d) const {
        return data[idx][d];
    }

    /// Comply with the interface.
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};

}  // namespace pnp_impl

/// Implementation of presented node placing algorithm.
template <typename scalar_t, int dim, typename spacing_fn_t, typename domain_fn_t, typename generator_t>
std::vector<Eigen::Matrix<scalar_t, dim, 1>,
        Eigen::aligned_allocator<Eigen::Matrix<scalar_t, dim, 1>>> pnp(
        domain_fn_t is_inside, spacing_fn_t h,
        const std::vector<Eigen::Matrix<scalar_t, dim, 1>,
                Eigen::aligned_allocator<Eigen::Matrix<scalar_t, dim, 1>>>& starting_points,
        int num_samples,
        size_t max_points, generator_t& generator) {
    typedef Eigen::Matrix<scalar_t, dim, 1> vec_t;

    typedef nanoflann::KDTreeSingleIndexDynamicAdaptor<
    nanoflann::L2_Simple_Adaptor<scalar_t, pnp_impl::PointCloud<scalar_t, dim>>,
                                 pnp_impl::PointCloud<scalar_t, dim>, dim> kd_tree_t;

    size_t cur_node = 0;
    size_t end_node = starting_points.size();
    double zeta = 1-1e-10;
    assert(cur_node < end_node && "At least one start node must be given");
    pnp_impl::PointCloud<scalar_t, dim> points(starting_points);
    kd_tree_t tree(dim, points, nanoflann::KDTreeSingleIndexAdaptorParams(20));

    while (cur_node < end_node && end_node < max_points) {
        vec_t p = points.data[cur_node];
        scalar_t r = h(p);
        auto candidates = pnp_impl::Sphere<scalar_t, dim>::discretize(r, num_samples, generator);
        for (const auto& f : candidates) {
            vec_t node = p + f;
            if (!is_inside(node)) continue;
            nanoflann::KNNResultSet<scalar_t, int> resultSet(1);
            int neighbor_index;
            scalar_t neighbor_dist;
            resultSet.init(&neighbor_index, &neighbor_dist);
            tree.findNeighbors(resultSet, node.data(), nanoflann::SearchParams(1));
            if (neighbor_dist < (zeta*r)*(zeta*r)) continue;
            points.data.push_back(node);
            tree.addPoints(end_node, end_node);
            end_node++;
        }
        cur_node++;
    }
    return points.data;
}


#endif  // PNP_HPP
