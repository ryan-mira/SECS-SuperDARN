#include <medusa/Medusa.hpp>
#include <medusa/IO.hpp>
#include <iostream>

using namespace mm;

// Input arguments are filenames
// argv[1] -- file with locations (first 3 columns) and velocities (last 3 columns)
// argv[2] -- file for output of divergence
int main(int argc, char* argv[]) {
    std::vector<std::string> args(&argv[0], &argv[0 + argc]);
    Eigen::MatrixXd matrix = CSV::readEigen(args[1]);

    // Add all locations from the input file as internal nodes to an unknown shape
    // Find 50 closets nodes as the support of each node
    UnknownShape<Vec3d> shape;
    DomainDiscretization<Vec3d> domain(shape);
    for (int i = 0; i < matrix.rows(); i++) {
        Vec3d vec(matrix(i, 0), matrix(i, 1), matrix(i, 2));
        domain.addInternalNode(vec, 1);
    }
    domain.findSupport(FindClosest(25));

    // Construct a vector field from the velocities
    VectorField3d velocityField(matrix.rows());
    for (int i = 0; i < matrix.rows(); i++) {
        Vec3d vec(matrix(i, 3), matrix(i, 4), matrix(i, 5));
        velocityField[i] = vec;
    }

    // Use WLS engine with monomials up to quadratic order
    // Compute the shapes and construct an explicit operator from the storage
    WLS<Monomials<Vec3d>, NoWeight<Vec3d>, ScaleToFarthest> wls(2);
    auto storage = domain.computeShapes(wls);
    ExplicitVectorOperators<decltype(storage)> op(storage);

    // Calculate divergence at every velocity point
    std::ofstream file;
    file.open(args[2]);
    for (int i = 0; i < matrix.rows(); i++) {
        double div = op.div(velocityField, i);
        file << div << "\n";
    }
    file.close();

    return 0;
}
