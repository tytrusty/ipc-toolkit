#include <catch2/catch.hpp>

#include <finitediff.hpp>

#include <ipc/distance/point_plane.hpp>

using namespace ipc;

TEST_CASE("Point-plane distance", "[distance][point-plane]")
{
    double x = GENERATE(take(10, random(-100.0, 100.0)));
    double y = GENERATE(take(10, random(-100.0, 100.0)));
    double z = GENERATE(take(10, random(-100.0, 100.0)));
    Eigen::Vector3d p(x, y, z);

    double y_plane = GENERATE(take(10, random(-100.0, 100.0)));
    Eigen::Vector3d t0(-1, y_plane, 0), t1(1, y_plane, -1), t2(1, y_plane, 0);

    DistanceMode dmode = GENERATE(DistanceMode::SQRT, DistanceMode::SQUARED);

    double distance = point_plane_distance(p, t0, t1, t2, dmode);
    double expected_distance = abs(y - y_plane);

    if (dmode == DistanceMode::SQRT) {
        expected_distance = std::sqrt(expected_distance);
    }
    CHECK(distance == Approx(expected_distance * expected_distance));
}

TEST_CASE("Point-plane distance gradient", "[distance][point-plane][gradient]")
{
    double x = GENERATE(take(10, random(-10.0, 10.0)));
    double y = GENERATE(take(10, random(-10.0, 10.0)));
    double z = GENERATE(take(10, random(-10.0, 10.0)));
    Eigen::Vector3d p(x, y, z);

    double y_plane = GENERATE(take(10, random(-10.0, 10.0)));
    Eigen::Vector3d t0(-1, y_plane, 0), t1(1, y_plane, -1), t2(1, y_plane, 0);

    DistanceMode dmode = GENERATE(DistanceMode::SQRT, DistanceMode::SQUARED);

    Eigen::VectorXd grad;
    point_plane_distance_gradient(p, t0, t1, t2, dmode, grad);

    Eigen::VectorXd x_vec(12);
    x_vec << p, t0, t1, t2;
    Eigen::VectorXd expected_grad;
    expected_grad.resize(grad.size());
    fd::finite_gradient(
        x_vec,
        [&dmode](const Eigen::VectorXd& x) {
            return point_plane_distance(
                x.head<3>(), x.segment<3>(3), x.segment<3>(6), x.tail<3>(),
                dmode);
        },
        expected_grad);

    CAPTURE((grad - expected_grad).norm());
    CHECK(fd::compare_gradient(grad, expected_grad));
}

TEST_CASE("Point-plane distance hessian", "[distance][point-plane][hessian]")
{
    double x = GENERATE(take(10, random(-10.0, 10.0)));
    double y = GENERATE(take(10, random(-10.0, 10.0)));
    double z = GENERATE(take(10, random(-10.0, 10.0)));
    Eigen::Vector3d p(x, y, z);

    double y_plane = GENERATE(take(10, random(-10.0, 10.0)));
    Eigen::Vector3d t0(-1, y_plane, 0), t1(1, y_plane, -1), t2(1, y_plane, 0);

    DistanceMode dmode = DistanceMode::SQUARED;

    Eigen::MatrixXd hess;
    point_plane_distance_hessian(p, t0, t1, t2, dmode, hess);

    Eigen::VectorXd x_vec(12);
    x_vec << p, t0, t1, t2;
    Eigen::MatrixXd expected_hess;
    expected_hess.resize(hess.rows(), hess.cols());
    fd::finite_hessian(
        x_vec,
        [&dmode](const Eigen::VectorXd& x) {
            return point_plane_distance(
                x.head<3>(), x.segment<3>(3), x.segment<3>(6), x.tail<3>(),
                dmode);
        },
        expected_hess);

    CAPTURE((hess - expected_hess).norm());
    CHECK(fd::compare_hessian(hess, expected_hess, 5e-2));
}
