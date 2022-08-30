#pragma once

#include <Eigen/Core>

#include <ipc/utils/eigen_ext.hpp>
#include <ipc/distance/distance_type.hpp>

namespace ipc {

/// @brief Compute the distance between two points.
/// @note The distance is actually squared distance.
/// @param[in] p0 The first point.
/// @param[in] p1 The second point.
/// @return The distance between p0 and p1.
template <typename DerivedP0, typename DerivedP1>
inline auto point_point_distance(
    const Eigen::MatrixBase<DerivedP0>& p0,
    const Eigen::MatrixBase<DerivedP1>& p1,
    const DistanceMode dmode)
{
    if (dmode == DistanceMode::SQUARED) {
        return (p1 - p0).squaredNorm();
    } else {
        return (p1 - p0).norm();
    }
}

/// @brief Compute the gradient of the distance between two points.
/// @note The distance is actually squared distance.
/// @param[in] p0 The first point.
/// @param[in] p1 The second point.
/// @param[out] grad The computed gradient.
template <typename DerivedP0, typename DerivedP1, typename DerivedGrad>
inline void point_point_distance_gradient(
    const Eigen::MatrixBase<DerivedP0>& p0,
    const Eigen::MatrixBase<DerivedP1>& p1,
    const DistanceMode dmode,
    Eigen::PlainObjectBase<DerivedGrad>& grad)
{

    assert(p0.size() == p1.size());
    grad.resize(p0.size() + p1.size());

    if (dmode == DistanceMode::SQUARED) {
        grad.head(p0.size()) = 2.0 * (p0 - p1);
        grad.tail(p1.size()) = -grad.head(p0.size());
    } else {
        double tmp = (p1 - p0).norm();
        grad.head(p0.size()) = (p0-p1) / tmp;
        grad.tail(p1.size()) = -grad.head(p0.size());
    }

}

/// @brief Compute the hessian of the distance between two points.
/// @note The distance is actually squared distance.
/// @param[in] p0 The first point.
/// @param[in] p1 The second point.
/// @param[out] hess The computed hessian.
template <typename DerivedP0, typename DerivedP1, typename DerivedHess>
inline void point_point_distance_hessian(
    const Eigen::MatrixBase<DerivedP0>& p0,
    const Eigen::MatrixBase<DerivedP1>& p1,
    const DistanceMode dmode,
    Eigen::PlainObjectBase<DerivedHess>& hess)
{
    int dim = p0.size();
    assert(p1.size() == dim);

    if (dmode == DistanceMode::SQRT) {
        logger().warn("point_point_distance_hessian sqrt hessian unsupported");
    }

    hess.resize(2 * dim, 2 * dim);

    hess.setZero();
    hess.diagonal().setConstant(2.0);
    for (int i = 0; i < dim; i++) {
        hess(i, i + dim) = hess(i + dim, i) = -2;
    }
}

} // namespace ipc
