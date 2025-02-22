// Barrier functions that grow to infinity as x -> 0+. Includes gradient and
// hessian functions, too. These barrier functions can be used to impose
// inequlity constraints on a function.

#pragma once

#include <Eigen/Core>

namespace ipc {

///////////////////////////////////////////////////////////////////////////////
// Barrier functions from [Li et al. 2020]
///////////////////////////////////////////////////////////////////////////////

/// @brief Function that grows to infinity as x approaches 0 from the right.
///
/// \f\[
///     b(d) = -(d-\hat{d})^2\ln\left(\frac{d}{\hat{d}}\right)
/// \f\]
///
/// @param d The distance.
/// @param dhat Activation distance of the barrier.
/// @return The value of the barrier function at d.
template <typename T> T barrier(const T& d, double dhat);

/// @brief Derivative of the barrier function.
///
/// \f\[
///     b'(d) = (\hat{d}-d) \left( 2\ln\left( \frac{d}{\hat{d}} \right) -
///     \frac{\hat{d}}{d} + 1\right)
/// \f\]
///
/// @param d The distance.
/// @param dhat Activation distance of the barrier.
/// @return The derivative of the barrier wrt d.
double barrier_gradient(double d, double dhat);

/// @brief Second derivative of the barrier function.
///
/// \f\[
///     b''(d) = \left( \frac{\hat{d}}{d} + 2 \right) \frac{\hat{d}}{d} -
///     2\ln\left( \frac{d}{\hat{d}} \right) - 3
/// \f\]
///
/// @param d The distance.
/// @param dhat Activation distance of the barrier.
/// @return The second derivative of the barrier wrt d.
double barrier_hessian(double d, double dhat);

///////////////////////////////////////////////////////////////////////////////
// Physically based barrier functions from [Li et al. 2022]
///////////////////////////////////////////////////////////////////////////////

template <typename T> T physical_barrier(const T& d, const double dhat);

double physical_barrier_gradient(const double d, const double dhat);

double physical_barrier_hessian(const double d, const double dhat);

} // namespace ipc

#include "barrier.tpp"
