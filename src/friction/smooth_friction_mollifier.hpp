#pragma once

#include <ipc/collision_constraint.hpp>
#include <ipc/friction/relative_displacement.hpp>
#include <ipc/utils/eigen_ext.hpp>

namespace ipc {

// C1 clamping
template <typename T> inline T f0_SF(const T& x, const double& epsv_times_h)
{
    assert(epsv_times_h >= 0);
    if (abs(x) >= epsv_times_h) {
        return x;
    }
    return x * x * (-x / (3 * epsv_times_h) + 1) / epsv_times_h
        + epsv_times_h / 3;
}

/// Derivative of f0_SF divided by x
template <typename T>
inline T f1_SF_over_x(const T& x, const double& epsv_times_h)
{
    assert(epsv_times_h >= 0);
    if (abs(x) >= epsv_times_h) {
        return 1 / x;
    }
    return (-x / epsv_times_h + 2) / epsv_times_h;
}

template <typename T>
inline T f2(const T& x, const double& epsv_times_h)
{
    // f1 = (-x^2/ epsv_times_h + 2x) / epsv_times_h
    // f2 = f1' = -2x / epsv_times_h^2 + 2 / epsv_times_h
    assert(epsv_times_h >= 0);
    if (abs(x) >= epsv_times_h) {
        return 0;
    }
    return (-2 * x / epsv_times_h + 2) / epsv_times_h;
}

/// \f$\frac{f_1'(x)x + f_1(x)}{x^3}\f$
template <typename T>
inline T df1_x_minus_f1_over_x3(const T& x, const double& epsv_times_h)
{
    assert(epsv_times_h >= 0);
    if (abs(x) >= epsv_times_h) {
        return -1 / (x * x * x);
    }
    return -1 / (x * epsv_times_h * epsv_times_h);
}

} // namespace ipc
