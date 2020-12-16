#include <ipc/friction/friction_constraint.hpp>

namespace ipc {

Eigen::VectorX2d FrictionConstraint::compute_potential_gradient_common(
    const Eigen::VectorX3d& relative_displacement, double epsv_times_h) const
{
    Eigen::VectorX2d tangent_relative_displacement =
        tangent_basis.transpose() * relative_displacement;

    double f1_div_rel_disp_norm = f1_SF_div_relative_displacement_norm(
        tangent_relative_displacement.squaredNorm(), epsv_times_h);

    tangent_relative_displacement *=
        f1_div_rel_disp_norm * mu * normal_force_magnitude;

    return tangent_relative_displacement;
}

Eigen::MatrixXd FrictionConstraint::compute_potential_hessian_common(
    const Eigen::VectorX3d& relative_displacement,
    const Eigen::MatrixXd& TT,
    const double epsv_times_h,
    bool project_to_psd,
    const int multiplicity) const
{
    double epsv_times_h_squared = epsv_times_h * epsv_times_h;

    Eigen::VectorX2d tangent_relative_displacement =
        tangent_basis.transpose() * relative_displacement;

    double tangent_relative_displacement_sqnorm =
        tangent_relative_displacement.squaredNorm();

    double f1_div_rel_disp_norm = f1_SF_div_relative_displacement_norm(
        tangent_relative_displacement_sqnorm, epsv_times_h);
    double f2_term = f2_SF(tangent_relative_displacement_sqnorm, epsv_times_h);

    Eigen::MatrixXd local_hess;

    double scale = multiplicity * mu * normal_force_magnitude;
    if (tangent_relative_displacement_sqnorm >= epsv_times_h_squared) {
        // no SPD projection needed
        Eigen::Vector2d ubar(
            -tangent_relative_displacement[1],
            tangent_relative_displacement[0]);
        local_hess = (TT.transpose()
                      * ((scale * f1_div_rel_disp_norm
                          / tangent_relative_displacement_sqnorm)
                         * ubar))
            * (ubar.transpose() * TT);
    } else {
        double tangent_relative_displacement_norm =
            sqrt(tangent_relative_displacement_sqnorm);
        if (tangent_relative_displacement_norm == 0) {
            // no SPD projection needed
            local_hess = ((scale * f1_div_rel_disp_norm) * TT.transpose()) * TT;
        } else {
            // only need to project the inner 2x2 matrix to SPD
            Eigen::Matrix2d inner_hess =
                ((f2_term / tangent_relative_displacement_norm)
                 * tangent_relative_displacement)
                * tangent_relative_displacement.transpose();
            inner_hess.diagonal().array() += f1_div_rel_disp_norm;
            if (project_to_psd) {
                inner_hess = Eigen::project_to_psd(inner_hess);
            }
            inner_hess *= scale;

            // tensor product:
            local_hess = TT.transpose() * inner_hess * TT;
        }
    }

    return local_hess;
}

///////////////////////////////////////////////////////////////////////////////

VertexVertexFrictionConstraint::VertexVertexFrictionConstraint(
    long vertex0_index, long vertex1_index)
    : VertexVertexConstraint(vertex0_index, vertex1_index)
{
}

VertexVertexFrictionConstraint::VertexVertexFrictionConstraint(
    const VertexVertexConstraint& constraint)
    : VertexVertexConstraint(constraint)
{
}

Eigen::VectorXd VertexVertexFrictionConstraint::compute_potential_gradient(
    const Eigen::MatrixXd& U,
    const Eigen::MatrixXi& E,
    const Eigen::MatrixXi& F,
    const double epsv_times_h) const
{
    Eigen::VectorX3d rel_u = relative_displacement(U);
    Eigen::VectorX2d tangent_rel_u =
        compute_potential_gradient_common(rel_u, epsv_times_h);
    tangent_rel_u *= multiplicity;
    return point_point_relative_mesh_displacements(
        tangent_rel_u, tangent_basis);
}

Eigen::MatrixXd VertexVertexFrictionConstraint::compute_potential_hessian(
    const Eigen::MatrixXd& U,
    const Eigen::MatrixXi& E,
    const Eigen::MatrixXi& F,
    const double epsv_times_h,
    const bool project_to_psd) const
{
    Eigen::VectorX3d rel_u = relative_displacement(U);
    Eigen::Matrix<double, 2, 6> TT;
    point_point_TT(tangent_basis, TT);
    return compute_potential_hessian_common(
        rel_u, TT, epsv_times_h, project_to_psd, multiplicity);
}

///////////////////////////////////////////////////////////////////////////////

EdgeVertexFrictionConstraint::EdgeVertexFrictionConstraint(
    long edge_index, long vertex_index)
    : EdgeVertexConstraint(edge_index, vertex_index)
{
}

EdgeVertexFrictionConstraint::EdgeVertexFrictionConstraint(
    const EdgeVertexConstraint& constraint)
    : EdgeVertexConstraint(constraint)
{
}

Eigen::VectorXd EdgeVertexFrictionConstraint::compute_potential_gradient(
    const Eigen::MatrixXd& U,
    const Eigen::MatrixXi& E,
    const Eigen::MatrixXi& F,
    const double epsv_times_h) const
{
    Eigen::VectorX3d rel_u = relative_displacement(U, E);
    Eigen::VectorX2d tangent_rel_u =
        compute_potential_gradient_common(rel_u, epsv_times_h);
    tangent_rel_u *= multiplicity;
    return point_edge_relative_mesh_displacements(
        tangent_rel_u, tangent_basis, closest_point[0]);
}

Eigen::MatrixXd EdgeVertexFrictionConstraint::compute_potential_hessian(
    const Eigen::MatrixXd& U,
    const Eigen::MatrixXi& E,
    const Eigen::MatrixXi& F,
    const double epsv_times_h,
    const bool project_to_psd) const
{
    Eigen::VectorX3d rel_u = relative_displacement(U, E);
    Eigen::Matrix<double, 2, 9> TT;
    point_edge_TT(tangent_basis, closest_point[0], TT);
    return compute_potential_hessian_common(
        rel_u, TT, epsv_times_h, project_to_psd, multiplicity);
}

///////////////////////////////////////////////////////////////////////////////

EdgeEdgeFrictionConstraint::EdgeEdgeFrictionConstraint(
    long edge0_index, long edge1_index)
    : EdgeEdgeConstraint(edge0_index, edge1_index, /*eps_x=*/-1)
{
}

EdgeEdgeFrictionConstraint::EdgeEdgeFrictionConstraint(
    const EdgeEdgeConstraint& constraint)
    : EdgeEdgeConstraint(constraint)
{
}

Eigen::VectorXd EdgeEdgeFrictionConstraint::compute_potential_gradient(
    const Eigen::MatrixXd& U,
    const Eigen::MatrixXi& E,
    const Eigen::MatrixXi& F,
    const double epsv_times_h) const
{
    Eigen::VectorX3d rel_u = relative_displacement(U, E);
    Eigen::VectorX2d tangent_rel_u =
        compute_potential_gradient_common(rel_u, epsv_times_h);
    return edge_edge_relative_mesh_displacements(
        tangent_rel_u, tangent_basis, closest_point);
}

Eigen::MatrixXd EdgeEdgeFrictionConstraint::compute_potential_hessian(
    const Eigen::MatrixXd& U,
    const Eigen::MatrixXi& E,
    const Eigen::MatrixXi& F,
    const double epsv_times_h,
    const bool project_to_psd) const
{
    Eigen::VectorX3d rel_u = relative_displacement(U, E);
    Eigen::Matrix<double, 2, 12> TT;
    edge_edge_TT(tangent_basis, closest_point, TT);
    return compute_potential_hessian_common(
        rel_u, TT, epsv_times_h, project_to_psd);
}

///////////////////////////////////////////////////////////////////////////////

FaceVertexFrictionConstraint::FaceVertexFrictionConstraint(
    long face_index, long vertex_index)
    : FaceVertexConstraint(face_index, vertex_index)
{
}

FaceVertexFrictionConstraint::FaceVertexFrictionConstraint(
    const FaceVertexConstraint& constraint)
    : FaceVertexConstraint(constraint)
{
}

Eigen::VectorXd FaceVertexFrictionConstraint::compute_potential_gradient(
    const Eigen::MatrixXd& U,
    const Eigen::MatrixXi& E,
    const Eigen::MatrixXi& F,
    const double epsv_times_h) const
{
    Eigen::VectorX3d rel_u = relative_displacement(U, F);
    Eigen::VectorX2d tangent_rel_u =
        compute_potential_gradient_common(rel_u, epsv_times_h);
    return point_triangle_relative_mesh_displacements(
        tangent_rel_u, tangent_basis, closest_point);
}

Eigen::MatrixXd FaceVertexFrictionConstraint::compute_potential_hessian(
    const Eigen::MatrixXd& U,
    const Eigen::MatrixXi& E,
    const Eigen::MatrixXi& F,
    const double epsv_times_h,
    const bool project_to_psd) const
{
    Eigen::VectorX3d rel_u = relative_displacement(U, F);
    Eigen::Matrix<double, 2, 12> TT;
    point_triangle_TT(tangent_basis, closest_point, TT);
    return compute_potential_hessian_common(
        rel_u, TT, epsv_times_h, project_to_psd);
}

///////////////////////////////////////////////////////////////////////////////

size_t FrictionConstraints::size() const
{
    return vv_constraints.size() + ev_constraints.size() + ee_constraints.size()
        + fv_constraints.size();
}

size_t FrictionConstraints::num_constraints() const
{
    size_t num_constraints = 0;
    for (const auto& vv_constraint : vv_constraints) {
        num_constraints += vv_constraint.multiplicity;
    }
    for (const auto& ev_constraint : ev_constraints) {
        num_constraints += ev_constraint.multiplicity;
    }
    num_constraints += ee_constraints.size() + fv_constraints.size();
    return num_constraints;
}

void FrictionConstraints::clear()
{
    vv_constraints.clear();
    ev_constraints.clear();
    ee_constraints.clear();
    fv_constraints.clear();
}

FrictionConstraint& FrictionConstraints::operator[](size_t idx)
{
    if (idx < vv_constraints.size()) {
        return vv_constraints[idx];
    }
    idx -= vv_constraints.size();
    if (idx < ev_constraints.size()) {
        return ev_constraints[idx];
    }
    idx -= ev_constraints.size();
    if (idx < ee_constraints.size()) {
        return ee_constraints[idx];
    }
    idx -= ee_constraints.size();
    if (idx < fv_constraints.size()) {
        return fv_constraints[idx];
    }
    assert(false);
    throw "Invalid friction constraint index!";
}

const FrictionConstraint& FrictionConstraints::operator[](size_t idx) const
{
    if (idx < vv_constraints.size()) {
        return vv_constraints[idx];
    }
    idx -= vv_constraints.size();
    if (idx < ev_constraints.size()) {
        return ev_constraints[idx];
    }
    idx -= ev_constraints.size();
    if (idx < ee_constraints.size()) {
        return ee_constraints[idx];
    }
    idx -= ee_constraints.size();
    if (idx < fv_constraints.size()) {
        return fv_constraints[idx];
    }
    assert(false);
    throw "Invalid friction constraint index!";
}

} // namespace ipc
