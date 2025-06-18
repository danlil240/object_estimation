#include "../include/ducmkf.h"
#include <cmath>
#include <stdexcept>

namespace aimm_cs_ducmkf
{
    DUCMKF::DUCMKF(int state_dim, int measurement_dim)
        : KalmanFilter(state_dim, measurement_dim), use_second_order_bias_(true),
          max_range_(1000000.0), min_range_(0.01), average_nees_(0.0)
    {
        nees_history_.reserve(1000); // Reserve space for NEES history
    }


    void DUCMKF::updateWithCartesianMeasurement(
        const ObservationVector &measurement,
        const Matrix3d &measurement_covariance)
    {
        if (!initialized_)
        {
            throw std::runtime_error("Filter not initialized");
        }
        // Update using Cartesian measurement
        update(measurement, measurement_covariance, Matrix3d::Identity());
    }

    void DUCMKF::updateWithPolarMeasurement(
        const MeasurementVector &measurement,
        const MeasurementMatrix &measurement_covariance)
    {
        if (!initialized_)
        {
            throw std::runtime_error("Filter not initialized");
        }
        // Extract position from predicted state [x, y, z, vx, vy, vz]
        Vector3d predicted_position = predicted_state_.head<3>();
        Matrix3d predicted_position_covariance =
            predicted_covariance_.block<3, 3>(0, 0);
        LOG_DEBUG("Predicted position: {}", fmt_mat(predicted_position));
        // Convert polar measurement to Cartesian
        ObservationVector converted_measurement;
        Matrix3d converted_covariance;
        convertPolarToCartesian(measurement, measurement_covariance, predicted_state_,
                                converted_measurement, converted_covariance);
        LOG_DEBUG("Converted measurement: {}", fmt_mat(converted_measurement));
        LOG_DEBUG("Converted covariance: {}", fmt_mat(converted_covariance));
        // Create measurement matrix H for position observation
        MatrixXd H = MatrixXd::Zero(3, state_dim_);
        H.block<3, 3>(0, 0) = Matrix3d::Identity(); // Observe position only
        // Update using converted Cartesian measurement
        update(converted_measurement, converted_covariance, H);
        // Calculate and store NEES for performance monitoring
        Vector3d innovation = converted_measurement - H * predicted_state_;
        double nees = calculateNEES(innovation, converted_covariance);
        nees_history_.push_back(nees);
        // Keep history size manageable
        if (nees_history_.size() > 1000)
        {
            nees_history_.erase(nees_history_.begin());
        }
        // Update average NEES
        average_nees_ = 0.0;
        for (double nees_val : nees_history_)
        {
            average_nees_ += nees_val;
        }
        average_nees_ /= nees_history_.size();
    }

    void DUCMKF::convertPolarToCartesian(const MeasurementVector &polar_measurement,
                                         const MeasurementMatrix &polar_covariance,
                                         const StateVector &predicted_state,
                                         ObservationVector &converted_measurement,
                                         Matrix3d &converted_covariance)
    {
        double range = polar_measurement(0);
        double azimuth = polar_measurement(1);
        double elevation = polar_measurement(2);
        // Validate range with more reasonable bounds
        if (range < min_range_ || range > max_range_ || std::isnan(range))
        {
            throw std::invalid_argument("Range measurement out of valid bounds or NaN");
        }
        // Validate angles
        if (std::isnan(azimuth) || std::isnan(elevation))
        {
            throw std::invalid_argument("Angle measurements contain NaN values");
        }
        // Check for extreme angle values that could cause numerical issues
        if (std::abs(elevation) > M_PI / 2.0)
        {
            throw std::invalid_argument(
                "Elevation angle out of valid range [-pi/2, pi/2]");
        }
        // Initial conversion (biased)
        converted_measurement = sphericalToCartesian(range, azimuth, elevation);
        // Check for NaN/Inf in converted measurement
        for (int i = 0; i < 3; ++i)
        {
            if (std::isnan(converted_measurement(i)) ||
                    std::isinf(converted_measurement(i)))
            {
                throw std::invalid_argument(
                    "Converted measurement contains NaN or Inf values");
            }
        }
        // Remove conversion bias (UCM step)
        if (use_second_order_bias_)
        {
            converted_measurement = removeConversionBias(
                                        polar_measurement, polar_covariance, converted_measurement);
        }
        // Calculate decorrelated covariance (DUCM step)
        Vector3d predicted_cartesian = predicted_state.head<3>();
        Matrix3d predicted_position_covariance =
            predicted_covariance_.block<3, 3>(0, 0);
        converted_covariance = calculateDecorrelatedCovariance(
                                   polar_measurement, polar_covariance, predicted_cartesian,
                                   predicted_position_covariance);
    }

    ObservationVector
    DUCMKF::removeConversionBias(const MeasurementVector &polar_measurement,
                                 const MeasurementMatrix &polar_covariance,
                                 const ObservationVector &cartesian_measurement)
    {
        // Calculate second-order bias correction
        Vector3d bias_correction =
            calculateSecondOrderBias(polar_measurement, polar_covariance);
        // Apply bias correction
        return cartesian_measurement - bias_correction;
    }

    Matrix3d DUCMKF::calculateDecorrelatedCovariance(
        const MeasurementVector &polar_measurement,
        const MeasurementMatrix &polar_covariance,
        const Vector3d &predicted_cartesian,
        const Matrix3d &predicted_position_covariance)
    {
        double range = polar_measurement(0);
        double azimuth = polar_measurement(1);
        double elevation = polar_measurement(2);
        LOG_DEBUG("DUCMKF Debug - Range: {}, Azimuth: {}, Elevation: {}",
                  range, azimuth, elevation);
        LOG_DEBUG("DUCMKF Debug - Predicted position covariance: {}",
                  fmt_mat(predicted_position_covariance));
        // Get conversion Jacobian
        Matrix3d J = getConversionJacobian(range, azimuth, elevation);
        LOG_DEBUG("DUCMKF Debug - Conversion Jacobian: {}", fmt_mat(J));
        // Initial covariance conversion: R_cart = J * R_polar * J^T
        Matrix3d R_cart = J * polar_covariance * J.transpose();
        LOG_DEBUG("DUCMKF Debug - Initial converted covariance (R_cart): {}",
                  fmt_mat(R_cart));
        // Ensure minimum diagonal elements for numerical stability
        for (int i = 0; i < 3; ++i)
        {
            if (R_cart(i, i) < 1e-8)
            {
                R_cart(i, i) = 1e-8;
            }
        }
        // Ensure positive definiteness of R_cart
        Eigen::LLT<Matrix3d> llt_cart(R_cart);
        if (llt_cart.info() != Eigen::Success)
        {
            // Add small diagonal elements to make it positive definite
            R_cart += Matrix3d::Identity() * 1e-6;
        }
        // Calculate innovation covariance: S = R_cart + P_pred
        Matrix3d S = R_cart + predicted_position_covariance;
        LOG_DEBUG("DUCMKF Debug - Innovation covariance (S): {}", fmt_mat(S));
        // Ensure S is well-conditioned and positive definite
        Eigen::LLT<Matrix3d> llt_S(S);
        if (llt_S.info() != Eigen::Success)
        {
            // Add small diagonal elements to make it positive definite
            S += Matrix3d::Identity() * 1e-6;
            llt_S = Eigen::LLT<Matrix3d>(S);
        }
        // Calculate Kalman gain: K = R_cart * S^(-1)
        Matrix3d K;
        if (llt_S.info() == Eigen::Success)
        {
            // Use Cholesky solve for numerical stability instead of direct inversion
            K = llt_S.solve(R_cart.transpose()).transpose();
            LOG_DEBUG("DUCMKF Debug - Kalman gain (K): {}", fmt_mat(K));
            // Calculate decorrelated covariance: R_decorr = R_cart - K * S * K^T
            Matrix3d R_decorr = R_cart - K * S * K.transpose();
            LOG_DEBUG("DUCMKF Debug - Decorrelated covariance: {}", fmt_mat(R_decorr));
            // Ensure positive definiteness of final result
            Eigen::LLT<Matrix3d> llt_decorr(R_decorr);
            if (llt_decorr.info() == Eigen::Success)
            {
                // Additional stability check - ensure minimum diagonal values
                for (int i = 0; i < 3; ++i)
                {
                    if (R_decorr(i, i) < 1e-8)
                    {
                        R_decorr(i, i) = 1e-8;
                    }
                }
                // Verify the result makes sense (decorrelated covariance should be smaller)
                double trace_original = R_cart.trace();
                double trace_decorr = R_decorr.trace();
                LOG_DEBUG("DUCMKF Debug - Trace comparison: Original={}, Decorrelated={}",
                          trace_original, trace_decorr);
                if (trace_decorr <= trace_original && trace_decorr > 0)
                {
                    return R_decorr;
                }
                else
                {
                    LOG_DEBUG("DUCMKF Debug - Decorrelation produced invalid result, using fallback");
                }
            }
            else
            {
                LOG_DEBUG("DUCMKF Debug - Decorrelated covariance not positive definite");
            }
        }
        else
        {
            LOG_DEBUG("DUCMKF Debug - Innovation covariance inversion failed");
        }
        // Fallback to initial covariance if decorrelation fails
        LOG_DEBUG("DUCMKF Debug - Using fallback initial covariance");
        return R_cart;
    }

// Alternative implementation using explicit matrix inversion (less numerically stable)
    // Matrix3d DUCMKF::calculateDecorrelatedCovarianceAlternative(
    //     const MeasurementVector &polar_measurement,
    //     const MeasurementMatrix &polar_covariance,
    //     const Vector3d &predicted_cartesian,
    //     const Matrix3d &predicted_position_covariance)
    // {
    //     double range = polar_measurement(0);
    //     double azimuth = polar_measurement(1);
    //     double elevation = polar_measurement(2);
    //     // Get conversion Jacobian and initial covariance
    //     Matrix3d J = getConversionJacobian(range, azimuth, elevation);
    //     Matrix3d R_cart = J * polar_covariance * J.transpose();
    //     // Ensure positive definiteness
    //     Eigen::LLT<Matrix3d> llt_cart(R_cart);
    //     if (llt_cart.info() != Eigen::Success)
    //     {
    //         R_cart += Matrix3d::Identity() * 1e-6;
    //     }
    //     // Innovation covariance
    //     Matrix3d S = R_cart + predicted_position_covariance;
    //     // Check if S is invertible
    //     double det_S = S.determinant();
    //     if (std::abs(det_S) < 1e-12)
    //     {
    //         S += Matrix3d::Identity() * 1e-6;
    //     }
    //     try
    //     {
    //         // Calculate K = R_cart * S^(-1)
    //         Matrix3d S_inv = S.inverse();
    //         Matrix3d K = R_cart * S_inv;
    //         // Calculate decorrelated covariance: R_decorr = R_cart - K * S * K^T
    //         Matrix3d R_decorr = R_cart - K * S * K.transpose();
    //         // Verify positive definiteness
    //         Eigen::LLT<Matrix3d> llt_decorr(R_decorr);
    //         if (llt_decorr.info() == Eigen::Success)
    //         {
    //             return R_decorr;
    //         }
    //     }
    //     catch (const std::exception &e)
    //     {
    //         LOG_DEBUG("DUCMKF Debug - Matrix inversion failed: {}", e.what());
    //     }
    //     // Fallback
    //     return R_cart;
    // }
    Vector3d DUCMKF::calculateSecondOrderBias(
        const MeasurementVector &polar_measurement,
        const MeasurementMatrix &polar_covariance) const
    {
        double range = polar_measurement(0);
        double azimuth = polar_measurement(1);
        double elevation = polar_measurement(2);
        // Second-order bias terms from Taylor expansion
        // Bias ≈ 0.5 * trace(Hessian * Covariance)
        double range_var = polar_covariance(0, 0);
        double azimuth_var = polar_covariance(1, 1);
        double elevation_var = polar_covariance(2, 2);
        // Second derivatives for bias calculation
        double cos_az = cos(azimuth);
        double sin_az = sin(azimuth);
        double cos_el = cos(elevation);
        double sin_el = sin(elevation);
        // Corrected bias components based on second derivatives
        double bias_x =
            0.5 * range *
            (-cos_el * cos_az * azimuth_var + -cos_el * cos_az * elevation_var);
        double bias_y =
            0.5 * range *
            (-cos_el * sin_az * azimuth_var + -cos_el * sin_az * elevation_var);
        double bias_z = 0.5 * range * (-sin_el * elevation_var);
        return Vector3d(bias_x, bias_y, bias_z);
    }

    Matrix3d DUCMKF::getConversionJacobian(double range, double azimuth,
                                           double elevation) const
    {
        Matrix3d J = Matrix3d::Zero();
        double cos_az = cos(azimuth);
        double sin_az = sin(azimuth);
        double cos_el = cos(elevation);
        double sin_el = sin(elevation);
        // ∂x/∂r, ∂x/∂θ, ∂x/∂φ
        J(0, 0) = cos_el * cos_az;          // ∂x/∂r
        J(0, 1) = -range * cos_el * sin_az; // ∂x/∂θ
        J(0, 2) = -range * sin_el * cos_az; // ∂x/∂φ
        // ∂y/∂r, ∂y/∂θ, ∂y/∂φ
        J(1, 0) = cos_el * sin_az;          // ∂y/∂r
        J(1, 1) = range * cos_el * cos_az;  // ∂y/∂θ
        J(1, 2) = -range * sin_el * sin_az; // ∂y/∂φ
        // ∂z/∂r, ∂z/∂θ, ∂z/∂φ
        J(2, 0) = sin_el;         // ∂z/∂r
        J(2, 1) = 0.0;            // ∂z/∂θ
        J(2, 2) = range * cos_el; // ∂z/∂φ
        return J;
    }

    double DUCMKF::calculateNEES(const ObservationVector &innovation,
                                 const Matrix3d &innovation_covariance) const
    {
        // Normalized Estimation Error Squared
        // NEES = innovation^T * S^(-1) * innovation
        // Should be approximately equal to measurement dimension (3) for consistency
        Eigen::LLT<Matrix3d> llt(innovation_covariance);
        if (llt.info() == Eigen::Success)
        {
            // Use solve method instead of inverse for better numerical stability
            Vector3d normalized_innovation = llt.solve(innovation);
            return innovation.transpose() * normalized_innovation;
        }
        return 0.0;
    }

    Vector3d DUCMKF::sphericalToCartesian(double range, double azimuth,
                                          double elevation) const
    {
        double x = range * cos(elevation) * cos(azimuth);
        double y = range * cos(elevation) * sin(azimuth);
        double z = range * sin(elevation);
        return Vector3d(x, y, z);
    }

    Vector3d DUCMKF::cartesianToSpherical(const Vector3d &cartesian) const
    {
        double range = cartesian.norm();
        double azimuth = (range > 1e-9) ? atan2(cartesian.y(), cartesian.x()) : 0.0;
        double elevation = (range > 1e-9) ? asin(cartesian.z() / range) : 0.0;
        return Vector3d(range, azimuth, elevation);
    }

} // namespace aimm_cs_ducmkf