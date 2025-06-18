#ifndef AIMM_CS_DUCMKF_DUCMKF_H
#define AIMM_CS_DUCMKF_DUCMKF_H

#include "kalman_filter.h"
#include "types.h"
#include <cmath>

namespace aimm_cs_ducmkf
{

    /**
     * @brief Decorrelated Unbiased Conversion Measurement Kalman Filter
     *
     * This class implements the DUCMKF algorithm for handling nonlinear
     * measurements in radar tracking applications. It converts polar/spherical
     * measurements to Cartesian coordinates while maintaining unbiased and
     * decorrelated estimates.
     */
    class DUCMKF : public KalmanFilter
    {
      public:
        /**
         * @brief Constructor
         * @param state_dim Dimension of state vector
         * @param measurement_dim Dimension of measurement vector
         */
        DUCMKF(int state_dim = 6, int measurement_dim = 3);

        /**
         * @brief Destructor
         */
        virtual ~DUCMKF() = default;

        /**
         * @brief Update step with polar/spherical measurements
         * @param measurement Polar measurement [range, azimuth, elevation]
         * @param measurement_covariance Measurement noise covariance
         */
        void
        updateWithPolarMeasurement(const MeasurementVector &measurement,
                                   const MeasurementMatrix &measurement_covariance);

        /**
         * @brief Update step with Cartesian measurements
         * @param measurement Cartesian measurement [x, y, z]
         * @param measurement_covariance Measurement noise covariance
         */
        void updateWithCartesianMeasurement(const ObservationVector &measurement,
                                            const Matrix3d &measurement_covariance);

        /**
         * @brief Convert polar measurement to Cartesian with bias correction
         * @param polar_measurement [range, azimuth, elevation] in polar coordinates
         * @param polar_covariance Measurement covariance in polar coordinates
         * @param predicted_state Predicted state for decorrelation
         * @param converted_measurement Output converted measurement
         * @param converted_covariance Output converted covariance
         */
        void convertPolarToCartesian(const MeasurementVector &polar_measurement,
                                     const MeasurementMatrix &polar_covariance,
                                     const StateVector &predicted_state,
                                     ObservationVector &converted_measurement,
                                     Matrix3d &converted_covariance);

        /**
         * @brief Calculate conversion bias and remove it (UCM step)
         * @param polar_measurement Polar measurement
         * @param polar_covariance Polar covariance
         * @param cartesian_measurement Initial converted measurement
         * @return Bias-corrected measurement
         */
        ObservationVector
        removeConversionBias(const MeasurementVector &polar_measurement,
                             const MeasurementMatrix &polar_covariance,
                             const ObservationVector &cartesian_measurement);

        /**
         * @brief Calculate decorrelated covariance matrix (DUCM step)
         * @param polar_measurement Polar measurement
         * @param polar_covariance Polar covariance
         * @param predicted_cartesian Predicted Cartesian position
         * @param predicted_covariance Predicted state covariance
         * @return Decorrelated covariance matrix
         */
        Matrix3d calculateDecorrelatedCovariance(
            const MeasurementVector &polar_measurement,
            const MeasurementMatrix &polar_covariance,
            const Vector3d &predicted_cartesian,
            const Matrix3d &predicted_position_covariance);

        /**
         * @brief Get the Jacobian matrix for polar to Cartesian conversion
         * @param range Range measurement
         * @param azimuth Azimuth angle in radians
         * @param elevation Elevation angle in radians
         * @return 3x3 Jacobian matrix
         */
        Matrix3d getConversionJacobian(double range, double azimuth,
                                       double elevation) const;

        /**
         * @brief Calculate second-order bias correction terms
         * @param polar_measurement Polar measurement
         * @param polar_covariance Polar covariance
         * @return Bias correction vector
         */
        Vector3d
        calculateSecondOrderBias(const MeasurementVector &polar_measurement,
                                 const MeasurementMatrix &polar_covariance) const;

        /**
         * @brief Validate measurement consistency using NEES test
         * @param innovation Innovation vector
         * @param innovation_covariance Innovation covariance
         * @return NEES value (should be close to measurement dimension for
         * consistency)
         */
        double calculateNEES(const ObservationVector &innovation,
                             const Matrix3d &innovation_covariance) const;

      protected:
        /**
         * @brief Convert spherical to Cartesian coordinates
         * @param range Range in meters
         * @param azimuth Azimuth in radians
         * @param elevation Elevation in radians
         * @return Cartesian coordinates [x, y, z]
         */
        Vector3d sphericalToCartesian(double range, double azimuth,
                                      double elevation) const;

        /**
         * @brief Convert Cartesian to spherical coordinates
         * @param cartesian Cartesian coordinates [x, y, z]
         * @return Spherical coordinates [range, azimuth, elevation]
         */
        Vector3d cartesianToSpherical(const Vector3d &cartesian) const;

      private:
        // Internal variables for bias correction
        bool use_second_order_bias_;
        double max_range_;
        double min_range_;

        // Statistics for performance monitoring
        std::vector<double> nees_history_;
        double average_nees_;
    };

} // namespace aimm_cs_ducmkf

#endif // AIMM_CS_DUCMKF_DUCMKF_H
