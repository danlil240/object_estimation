#ifndef AIMM_CS_DUCMKF_CURRENT_STATISTICAL_MODEL_H
#define AIMM_CS_DUCMKF_CURRENT_STATISTICAL_MODEL_H

#include "kalman_filter.h"
#include "types.h"
#include <deque>

namespace aimm_cs_ducmkf {

/**
 * @brief Current Statistical Model for maneuvering target tracking
 *
 * The Current Statistical Model (CS) estimates the current acceleration
 * statistics based on recent target motion history. It adapts to changing
 * acceleration patterns and is particularly effective for tracking targets with
 * varying maneuver characteristics.
 */
class CurrentStatisticalModel : public KalmanFilter {
public:
  /**
   * @brief Constructor
   * @param state_dim Dimension of state vector (typically 6 for
   * [x,y,z,vx,vy,vz])
   * @param measurement_dim Dimension of measurement vector
   * @param window_size Number of past samples to use for acceleration
   * estimation
   * @param maneuver_detection_threshold Threshold for maneuver detection
   */
  CurrentStatisticalModel(int state_dim = 6, int measurement_dim = 3,
                          int window_size = 10,
                          double maneuver_detection_threshold = 3.0);

  /**
   * @brief Destructor
   */
  virtual ~CurrentStatisticalModel() = default;

  /**
   * @brief Predict step with adaptive process noise
   * @param dt Time step
   */
  void predict(double dt);

  /**
   * @brief Update the acceleration statistics based on current state
   * @param dt Time step
   */
  void updateAccelerationStatistics(double dt);

  /**
   * @brief Get the current acceleration estimate
   * @return Acceleration vector [ax, ay, az]
   */
  Vector3d getCurrentAcceleration() const { return current_acceleration_; }

  /**
   * @brief Get the acceleration variance estimate
   * @return Acceleration variance vector
   */
  Vector3d getAccelerationVariance() const { return acceleration_variance_; }

  /**
   * @brief Get the maneuver detection probability
   * @return Probability of maneuver (0-1)
   */
  double getManeuverProbability() const { return maneuver_probability_; }

  /**
   * @brief Check if a maneuver is currently detected
   * @return True if maneuver is detected
   */
  bool isManeuverDetected() const { return maneuver_detected_; }

  /**
   * @brief Set the correlation time constant for acceleration
   * @param tau_accel Acceleration correlation time in seconds
   */
  void setAccelerationCorrelationTime(double tau_accel) {
    tau_accel_ = tau_accel;
  }

  /**
   * @brief Get the correlation time constant
   * @return Acceleration correlation time
   */
  double getAccelerationCorrelationTime() const { return tau_accel_; }

  /**
   * @brief Set maximum acceleration limit
   * @param max_accel Maximum acceleration in m/s^2
   */
  void setMaxAcceleration(double max_accel) { max_acceleration_ = max_accel; }

  /**
   * @brief Reset acceleration history
   */
  void resetAccelerationHistory();

  /**
   * @brief Update step with polar measurements (DUCMKF integration)
   * @param measurement Polar measurement [range, azimuth, elevation]
   * @param measurement_covariance Measurement noise covariance
   */
  void
  updateWithPolarMeasurement(const MeasurementVector &measurement,
                             const MeasurementMatrix &measurement_covariance);

protected:
  /**
   * @brief Build the CS model transition matrix
   * @param dt Time step
   * @return State transition matrix with CS dynamics
   */
  StateMatrix buildCSTransitionMatrix(double dt) const;

  /**
   * @brief Build the CS model process noise matrix
   * @param dt Time step
   * @return Process noise covariance matrix
   */
  StateMatrix buildCSProcessNoise(double dt) const;

  /**
   * @brief Estimate current acceleration from velocity history
   * @return Estimated acceleration vector
   */
  Vector3d estimateAccelerationFromHistory() const;

  /**
   * @brief Calculate acceleration variance from history
   * @return Acceleration variance vector
   */
  Vector3d calculateAccelerationVariance() const;

  /**
   * @brief Detect maneuver based on acceleration changes
   * @return Maneuver detection probability
   */
  double detectManeuver() const;

  /**
   * @brief Update process noise based on maneuver detection
   * @param base_noise Base process noise
   * @param maneuver_factor Multiplication factor during maneuvers
   * @return Adapted process noise matrix
   */
  StateMatrix adaptProcessNoise(const StateMatrix &base_noise,
                                double maneuver_factor) const;

private:
  // CS model parameters
  int window_size_;
  double tau_accel_;          // Acceleration correlation time
  double maneuver_threshold_; // Maneuver detection threshold
  double max_acceleration_;   // Maximum allowed acceleration

  // Current acceleration statistics
  Vector3d current_acceleration_;
  Vector3d acceleration_variance_;
  double maneuver_probability_;
  bool maneuver_detected_;

  // History buffers for acceleration estimation
  std::deque<Vector3d> velocity_history_;
  std::deque<double> time_history_;
  std::deque<Vector3d> acceleration_history_;

  // Base process noise parameters
  double base_position_noise_;
  double base_velocity_noise_;
  double maneuver_noise_factor_;

  /**
   * @brief Add velocity sample to history
   * @param velocity Current velocity
   * @param timestamp Current time
   */
  void addVelocityToHistory(const Vector3d &velocity, double timestamp);

  /**
   * @brief Compute least squares acceleration estimate
   * @return Acceleration estimate from least squares fit
   */
  Vector3d computeLeastSquaresAcceleration() const;

  /**
   * @brief Update maneuver detection statistics
   */
  void updateManeuverDetection();
};

} // namespace aimm_cs_ducmkf

#endif // AIMM_CS_DUCMKF_CURRENT_STATISTICAL_MODEL_H
