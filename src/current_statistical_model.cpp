#include "current_statistical_model.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace aimm_cs_ducmkf {

CurrentStatisticalModel::CurrentStatisticalModel(
    int state_dim, int measurement_dim, int window_size,
    double maneuver_detection_threshold)
    : KalmanFilter(state_dim, measurement_dim), window_size_(window_size),
      tau_accel_(5.0), maneuver_threshold_(maneuver_detection_threshold),
      max_acceleration_(20.0), maneuver_probability_(0.0),
      maneuver_detected_(false), base_position_noise_(0.1),
      base_velocity_noise_(0.01), maneuver_noise_factor_(10.0) {

  current_acceleration_.setZero();
  acceleration_variance_.setOnes();

  // Reserve space for history buffers
  velocity_history_.resize(window_size_);
  time_history_.resize(window_size_);
  acceleration_history_.resize(window_size_);
}

void CurrentStatisticalModel::predict(double dt) {
  if (!initialized_) {
    throw std::runtime_error("Filter not initialized");
  }

  // Update acceleration statistics
  updateAccelerationStatistics(dt);

  // Build CS model matrices
  StateMatrix F = buildCSTransitionMatrix(dt);
  StateMatrix Q = buildCSProcessNoise(dt);

  // Perform standard Kalman prediction with CS matrices
  int state_dim = getStateDimension();
  StateVector zero_control = StateVector::Zero(state_dim);
  StateMatrix zero_control_matrix = StateMatrix::Zero(state_dim, state_dim);
  KalmanFilter::predict(F, Q, zero_control, zero_control_matrix, dt);
}

StateMatrix CurrentStatisticalModel::buildCSTransitionMatrix(double dt) const {
  int state_dim = getStateDimension();
  StateMatrix F = StateMatrix::Identity(state_dim, state_dim);

  // Position-velocity coupling
  F.block<3, 3>(0, 3) = Matrix3d::Identity() * dt;

  // Current Statistical model for acceleration
  // Assumes exponential correlation for acceleration: a(t+dt) = exp(-dt/tau) *
  // a(t) + noise But since we're using a 6-state model, acceleration is modeled
  // in process noise

  return F;
}

StateMatrix CurrentStatisticalModel::buildCSProcessNoise(double dt) const {
  int state_dim = getStateDimension();
  StateMatrix Q = StateMatrix::Zero(state_dim, state_dim);

  // Base process noise for position and velocity
  double pos_noise = base_position_noise_;
  double vel_noise = base_velocity_noise_;

  // Acceleration-derived noise model
  // The CS model accounts for acceleration through enhanced process noise
  double accel_noise = acceleration_variance_.norm();

  // Current acceleration magnitude for adaptive noise
  double current_accel_mag = current_acceleration_.norm();

  // Adaptive noise based on current acceleration statistics
  double adaptive_factor =
      1.0 + (current_accel_mag / max_acceleration_) * maneuver_noise_factor_;

  if (maneuver_detected_) {
    adaptive_factor *= maneuver_noise_factor_;
  }

  // Position noise (from acceleration integration)
  double dt2 = dt * dt;
  double dt3 = dt2 * dt;
  double dt4 = dt3 * dt;

  Q.block<3, 3>(0, 0) = Matrix3d::Identity() *
                        (accel_noise * dt4 / 4.0 + pos_noise) * adaptive_factor;
  Q.block<3, 3>(0, 3) = Matrix3d::Identity() * (accel_noise * dt3 / 2.0);
  Q.block<3, 3>(3, 0) = Matrix3d::Identity() * (accel_noise * dt3 / 2.0);
  Q.block<3, 3>(3, 3) =
      Matrix3d::Identity() * (accel_noise * dt2 + vel_noise) * adaptive_factor;

  return Q;
}

void CurrentStatisticalModel::updateAccelerationStatistics(double dt) {
  if (!initialized_)
    return;

  // Update accumulated time
  static double accumulated_time = 0.0;
  accumulated_time += dt;

  // Extract current velocity from state
  Vector3d current_velocity = state_.segment<3>(3);

  // Add current velocity to history with proper timestamp
  addVelocityToHistory(current_velocity, accumulated_time);

  // Estimate acceleration from velocity history
  if (velocity_history_.size() >= 3) {
    current_acceleration_ = estimateAccelerationFromHistory();
    acceleration_variance_ = calculateAccelerationVariance();

    // Update maneuver detection
    updateManeuverDetection();
  }
}

void CurrentStatisticalModel::addVelocityToHistory(const Vector3d &velocity,
                                                   double timestamp) {
  velocity_history_.push_back(velocity);
  time_history_.push_back(timestamp);

  // Maintain window size
  if (velocity_history_.size() > window_size_) {
    velocity_history_.pop_front();
    time_history_.pop_front();
  }
}

Vector3d CurrentStatisticalModel::estimateAccelerationFromHistory() const {
  if (velocity_history_.size() < 3) {
    return Vector3d::Zero();
  }

  // Use least squares estimation of acceleration
  return computeLeastSquaresAcceleration();
}

Vector3d CurrentStatisticalModel::computeLeastSquaresAcceleration() const {
  int n = velocity_history_.size();
  if (n < 3)
    return Vector3d::Zero();

  // Simple finite difference approximation
  // a(t) ≈ (v(t+1) - v(t-1)) / (2*dt)
  Vector3d accel = Vector3d::Zero();
  int count = 0;

  for (int i = 1; i < n - 1; ++i) {
    double dt_forward = time_history_[i + 1] - time_history_[i];
    double dt_backward = time_history_[i] - time_history_[i - 1];
    double dt_avg = (dt_forward + dt_backward) / 2.0;

    if (dt_avg > 1e-6) {
      Vector3d local_accel =
          (velocity_history_[i + 1] - velocity_history_[i - 1]) /
          (2.0 * dt_avg);
      accel += local_accel;
      count++;
    }
  }

  if (count > 0) {
    accel /= count;
  }

  return accel;
}

Vector3d CurrentStatisticalModel::calculateAccelerationVariance() const {
  if (acceleration_history_.size() < 2) {
    return Vector3d::Ones();
  }

  // Calculate variance from acceleration history
  Vector3d mean_accel = Vector3d::Zero();
  for (const auto &accel : acceleration_history_) {
    mean_accel += accel;
  }
  mean_accel /= acceleration_history_.size();

  Vector3d variance = Vector3d::Zero();
  for (const auto &accel : acceleration_history_) {
    Vector3d diff = accel - mean_accel;
    variance += diff.cwiseProduct(diff);
  }
  variance /= (acceleration_history_.size() - 1);

  // Ensure minimum variance
  for (int i = 0; i < 3; ++i) {
    if (variance(i) < 0.01) {
      variance(i) = 0.01;
    }
  }

  return variance;
}

void CurrentStatisticalModel::updateManeuverDetection() {
  // Add current acceleration to history
  acceleration_history_.push_back(current_acceleration_);
  if (acceleration_history_.size() > window_size_) {
    acceleration_history_.pop_front();
  }

  // Calculate maneuver probability based on acceleration magnitude and change
  // rate
  double accel_magnitude = current_acceleration_.norm();
  maneuver_probability_ = detectManeuver();

  // Set maneuver detected flag
  maneuver_detected_ = (maneuver_probability_ > 0.5) && (accel_magnitude > 0.1);
}

double CurrentStatisticalModel::detectManeuver() const {
  if (acceleration_history_.size() < 3) {
    return 0.0;
  }

  // Calculate acceleration change rate
  Vector3d current_accel = acceleration_history_.back();
  Vector3d prev_accel = acceleration_history_[acceleration_history_.size() - 2];
  Vector3d accel_change = current_accel - prev_accel;

  double change_magnitude = accel_change.norm();
  double accel_magnitude = current_accel.norm();

  // Probability based on normalized acceleration magnitude and change rate
  double prob_magnitude = std::min(1.0, accel_magnitude / max_acceleration_);
  double prob_change =
      std::min(1.0, change_magnitude / (max_acceleration_ * 0.1));

  // Combined probability
  double combined_prob = 0.6 * prob_magnitude + 0.4 * prob_change;

  return std::min(1.0, combined_prob);
}

void CurrentStatisticalModel::resetAccelerationHistory() {
  velocity_history_.clear();
  time_history_.clear();
  acceleration_history_.clear();
  current_acceleration_.setZero();
  acceleration_variance_.setOnes();
  maneuver_probability_ = 0.0;
  maneuver_detected_ = false;
}

void CurrentStatisticalModel::updateWithPolarMeasurement(
    const MeasurementVector &measurement,
    const MeasurementMatrix &measurement_covariance) {
  if (!initialized_) {
    throw std::runtime_error("Filter not initialized");
  }

  // Convert polar measurement to Cartesian using DUCMKF principles
  double range = measurement(0);
  double azimuth = measurement(1);
  double elevation = measurement(2);

  // Basic spherical to Cartesian conversion
  Vector3d cartesian_measurement;
  cartesian_measurement(0) = range * cos(elevation) * cos(azimuth);
  cartesian_measurement(1) = range * cos(elevation) * sin(azimuth);
  cartesian_measurement(2) = range * sin(elevation);

  // Apply bias correction (simplified second-order bias)
  double range_var = measurement_covariance(0, 0);
  double azimuth_var = measurement_covariance(1, 1);
  double elevation_var = measurement_covariance(2, 2);

  // Second-order bias correction terms
  double cos_az = cos(azimuth);
  double sin_az = sin(azimuth);
  double cos_el = cos(elevation);
  double sin_el = sin(elevation);

  Vector3d bias_correction;
  bias_correction(0) =
      0.5 * range * (-cos_el * cos_az * (azimuth_var + elevation_var));
  bias_correction(1) =
      0.5 * range * (-cos_el * sin_az * (azimuth_var + elevation_var));
  bias_correction(2) = 0.5 * range * (-sin_el * elevation_var);

  cartesian_measurement -= bias_correction;

  // Convert covariance using Jacobian
  Matrix3d J = Matrix3d::Zero();
  J(0, 0) = cos_el * cos_az;
  J(0, 1) = -range * cos_el * sin_az;
  J(0, 2) = -range * sin_el * cos_az;
  J(1, 0) = cos_el * sin_az;
  J(1, 1) = range * cos_el * cos_az;
  J(1, 2) = -range * sin_el * sin_az;
  J(2, 0) = sin_el;
  J(2, 1) = 0.0;
  J(2, 2) = range * cos_el;

  Matrix3d cartesian_covariance = J * measurement_covariance * J.transpose();

  // Create measurement matrix for position observation
  MatrixXd H = MatrixXd::Zero(3, getStateDimension());
  H.block<3, 3>(0, 0) = Matrix3d::Identity();

  // Update using converted measurement
  update(cartesian_measurement, cartesian_covariance, H);
}

} // namespace aimm_cs_ducmkf
