#ifndef AIMM_CS_DUCMKF_H
#define AIMM_CS_DUCMKF_H

/**
 * @file aimm_cs_ducmkf.h
 * @brief Main header file for AIMM-CS-DUCMKF library
 *
 * This library implements the Adaptive Interactive Multiple Model with Current
 * Statistical model and Decorrelated Unbiased Conversion Measurement Kalman
 * Filter (AIMM-CS-DUCMKF) for robust object tracking in the presence of wind
 * gusts and maneuvers.
 *
 * Key Features:
 * - Adaptive Interactive Multiple Model (AIMM) for handling multiple motion
 * models
 * - Current Statistical (CS) model for adaptive maneuver tracking
 * - Decorrelated Unbiased Conversion Measurement Kalman Filter (DUCMKF) for
 * nonlinear measurements
 * - Bandpass filtering for wind gust detection and mitigation
 * - Wind disturbance estimation and compensation
 *
 * @author AIMM-CS-DUCMKF Development Team
 * @version 1.0.0
 * @date 2025
 */

// Include all library components
#include "bandpass_filter.h"
#include "current_statistical_model.h"
#include "ducmkf.h"
#include "imm_filter.h"
#include "kalman_filter.h"
#include "types.h"
#include "wind_gust_handler.h"

namespace aimm_cs_ducmkf {

/**
 * @brief Main AIMM-CS-DUCMKF Tracker Class
 *
 * This class provides a high-level interface to the complete AIMM-CS-DUCMKF
 * tracking system, integrating all components for robust object tracking.
 */
class AIMM_CS_DUCMKF_Tracker {
public:
  /**
   * @brief Configuration structure for AIMM-CS-DUCMKF tracker
   */
  struct TrackerConfig {
    // IMM parameters
    int num_models = 3;
    std::vector<double> initial_model_probabilities = {0.6, 0.3, 0.1};
    MatrixXd model_transition_matrix;

    // Wind handling parameters
    double sampling_frequency = 10.0; // Hz
    double wind_gust_low_freq = 0.1;  // Hz
    double wind_gust_high_freq = 2.0; // Hz
    int bandpass_filter_taps = 101;
    double wind_detection_threshold = 3.0;
    bool enable_wind_compensation = true;

    // Process noise parameters
    double position_noise = 0.1;     // m^2
    double velocity_noise = 0.01;    // (m/s)^2
    double acceleration_noise = 1.0; // (m/s^2)^2

    // Measurement noise parameters
    double range_noise = 1.0;       // m^2
    double azimuth_noise = 0.001;   // rad^2
    double elevation_noise = 0.001; // rad^2

    // CS model parameters
    int cs_window_size = 10;
    double cs_correlation_time = 5.0; // seconds
    double maneuver_threshold = 3.0;

    // DUCMKF parameters
    bool enable_second_order_bias = true;
    double max_range = 100000.0; // meters
    double min_range = 1.0;      // meters

    /**
     * @brief Default constructor with sensible defaults
     */
    TrackerConfig();

    /**
     * @brief Validate configuration parameters
     * @return True if configuration is valid
     */
    bool validate() const;
  };

  struct TrackingResult {
    StateVector state;
    StateMatrix covariance;
    ModelProbabilities model_probabilities;
    int most_likely_model;
    bool maneuver_detected;
    bool wind_gust_detected;
    WindVector wind_estimate;
    double timestamp;
    double processing_time;
    std::map<std::string, double> performance_metrics;

    TrackingResult()
        : most_likely_model(0), maneuver_detected(false),
          wind_gust_detected(false), timestamp(0.0), processing_time(0.0) {}
  };
  /**
   * @brief Constructor
   * @param config Configuration parameters for the tracker
   */
  explicit AIMM_CS_DUCMKF_Tracker(TrackerConfig config = TrackerConfig());

  /**
   * @brief Destructor
   */
  ~AIMM_CS_DUCMKF_Tracker() = default;

  /**
   * @brief Initialize the tracker
   * @param initial_state Initial state estimate [x,y,z,vx,vy,vz]
   * @param initial_covariance Initial state covariance matrix
   * @return True if initialization successful
   */
  bool initialize(const StateVector &initial_state,
                  const StateMatrix &initial_covariance);

  /**
   * @brief Process a new measurement
   * @param measurement Measurement data (polar or Cartesian)
   * @param timestamp Measurement timestamp
   * @return True if measurement processed successfully
   */
  bool processMeasurement(const Measurement &measurement, double timestamp);

  /**
   * @brief Get current state estimate
   * @return Current state vector [x,y,z,vx,vy,vz]
   */
  const StateVector &getState() const;

  /**
   * @brief Get current covariance estimate
   * @return Current state covariance matrix
   */
  const StateMatrix &getCovariance() const;

  /**
   * @brief Get tracking results with extended information
   * @return Complete tracking result structure
   */
  TrackingResult getTrackingResult() const;

  /**
   * @brief Check if tracker is initialized
   * @return True if initialized
   */
  bool isInitialized() const { return initialized_; }

  /**
   * @brief Reset the tracker
   */
  void reset();

  /**
   * @brief Get configuration
   * @return Current tracker configuration
   */
  const TrackerConfig &getConfig() const { return config_; }

  /**
   * @brief Update configuration
   * @param config New configuration
   */
  void updateConfig(const TrackerConfig &config);

  /**
   * @brief Update performance metrics
   * @param true_state True state vector
   * @param timestamp Timestamp
   */
  void updatePerformanceMetrics(const StateVector &true_state,
                                const double timestamp);

private:
  // Core components
  std::shared_ptr<IMMFilter> imm_filter_;
  std::unique_ptr<WindGustHandler> wind_handler_;

  // Configuration and state
  TrackerConfig config_;
  bool initialized_;
  double last_timestamp_;

  // Statistics
  int measurement_count_;
  std::vector<double> processing_times_;

  // Latest tracking result
  TrackingResult result_;

  // Performance metric calculations
  double calculatePositionRMSE() const;
  double calculateVelocityRMSE() const;
  double calculateModelSwitchingRate() const;
  double calculateWindDetectionAccuracy() const;
};

/**
 * @brief Tracking result structure
 */

// Utility functions

/**
 * @brief Convert polar measurement to Cartesian
 * @param range Range in meters
 * @param azimuth Azimuth in radians
 * @param elevation Elevation in radians
 * @return Cartesian coordinates [x, y, z]
 */
Vector3d polarToCartesian(double range, double azimuth, double elevation);

/**
 * @brief Convert Cartesian coordinates to polar
 * @param cartesian Cartesian coordinates [x, y, z]
 * @return Polar coordinates [range, azimuth, elevation]
 */
Vector3d cartesianToPolar(const Vector3d &cartesian);

/**
 * @brief Calculate distance between two 3D points
 * @param p1 First point
 * @param p2 Second point
 * @return Euclidean distance
 */
double calculateDistance(const Vector3d &p1, const Vector3d &p2);

/**
 * @brief Normalize angle to [-pi, pi] range
 * @param angle Angle in radians
 * @return Normalized angle
 */
double normalizeAngle(double angle);

/**
 * @brief Create identity matrix of given size
 * @param size Matrix size
 * @return Identity matrix
 */
MatrixXd createIdentityMatrix(int size);

/**
 * @brief Check if matrix is positive definite
 * @param matrix Matrix to check
 * @return True if positive definite
 */
bool isPositiveDefinite(const MatrixXd &matrix);

} // namespace aimm_cs_ducmkf

#endif // AIMM_CS_DUCMKF_H
