#ifndef AIMM_CS_DUCMKF_IMM_FILTER_H
#define AIMM_CS_DUCMKF_IMM_FILTER_H

#include "current_statistical_model.h"
#include "ducmkf.h"
#include "types.h"
#include <map>
#include <memory>
#include <vector>

namespace aimm_cs_ducmkf {

/**
 * @brief Adaptive Interacting Multiple Model with Current Statistical model and
 * DUCMKF
 *
 * This class implements the AIMM-CS-DUCMKF algorithm combining:
 * - Adaptive Interactive Multiple Model (AIMM) for model switching
 * - Current Statistical (CS) model for maneuver adaptation
 * - Decorrelated Unbiased Conversion Measurement Kalman Filter (DUCMKF) for
 * nonlinear measurements
 */
class IMMFilter {
public:
  /**
   * @brief Constructor
   * @param num_models Number of models in the IMM (default: 3)
   */
  explicit IMMFilter(int num_models = 3);

  /**
   * @brief Destructor
   */
  ~IMMFilter() = default;

  /**
   * @brief Initialize the IMM filter
   * @param initial_state Initial state estimate
   * @param initial_covariance Initial state covariance
   * @param model_probabilities Initial model probabilities
   * @param transition_matrix Model transition probability matrix
   */
  void initialize(const StateVector &initial_state,
                  const StateMatrix &initial_covariance,
                  const ModelProbabilities &model_probabilities,
                  const TransitionMatrix &transition_matrix);

  /**
   * @brief Add a filter model to the IMM
   * @param filter Shared pointer to the filter
   * @param model_type Type of the model
   * @param initial_probability Initial probability of this model
   */
  void addFilterModel(std::shared_ptr<KalmanFilter> filter,
                      FilterModel model_type, double initial_probability);

  /**
   * @brief Prediction step for all models
   * @param dt Time step
   */
  void predict(double dt);

  /**
   * @brief Update step with polar measurements (uses DUCMKF)
   * @param measurement Polar measurement [range, azimuth, elevation]
   * @param measurement_covariance Measurement noise covariance
   */
  void
  updateWithPolarMeasurement(const MeasurementVector &measurement,
                             const MeasurementMatrix &measurement_covariance);

  /**
   * @brief Update step with Cartesian measurements
   * @param measurement Cartesian measurement
   * @param measurement_covariance Measurement noise covariance
   */
  void updateWithCartesianMeasurement(const ObservationVector &measurement,
                                      const Matrix3d &measurement_covariance);

  /**
   * @brief Get the mixed state estimate
   * @return Current mixed state estimate
   */
  const StateVector &getState() const { return mixed_state_; }

  /**
   * @brief Get the mixed covariance estimate
   * @return Current mixed covariance estimate
   */
  const StateMatrix &getCovariance() const { return mixed_covariance_; }

  /**
   * @brief Get the current model probabilities
   * @return Vector of model probabilities
   */
  const ModelProbabilities &getModelProbabilities() const {
    return model_probabilities_;
  }

  /**
   * @brief Get the most likely model index
   * @return Index of most probable model
   */
  int getMostLikelyModel() const;

  /**
   * @brief Get the likelihood of each model
   * @return Vector of model likelihoods
   */
  const VectorXd &getModelLikelihoods() const { return model_likelihoods_; }

  /**
   * @brief Get filter hypothesis for specific model
   * @param model_index Index of the model
   * @return Filter hypothesis
   */
  const FilterHypothesis &getHypothesis(int model_index) const;

  /**
   * @brief Get number of models
   * @return Number of models in IMM
   */
  int getNumModels() const { return num_models_; }

  /**
   * @brief Set the model transition probabilities
   * @param transition_matrix New transition probability matrix
   */
  void setTransitionProbabilities(const TransitionMatrix &transition_matrix);

  /**
   * @brief Enable/disable adaptive model switching
   * @param enable True to enable adaptive switching
   * @param adaptation_rate Rate of adaptation (0-1)
   */
  void setAdaptiveModelSwitching(bool enable, double adaptation_rate = 0.1);

  /**
   * @brief Get current maneuver detection status
   * @return True if maneuver is detected
   */
  bool isManeuverDetected() const;

  /**
   * @brief Get performance statistics
   * @return Map of performance metrics
   */
  std::map<std::string, double> getPerformanceStats() const;

protected:
  /**
   * @brief Perform model mixing step
   */
  void performMixing();

  /**
   * @brief Update model probabilities
   */
  void updateModelProbabilities();

  /**
   * @brief Compute mixed estimate
   */
  void computeMixedEstimate();

  /**
   * @brief Calculate mixing probabilities
   */
  void calculateMixingProbabilities();

  /**
   * @brief Adapt transition probabilities based on performance
   */
  void adaptTransitionProbabilities();

  /**
   * @brief Validate IMM consistency
   * @return True if IMM is consistent
   */
  bool validateConsistency() const;

private:
  // Filter configuration
  int num_models_;
  std::vector<std::shared_ptr<KalmanFilter>> filters_;
  std::vector<FilterModel> model_types_;
  std::vector<FilterHypothesis> hypotheses_;

  // IMM state
  StateVector mixed_state_;
  StateMatrix mixed_covariance_;
  ModelProbabilities model_probabilities_;
  VectorXd model_likelihoods_;
  TransitionMatrix transition_probabilities_;
  MatrixXd mixing_probabilities_;

  // Adaptive parameters
  bool adaptive_switching_;
  double adaptation_rate_;
  VectorXd performance_history_;

  // Statistics
  int update_count_;
  std::vector<double> nees_history_;

  // Internal state
  bool initialized_;

  /**
   * @brief Create default filter models
   */
  void createDefaultModels();

  /**
   * @brief Setup constant velocity model
   * @return Shared pointer to CV model
   */
  std::shared_ptr<KalmanFilter> createConstantVelocityModel();

  /**
   * @brief Setup constant acceleration model
   * @return Shared pointer to CA model
   */
  std::shared_ptr<KalmanFilter> createConstantAccelerationModel();

  /**
   * @brief Setup current statistical model
   * @return Shared pointer to CS model
   */
  std::shared_ptr<CurrentStatisticalModel> createCurrentStatisticalModel();

  /**
   * @brief Ensure probability vector is normalized
   * @param probabilities Probability vector to normalize
   */
  void normalizeProbabilities(VectorXd &probabilities);

  /**
   * @brief Check matrix dimensions for consistency
   * @return True if all matrices have consistent dimensions
   */
  bool checkDimensions() const;
};

} // namespace aimm_cs_ducmkf

#endif // AIMM_CS_DUCMKF_IMM_FILTER_H
