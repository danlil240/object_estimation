#include "imm_filter.h"
#include "ducmkf.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace aimm_cs_ducmkf {
// Helper: build state transition matrix for simple kinematic models (CV, CA)
static Eigen::MatrixXd buildTransitionMatrix(FilterModel model, double dt,
                                             int state_dim = 6) {
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(state_dim, state_dim);

  switch (model) {
  case FilterModel::CONSTANT_VELOCITY:
    // 6-state model: [x, y, z, vx, vy, vz]
    // x_k+1 = x_k + vx_k * dt
    F(0, 3) = dt;
    F(1, 4) = dt;
    F(2, 5) = dt;
    break;

  case FilterModel::CONSTANT_ACCELERATION:
    if (state_dim == 9) {
      // 9-state model: [x, y, z, vx, vy, vz, ax, ay, az]
      // x_k+1 = x_k + vx_k * dt + 0.5 * ax_k * dt^2
      // vx_k+1 = vx_k + ax_k * dt
      // ax_k+1 = ax_k (constant acceleration assumption)

      double dt2 = dt * dt;

      // Position updates: x += vx*dt + 0.5*ax*dt^2
      F(0, 3) = dt;        // x += vx*dt
      F(1, 4) = dt;        // y += vy*dt
      F(2, 5) = dt;        // z += vz*dt
      F(0, 6) = 0.5 * dt2; // x += 0.5*ax*dt^2
      F(1, 7) = 0.5 * dt2; // y += 0.5*ay*dt^2
      F(2, 8) = 0.5 * dt2; // z += 0.5*az*dt^2

      // Velocity updates: vx += ax*dt
      F(3, 6) = dt; // vx += ax*dt
      F(4, 7) = dt; // vy += ay*dt
      F(5, 8) = dt; // vz += az*dt

      // Acceleration states remain constant (F(6,6) = F(7,7) = F(8,8) = 1,
      // already set by Identity)
    } else {
      // Fallback to 6-state with acceleration effects in process noise
      F(0, 3) = dt;
      F(1, 4) = dt;
      F(2, 5) = dt;
    }
    break;

  default:
    // For other (non-linear / CS) models, identity suffices – dedicated predict
    // implemented separately
    break;
  }
  return F;
}

// Helper: rough process-noise for simple models (tunable)
static Eigen::MatrixXd buildProcessNoise(FilterModel model, double dt,
                                         int state_dim = 6) {
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);

  double q_pos = 0.01; // position process noise (reduced for stability)
  double q_vel = 0.1;  // velocity process noise (increased for better tracking)
  double q_acc = 1.0;  // acceleration process noise (increased for CA model)

  if (model == FilterModel::CONSTANT_ACCELERATION && state_dim == 9) {
    // 9-state process noise matrix for constant acceleration model
    // Using continuous-time noise model discretized

    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    double dt4 = dt3 * dt;

    // Process noise for each axis (x, y, z) follows the same pattern
    for (int i = 0; i < 3; ++i) {
      int pos_idx = i;     // position indices: 0, 1, 2
      int vel_idx = i + 3; // velocity indices: 3, 4, 5
      int acc_idx = i + 6; // acceleration indices: 6, 7, 8

      // Continuous-time white noise acceleration model
      // Q matrix blocks for [pos, vel, acc] for each axis
      Q(pos_idx, pos_idx) = q_acc * dt4 / 4.0;
      Q(pos_idx, vel_idx) = q_acc * dt3 / 2.0;
      Q(pos_idx, acc_idx) = q_acc * dt2 / 2.0;

      Q(vel_idx, pos_idx) = q_acc * dt3 / 2.0;
      Q(vel_idx, vel_idx) = q_acc * dt2;
      Q(vel_idx, acc_idx) = q_acc * dt;

      Q(acc_idx, pos_idx) = q_acc * dt2 / 2.0;
      Q(acc_idx, vel_idx) = q_acc * dt;
      Q(acc_idx, acc_idx) = q_acc;
    }
  } else {
    // 6-state process noise for CV model or fallback CA model
    // Improved noise model for better tracking
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    double dt4 = dt3 * dt;

    for (int i = 0; i < 3; ++i) {
      int pos_idx = i;
      int vel_idx = i + 3;

      // Position noise includes velocity uncertainty
      Q(pos_idx, pos_idx) = q_pos + q_vel * dt2;
      Q(pos_idx, vel_idx) = q_vel * dt;
      Q(vel_idx, pos_idx) = q_vel * dt;
      Q(vel_idx, vel_idx) = q_vel;
    }
  }

  return Q;
}

IMMFilter::IMMFilter(int num_models)
    : num_models_(num_models), adaptive_switching_(false),
      adaptation_rate_(0.1), update_count_(0), initialized_(false) {

  if (num_models_ < 2) {
    throw std::invalid_argument("IMM requires at least 2 models");
  }

  // Initialize containers
  filters_.reserve(num_models_);
  model_types_.reserve(num_models_);
  hypotheses_.resize(num_models_);

  model_probabilities_.resize(num_models_);
  model_likelihoods_.resize(num_models_);
  mixing_probabilities_.resize(num_models_, num_models_);
  transition_probabilities_.resize(num_models_, num_models_);

  // Create default models if none specified
  createDefaultModels();
}

void IMMFilter::initialize(const StateVector &initial_state,
                           const StateMatrix &initial_covariance,
                           const ModelProbabilities &model_probabilities,
                           const TransitionMatrix &transition_matrix) {

  if (model_probabilities.size() != num_models_) {
    throw std::invalid_argument("Model probabilities size mismatch");
  }
  if (transition_matrix.rows() != num_models_ ||
      transition_matrix.cols() != num_models_) {
    throw std::invalid_argument("Transition matrix size mismatch");
  }

  // Initialize state estimates
  mixed_state_ = initial_state;
  mixed_covariance_ = initial_covariance;
  model_probabilities_ = model_probabilities;
  transition_probabilities_ = transition_matrix;

  // Normalize initial probabilities
  normalizeProbabilities(model_probabilities_);

  // Initialize all filter models
  for (int i = 0; i < num_models_; ++i) {
    int filter_state_dim = filters_[i]->getStateDimension();

    // Create appropriately sized state and covariance for this filter
    StateVector filter_initial_state;
    StateMatrix filter_initial_covariance;

    if (filter_state_dim == initial_state.size()) {
      // Same dimension - use directly
      filter_initial_state = initial_state;
      filter_initial_covariance = initial_covariance;
    } else if (filter_state_dim > initial_state.size()) {
      // Expand state vector (e.g., 6-state to 9-state for CA model)
      filter_initial_state = StateVector::Zero(filter_state_dim);
      filter_initial_covariance =
          StateMatrix::Identity(filter_state_dim, filter_state_dim) * 10.0;

      // Copy existing state components
      int copy_size = std::min<int>(initial_state.size(), filter_state_dim);
      filter_initial_state.head(copy_size) = initial_state.head(copy_size);

      // Copy existing covariance components
      int cov_copy_size =
          std::min<int>(initial_covariance.rows(), filter_state_dim);
      filter_initial_covariance.topLeftCorner(cov_copy_size, cov_copy_size) =
          initial_covariance.topLeftCorner(cov_copy_size, cov_copy_size);
    } else {
      // Reduce state vector (if needed)
      filter_initial_state = initial_state.head(filter_state_dim);
      filter_initial_covariance =
          initial_covariance.topLeftCorner(filter_state_dim, filter_state_dim);
    }

    filters_[i]->initialize(filter_initial_state, filter_initial_covariance);

    // Initialize hypothesis (use the original dimensions for mixing)
    hypotheses_[i].state = initial_state;
    hypotheses_[i].covariance = initial_covariance;
    hypotheses_[i].predicted_state = initial_state;
    hypotheses_[i].predicted_covariance = initial_covariance;
    hypotheses_[i].probability = model_probabilities_(i);
    hypotheses_[i].likelihood = 1.0;
    hypotheses_[i].model_type = model_types_[i];
  }

  initialized_ = true;
}

void IMMFilter::addFilterModel(std::shared_ptr<KalmanFilter> filter,
                               FilterModel model_type,
                               double initial_probability) {
  if (filters_.size() >= num_models_) {
    throw std::runtime_error(
        "Cannot add more models than specified in constructor");
  }

  filters_.push_back(filter);
  model_types_.push_back(model_type);

  // Update model probabilities if this is a new model
  if (filters_.size() <= model_probabilities_.size()) {
    model_probabilities_(filters_.size() - 1) = initial_probability;
  }
}

void IMMFilter::predict(double dt) {
  if (!initialized_) {
    throw std::runtime_error("IMM filter not initialized");
  }

  // Step 1: Model mixing
  performMixing();

  // Step 2: Model-specific predictions
  for (int i = 0; i < num_models_; ++i) {
    int filter_state_dim = filters_[i]->getStateDimension();
    StateVector filter_state;
    StateMatrix filter_covariance;

    // Always start from the mixed 6D state/covariance
    StateVector mixed_state = hypotheses_[i].state;
    StateMatrix mixed_cov = hypotheses_[i].covariance;
    if (filter_state_dim == 6) {
      filter_state = mixed_state;
      filter_covariance = mixed_cov;
    } else if (filter_state_dim == 9) {
      // Expand to 9D: [x, y, z, vx, vy, vz, ax, ay, az]
      filter_state = StateVector::Zero(9);
      filter_state.head(6) = mixed_state;
      // Set acceleration to zero
      filter_state.tail(3).setZero();
      filter_covariance = StateMatrix::Identity(9, 9) * 1e-2;
      filter_covariance.topLeftCorner(6, 6) = mixed_cov;
      // Lower-right 3x3 block: small value
      filter_covariance.bottomRightCorner(3, 3) = Matrix3d::Identity() * 1e-2;
    } else {
      // Fallback: use as much as possible
      int copy_size = std::min<int>(filter_state_dim, mixed_state.size());
      filter_state = StateVector::Zero(filter_state_dim);
      filter_state.head(copy_size) = mixed_state.head(copy_size);
      filter_covariance =
          StateMatrix::Identity(filter_state_dim, filter_state_dim) * 1e-2;
      filter_covariance.topLeftCorner(copy_size, copy_size) =
          mixed_cov.topLeftCorner(copy_size, copy_size);
    }

    // NaN/Inf check for state and covariance
    for (int k = 0; k < filter_state_dim; ++k) {
      if (std::isnan(filter_state(k)) || std::isinf(filter_state(k))) {
        filter_state(k) = 0.0;
      }
      for (int l = 0; l < filter_state_dim; ++l) {
        if (std::isnan(filter_covariance(k, l)) ||
            std::isinf(filter_covariance(k, l))) {
          filter_covariance(k, l) = (k == l) ? 1.0 : 0.0;
        }
      }
    }

    filters_[i]->initialize(filter_state, filter_covariance);

    // Perform prediction based on model type
    if (model_types_[i] == FilterModel::CURRENT_STATISTICAL) {
      auto cs_filter =
          std::dynamic_pointer_cast<CurrentStatisticalModel>(filters_[i]);
      if (cs_filter) {
        cs_filter->predict(dt);
      }
    } else {
      Eigen::MatrixXd F = buildTransitionMatrix(
          model_types_[i], dt, filters_[i]->getStateDimension());
      Eigen::MatrixXd Q = buildProcessNoise(model_types_[i], dt,
                                            filters_[i]->getStateDimension());
      int state_dim = filters_[i]->getStateDimension();
      StateVector zero_control = StateVector::Zero(state_dim);
      StateMatrix zero_control_matrix = StateMatrix::Zero(
          state_dim, state_dim); // Fixed: use zero matrix instead of identity
      filters_[i]->predict(F, Q, zero_control, zero_control_matrix,
                           dt); // Fixed: use dt instead of 1.0
    }

    // Store prediction results
    hypotheses_[i].predicted_state = filters_[i]->getPredictedState();
    hypotheses_[i].predicted_covariance = filters_[i]->getPredictedCovariance();

    // NaN/Inf check for predicted state and covariance
    for (int k = 0; k < hypotheses_[i].predicted_state.size(); ++k) {
      if (std::isnan(hypotheses_[i].predicted_state(k)) ||
          std::isinf(hypotheses_[i].predicted_state(k))) {
        hypotheses_[i].predicted_state(k) = 0.0;
      }
      for (int l = 0; l < hypotheses_[i].predicted_covariance.rows(); ++l) {
        if (std::isnan(hypotheses_[i].predicted_covariance(k, l)) ||
            std::isinf(hypotheses_[i].predicted_covariance(k, l))) {
          hypotheses_[i].predicted_covariance(k, l) = (k == l) ? 1.0 : 0.0;
        }
      }
    }
  }

  // Step 3: Compute mixed prediction
  computeMixedEstimate();
}

void IMMFilter::createDefaultModels() {
  // Create default models if none exist
  if (filters_.empty()) {
    // Create constant velocity model with DUCMKF
    auto cv_ducmkf = std::make_shared<DUCMKF>(
        6, 3); // 6 states (3D pos + vel), 3 measurements (polar)
    addFilterModel(cv_ducmkf, FilterModel::CONSTANT_VELOCITY, 0.6);

    // Create constant acceleration model with DUCMKF
    auto ca_ducmkf = std::make_shared<DUCMKF>(
        9, 3); // 9 states (3D pos + vel + acc), 3 measurements (polar)
    addFilterModel(ca_ducmkf, FilterModel::CONSTANT_ACCELERATION, 0.3);

    // Create current statistical model
    auto cs_filter = std::make_shared<CurrentStatisticalModel>(6, 3, 10, 3.0);
    addFilterModel(cs_filter, FilterModel::CURRENT_STATISTICAL, 0.1);
  }
}

void IMMFilter::normalizeProbabilities(VectorXd &probabilities) {
  double sum = probabilities.sum();
  if (sum > 0.0) {
    probabilities /= sum;
  } else {
    // If all probabilities are zero, set them to equal probabilities
    probabilities.setConstant(1.0 / probabilities.size());
  }
}

void IMMFilter::performMixing() {
  // Calculate mixing probabilities
  for (int i = 0; i < num_models_; ++i) {
    for (int j = 0; j < num_models_; ++j) {
      double denominator = 0.0;
      for (int k = 0; k < num_models_; ++k) {
        denominator +=
            transition_probabilities_(i, k) * model_probabilities_(k);
      }
      if (denominator > 1e-10) {
        mixing_probabilities_(i, j) =
            (transition_probabilities_(i, j) * model_probabilities_(j)) /
            denominator;
      } else {
        mixing_probabilities_(i, j) = 1.0 / num_models_;
      }
    }
  }

  // Always use 6 as the common dimension for mixing
  constexpr int common_dim = 6;
  StateVector mixed_state_common = StateVector::Zero(common_dim);
  StateMatrix mixed_covariance_common =
      StateMatrix::Zero(common_dim, common_dim);

  for (int i = 0; i < num_models_; ++i) {
    // Compute mixed state in common dimension
    mixed_state_common.setZero();
    for (int j = 0; j < num_models_; ++j) {
      VectorXd state_j = hypotheses_[j].state;
      if (state_j.size() > common_dim)
        state_j = state_j.head(common_dim);
      else if (state_j.size() < common_dim) {
        VectorXd tmp = VectorXd::Zero(common_dim);
        tmp.head(state_j.size()) = state_j;
        state_j = tmp;
      }
      mixed_state_common += mixing_probabilities_(i, j) * state_j;
    }

    // Compute mixed covariance in common dimension
    mixed_covariance_common.setZero();
    for (int j = 0; j < num_models_; ++j) {
      VectorXd state_j = hypotheses_[j].state;
      if (state_j.size() > common_dim)
        state_j = state_j.head(common_dim);
      else if (state_j.size() < common_dim) {
        VectorXd tmp = VectorXd::Zero(common_dim);
        tmp.head(state_j.size()) = state_j;
        state_j = tmp;
      }
      StateMatrix cov_j = hypotheses_[j].covariance;
      if (cov_j.rows() > common_dim || cov_j.cols() > common_dim)
        cov_j = cov_j.topLeftCorner(common_dim, common_dim);
      else if (cov_j.rows() < common_dim || cov_j.cols() < common_dim) {
        StateMatrix tmp = StateMatrix::Zero(common_dim, common_dim);
        tmp.topLeftCorner(cov_j.rows(), cov_j.cols()) = cov_j;
        cov_j = tmp;
      }
      VectorXd state_diff = state_j - mixed_state_common;
      mixed_covariance_common += mixing_probabilities_(i, j) *
                                 (cov_j + state_diff * state_diff.transpose());
    }

    // NaN/Inf check for state and covariance
    for (int k = 0; k < common_dim; ++k) {
      if (std::isnan(mixed_state_common(k)) ||
          std::isinf(mixed_state_common(k))) {
        mixed_state_common(k) = 0.0;
      }
      for (int l = 0; l < common_dim; ++l) {
        if (std::isnan(mixed_covariance_common(k, l)) ||
            std::isinf(mixed_covariance_common(k, l))) {
          mixed_covariance_common(k, l) = (k == l) ? 1.0 : 0.0;
        }
      }
    }

    // Update hypothesis with mixed values (keep in common dimension for
    // consistency)
    hypotheses_[i].state = mixed_state_common;
    hypotheses_[i].covariance = mixed_covariance_common;
  }
}

void IMMFilter::computeMixedEstimate() {
  // Always use 6 as the common dimension for output
  constexpr int common_dim = 6;
  mixed_state_ = StateVector::Zero(common_dim);
  mixed_covariance_ = StateMatrix::Zero(common_dim, common_dim);

  // Compute mixed state estimate
  for (int i = 0; i < num_models_; ++i) {
    StateVector pred = filters_[i]->getPredictedState();
    if (pred.size() > common_dim)
      pred = pred.head(common_dim);
    else if (pred.size() < common_dim) {
      StateVector tmp = StateVector::Zero(common_dim);
      tmp.head(pred.size()) = pred;
      pred = tmp;
    }
    mixed_state_ += model_probabilities_(i) * pred;
  }

  // Compute mixed covariance estimate
  for (int i = 0; i < num_models_; ++i) {
    StateVector pred = filters_[i]->getPredictedState();
    if (pred.size() > common_dim)
      pred = pred.head(common_dim);
    else if (pred.size() < common_dim) {
      StateVector tmp = StateVector::Zero(common_dim);
      tmp.head(pred.size()) = pred;
      pred = tmp;
    }
    StateMatrix cov = filters_[i]->getPredictedCovariance();
    if (cov.rows() > common_dim || cov.cols() > common_dim)
      cov = cov.topLeftCorner(common_dim, common_dim);
    else if (cov.rows() < common_dim || cov.cols() < common_dim) {
      StateMatrix tmp = StateMatrix::Zero(common_dim, common_dim);
      tmp.topLeftCorner(cov.rows(), cov.cols()) = cov;
      cov = tmp;
    }
    StateVector state_diff = pred - mixed_state_;
    mixed_covariance_ +=
        model_probabilities_(i) * (cov + state_diff * state_diff.transpose());
  }
}

void IMMFilter::updateWithPolarMeasurement(
    const MeasurementVector &measurement,
    const MeasurementMatrix &measurement_covariance) {
  // Calculate likelihoods for each filter
  model_likelihoods_.resize(num_models_);

  // Use DUCMKF for proper polar measurement handling
  for (int i = 0; i < num_models_; ++i) {
    // Check if this filter is a DUCMKF
    auto ducmkf_filter = std::dynamic_pointer_cast<DUCMKF>(filters_[i]);
    if (ducmkf_filter) {
      // Use DUCMKF's specialized polar update
      ducmkf_filter->updateWithPolarMeasurement(measurement,
                                                measurement_covariance);

      // Get updated state and covariance
      hypotheses_[i].state = ducmkf_filter->getState();
      hypotheses_[i].covariance = ducmkf_filter->getCovariance();

      // Calculate likelihood for this model
      Vector3d predicted_pos = hypotheses_[i].predicted_state.head<3>();
      Vector3d updated_pos = hypotheses_[i].state.head<3>();
      Vector3d innovation = updated_pos - predicted_pos;

      Matrix3d innovation_cov =
          hypotheses_[i].predicted_covariance.block<3, 3>(0, 0) +
          ducmkf_filter->getCovariance().block<3, 3>(0, 0);

      // Ensure positive definiteness
      Eigen::LLT<Matrix3d> llt(innovation_cov);
      if (llt.info() == Eigen::Success) {
        // Use solve method instead of inverse for better numerical stability
        Vector3d normalized_innovation = llt.solve(innovation);
        double mahalanobis = innovation.transpose() * normalized_innovation;
        model_likelihoods_(i) =
            exp(-0.5 * mahalanobis) /
            sqrt(pow(2 * M_PI, 3) * innovation_cov.determinant());
      } else {
        model_likelihoods_(i) = 1e-6; // Very small likelihood
      }

    } else {
      // Check if this is a Current Statistical Model
      auto cs_filter =
          std::dynamic_pointer_cast<CurrentStatisticalModel>(filters_[i]);
      if (cs_filter) {
        // Use CS model's polar update method
        cs_filter->updateWithPolarMeasurement(measurement,
                                              measurement_covariance);

        // Get updated state and covariance
        hypotheses_[i].state = cs_filter->getState();
        hypotheses_[i].covariance = cs_filter->getCovariance();

        // Calculate likelihood for CS model
        Vector3d predicted_pos = hypotheses_[i].predicted_state.head<3>();
        Vector3d updated_pos = hypotheses_[i].state.head<3>();
        Vector3d innovation = updated_pos - predicted_pos;

        Matrix3d innovation_cov =
            hypotheses_[i].predicted_covariance.block<3, 3>(0, 0) +
            cs_filter->getCovariance().block<3, 3>(0, 0);

        // Ensure positive definiteness
        Eigen::LLT<Matrix3d> llt(innovation_cov);
        if (llt.info() == Eigen::Success) {
          // Use solve method instead of inverse for better numerical stability
          Vector3d normalized_innovation = llt.solve(innovation);
          double mahalanobis = innovation.transpose() * normalized_innovation;
          model_likelihoods_(i) =
              exp(-0.5 * mahalanobis) /
              sqrt(pow(2 * M_PI, 3) * innovation_cov.determinant());
        } else {
          model_likelihoods_(i) = 1e-6; // Very small likelihood
        }

      } else {
        // Fallback to simplified conversion for other filters
        Vector3d cart_pos =
            Vector3d(measurement(0) * cos(measurement(2)) * cos(measurement(1)),
                     measurement(0) * cos(measurement(2)) * sin(measurement(1)),
                     measurement(0) * sin(measurement(2)));

        ObservationVector cart_meas = cart_pos;
        Matrix3d cart_cov = Matrix3d::Identity() * 1.0; // Simplified covariance

        // Update filter with Cartesian measurement
        MatrixXd H = MatrixXd::Zero(3, filters_[i]->getStateDimension());
        H.block<3, 3>(0, 0) = Matrix3d::Identity(); // Observe position only
        filters_[i]->update(cart_meas, cart_cov, H);
        hypotheses_[i].state = filters_[i]->getState();
        hypotheses_[i].covariance = filters_[i]->getCovariance();

        // Calculate likelihood
        Vector3d predicted_pos = hypotheses_[i].predicted_state.head<3>();
        Vector3d updated_pos = hypotheses_[i].state.head<3>();
        Vector3d innovation = updated_pos - predicted_pos;

        Matrix3d innovation_cov =
            hypotheses_[i].predicted_covariance.block<3, 3>(0, 0) + cart_cov;

        // Ensure positive definiteness
        Eigen::LLT<Matrix3d> llt(innovation_cov);
        if (llt.info() == Eigen::Success) {
          // Use solve method instead of inverse for better numerical stability
          Vector3d normalized_innovation = llt.solve(innovation);
          double mahalanobis = innovation.transpose() * normalized_innovation;
          model_likelihoods_(i) =
              exp(-0.5 * mahalanobis) /
              sqrt(pow(2 * M_PI, 3) * innovation_cov.determinant());
        } else {
          model_likelihoods_(i) = 1e-6; // Very small likelihood
        }
      }
    }
  }

  // Update model probabilities and compute mixed estimate
  updateModelProbabilities();
  computeMixedEstimate();
}

void IMMFilter::updateWithCartesianMeasurement(
    const ObservationVector &measurement,
    const Matrix3d &measurement_covariance) {
  
  // Update each individual filter with the Cartesian measurement
  for (int i = 0; i < num_models_; ++i) {
    // Create measurement matrix H for position observation
    MatrixXd H = MatrixXd::Zero(3, filters_[i]->getStateDimension());
    H.block<3, 3>(0, 0) = Matrix3d::Identity(); // Observe position only
    
    // Update the individual filter
    filters_[i]->update(measurement, measurement_covariance, H);
    
    // Store the updated state and covariance
    hypotheses_[i].state = filters_[i]->getState();
    hypotheses_[i].covariance = filters_[i]->getCovariance();
  }

  // Calculate likelihoods for each filter
  model_likelihoods_.resize(num_models_);

  for (int i = 0; i < num_models_; ++i) {
    // Get predicted state and covariance from hypothesis
    StateVector predicted_state = hypotheses_[i].predicted_state;
    StateMatrix predicted_cov = hypotheses_[i].predicted_covariance;
    StateVector updated_state = hypotheses_[i].state;

    // Extract position from state for measurement comparison
    ObservationVector predicted_obs = predicted_state.head<3>();
    ObservationVector updated_obs = updated_state.head<3>();
    ObservationVector innovation = measurement - predicted_obs;

    // Innovation covariance
    Matrix3d S = predicted_cov.topLeftCorner<3, 3>() + measurement_covariance;

    // Calculate likelihood (simplified Gaussian)
    double det_S = S.determinant();
    if (det_S > 1e-6) {
      double mahalanobis = innovation.transpose() * S.inverse() * innovation;
      model_likelihoods_(i) =
          exp(-0.5 * mahalanobis) / sqrt(pow(2 * M_PI, 3) * det_S);
    } else {
      model_likelihoods_(i) = 1e-6; // Very small likelihood
    }
  }

  // Update model probabilities
  updateModelProbabilities();

  // Compute mixed estimate
  computeMixedEstimate();
}

void IMMFilter::updateModelProbabilities() {
  // Update model probabilities based on likelihoods
  double total_likelihood = 0.0;

  // Ensure all likelihoods are positive and finite
  for (int i = 0; i < num_models_; ++i) {
    if (std::isnan(model_likelihoods_(i)) ||
        std::isinf(model_likelihoods_(i)) || model_likelihoods_(i) < 0) {
      model_likelihoods_(i) = 1e-6;
    }
  }

  for (int i = 0; i < num_models_; ++i) {
    model_probabilities_(i) = model_probabilities_(i) * model_likelihoods_(i);
    total_likelihood += model_probabilities_(i);
  }

  // Normalize probabilities
  if (total_likelihood > 1e-12) {
    model_probabilities_ /= total_likelihood;
  } else {
    // If all likelihoods are very small, reset to uniform distribution
    model_probabilities_.setConstant(1.0 / num_models_);
  }

  // Ensure probabilities sum to 1 and are within valid range
  normalizeProbabilities(model_probabilities_);
}

int IMMFilter::getMostLikelyModel() const {
  int max_index = 0;
  double max_prob = model_probabilities_(0);

  for (int i = 1; i < num_models_; ++i) {
    if (model_probabilities_(i) > max_prob) {
      max_prob = model_probabilities_(i);
      max_index = i;
    }
  }

  return max_index;
}

bool IMMFilter::isManeuverDetected() const {
  // Check if any of the Current Statistical models detect a maneuver
  // We'll use a weighted approach based on model probabilities
  double maneuver_probability = 0.0;

  for (int i = 0; i < num_models_; ++i) {
    // Check if this filter is a Current Statistical model
    auto cs_filter =
        std::dynamic_pointer_cast<CurrentStatisticalModel>(filters_[i]);
    if (cs_filter && cs_filter->isManeuverDetected()) {
      maneuver_probability += model_probabilities_(i);
    }
  }

  // Return true if the weighted maneuver probability exceeds threshold
  return maneuver_probability > 0.3; // Threshold for maneuver detection
}

} // namespace aimm_cs_ducmkf