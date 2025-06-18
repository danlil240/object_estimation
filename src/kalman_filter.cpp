#include "../include/kalman_filter.h"
#include <stdexcept>
#include <Eigen/Eigenvalues>
#include <cmath>

namespace aimm_cs_ducmkf {

KalmanFilter::KalmanFilter(int state_dim, int measurement_dim)
    : state_dim_(state_dim), measurement_dim_(measurement_dim), initialized_(false) {


    if (state_dim <= 0 || measurement_dim <= 0) {
        throw std::invalid_argument("State and measurement dimensions must be positive");
    }

    // Initialize matrices
    state_.setZero(state_dim_);
    covariance_.setIdentity(state_dim_, state_dim_);
    predicted_state_.setZero(state_dim_);
    predicted_covariance_.setIdentity(state_dim_, state_dim_);

    // Initialize innovation-related matrices
    innovation_.setZero(measurement_dim_);
    innovation_covariance_.setIdentity(measurement_dim_, measurement_dim_);
    kalman_gain_.setZero(state_dim_, measurement_dim_);

}

// Convenience overload: no control input
void KalmanFilter::predict(const StateMatrix& transition_matrix,
                           const StateMatrix& process_noise) {
    StateVector zero_control = StateVector::Zero(state_dim_);
    StateMatrix zero_control_matrix = StateMatrix::Zero(state_dim_, state_dim_);
    predict(transition_matrix, process_noise, zero_control, zero_control_matrix, 0.0);
}


void KalmanFilter::initialize(const StateVector& initial_state, 
                             const StateMatrix& initial_covariance) {
    if (initial_state.size() != state_dim_) {
        throw std::invalid_argument("Initial state dimension mismatch");
    }
    if (initial_covariance.rows() != state_dim_ || initial_covariance.cols() != state_dim_) {
        throw std::invalid_argument("Initial covariance dimension mismatch");
    }

    state_ = initial_state;
    covariance_ = initial_covariance;

    // Ensure covariance is positive definite
    ensurePositiveDefinite(covariance_);

    initialized_ = true;
}

void KalmanFilter::predict(const StateMatrix& transition_matrix,
                          const StateMatrix& process_noise,
                          const StateVector& control_input,
                          const StateMatrix& control_matrix,
                          double dt) {
    if (!initialized_) {
        throw std::runtime_error("Filter not initialized");
    }

    if (transition_matrix.rows() != state_dim_ || transition_matrix.cols() != state_dim_) {
        throw std::invalid_argument("Transition matrix dimension mismatch");
    }
    if (process_noise.rows() != state_dim_ || process_noise.cols() != state_dim_) {
        throw std::invalid_argument("Process noise dimension mismatch");
    }

    // State prediction: x_k+1|k = F * x_k|k + B * u_k
    predicted_state_ = transition_matrix * state_;
    if (control_input.size() > 0 && control_matrix.rows() > 0) {
        predicted_state_ += control_matrix * control_input;
    }

    // Covariance prediction: P_k+1|k = F * P_k|k * F^T + Q
    predicted_covariance_ = transition_matrix * covariance_ * transition_matrix.transpose() + process_noise;

    // Ensure positive definiteness
    ensurePositiveDefinite(predicted_covariance_);
}

void KalmanFilter::update(const VectorXd& measurement,
                         const MatrixXd& measurement_covariance,
                         const MatrixXd& measurement_matrix) {
    if (!initialized_) {
        throw std::runtime_error("Filter not initialized");
    }

    if (measurement.size() != measurement_dim_) {
        throw std::invalid_argument("Measurement dimension mismatch");
    }
    if (measurement_matrix.rows() != measurement_dim_ || measurement_matrix.cols() != state_dim_) {
        throw std::invalid_argument("Measurement matrix dimension mismatch");
    }
    if (measurement_covariance.rows() != measurement_dim_ || measurement_covariance.cols() != measurement_dim_) {
        throw std::invalid_argument("Measurement covariance dimension mismatch");
    }

    // Check for NaN/Inf in inputs
    for (int i = 0; i < measurement.size(); ++i) {
        if (std::isnan(measurement(i)) || std::isinf(measurement(i))) {
            throw std::invalid_argument("Measurement contains NaN or Inf values");
        }
    }

    // Innovation: y = z - H * x_k+1|k
    VectorXd predicted_measurement = measurement_matrix * predicted_state_;
    innovation_ = measurement - predicted_measurement;

    // Check for NaN/Inf in innovation
    for (int i = 0; i < innovation_.size(); ++i) {
        if (std::isnan(innovation_(i)) || std::isinf(innovation_(i))) {
            throw std::invalid_argument("Innovation contains NaN or Inf values");
        }
    }

    // Innovation covariance: S = H * P_k+1|k * H^T + R
    innovation_covariance_ = measurement_matrix * predicted_covariance_ * measurement_matrix.transpose() + measurement_covariance;

    // Ensure innovation covariance is invertible
    ensurePositiveDefinite(innovation_covariance_);

    // Use Cholesky decomposition for better numerical stability
    Eigen::LLT<MatrixXd> llt(innovation_covariance_);
    if (llt.info() != Eigen::Success) {
        // Add small diagonal elements to make it positive definite
        innovation_covariance_ += MatrixXd::Identity(measurement_dim_, measurement_dim_) * 1e-6;
        llt = Eigen::LLT<MatrixXd>(innovation_covariance_);
    }

    // Kalman gain: K = P_k+1|k * H^T * S^(-1)
    // Use solve method instead of inverse for better numerical stability
    kalman_gain_ = predicted_covariance_ * measurement_matrix.transpose() * llt.solve(MatrixXd::Identity(measurement_dim_, measurement_dim_));

    // Check for NaN/Inf in Kalman gain
    for (int i = 0; i < kalman_gain_.rows(); ++i) {
        for (int j = 0; j < kalman_gain_.cols(); ++j) {
            if (std::isnan(kalman_gain_(i, j)) || std::isinf(kalman_gain_(i, j))) {
                kalman_gain_(i, j) = 0.0;
            }
        }
    }

    // State update: x_k+1|k+1 = x_k+1|k + K * y
    state_ = predicted_state_ + kalman_gain_ * innovation_;

    // Check for NaN/Inf in updated state
    for (int i = 0; i < state_.size(); ++i) {
        if (std::isnan(state_(i)) || std::isinf(state_(i))) {
            state_(i) = predicted_state_(i);  // Fallback to predicted state
        }
    }

    // Covariance update: P_k+1|k+1 = (I - K * H) * P_k+1|k * (I - K * H)^T + K * R * K^T
    // Using Joseph form for numerical stability
    MatrixXd I_KH = MatrixXd::Identity(state_dim_, state_dim_) - kalman_gain_ * measurement_matrix;
    covariance_ = I_KH * predicted_covariance_ * I_KH.transpose() + 
                  kalman_gain_ * measurement_covariance * kalman_gain_.transpose();

    // Ensure positive definiteness
    ensurePositiveDefinite(covariance_);
}

double KalmanFilter::calculateLogLikelihood(const VectorXd& measurement,
                                           const MatrixXd& measurement_covariance,
                                           const MatrixXd& measurement_matrix) const {
    if (!initialized_) {
        return -std::numeric_limits<double>::infinity();
    }

    // Calculate innovation
    VectorXd predicted_measurement = measurement_matrix * predicted_state_;
    VectorXd innovation = measurement - predicted_measurement;

    // Calculate innovation covariance
    MatrixXd S = measurement_matrix * predicted_covariance_ * measurement_matrix.transpose() + measurement_covariance;

    // Log-likelihood calculation
    double det_S = S.determinant();
    if (det_S <= 0) {
        return -std::numeric_limits<double>::infinity();
    }

    double mahalanobis = innovation.transpose() * S.inverse() * innovation;
    double log_likelihood = -0.5 * (measurement_dim_ * log(2.0 * M_PI) + log(det_S) + mahalanobis);

    return log_likelihood;
}

double KalmanFilter::calculateNEES(const StateVector& true_state) const {
    if (!initialized_) {
        return std::numeric_limits<double>::infinity();
    }

    StateVector error = state_ - true_state;
    double nees = error.transpose() * covariance_.inverse() * error;
    return nees / state_dim_;  // Normalized by state dimension
}

void KalmanFilter::ensurePositiveDefinite(Eigen::Ref<MatrixXd> matrix, double min_eigenvalue) {
    // Eigenvalue decomposition
    Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(matrix);
    if (eigensolver.info() != Eigen::Success) {
        // If eigenvalue decomposition fails, add small diagonal
        matrix += min_eigenvalue * MatrixXd::Identity(matrix.rows(), matrix.cols());
        return;
    }

    VectorXd eigenvalues = eigensolver.eigenvalues();
    MatrixXd eigenvectors = eigensolver.eigenvectors();

    // Clamp negative eigenvalues
    bool modified = false;
    for (int i = 0; i < eigenvalues.size(); ++i) {
        if (eigenvalues(i) < min_eigenvalue) {
            eigenvalues(i) = min_eigenvalue;
            modified = true;
        }
    }

    // Reconstruct matrix if modified
    if (modified) {
        matrix = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
    }
}

} // namespace aimm_cs_ducmkf
