#ifndef AIMM_CS_DUCMKF_KALMAN_FILTER_H
#define AIMM_CS_DUCMKF_KALMAN_FILTER_H

#include "types.h"

namespace aimm_cs_ducmkf {

/**
 * @brief Base Kalman Filter class
 * 
 * This class provides the basic Kalman filter functionality with predict and update steps.
 * It serves as the base class for more specialized filter implementations.
 */
class KalmanFilter {
public:
    /**
     * @brief Constructor
     * @param state_dim Dimension of state vector
     * @param measurement_dim Dimension of measurement vector
     */
    KalmanFilter(int state_dim, int measurement_dim);

    /**
     * @brief Virtual destructor
     */
    virtual ~KalmanFilter() = default;

    /**
     * @brief Initialize the filter with initial state and covariance
     * @param initial_state Initial state estimate
     * @param initial_covariance Initial state covariance
     */
    virtual void initialize(const StateVector& initial_state, 
                           const StateMatrix& initial_covariance);

    /**
     * @brief Prediction step
     * @param transition_matrix State transition matrix F
     * @param process_noise Process noise covariance Q
     * @param control_input Control input (optional)
     * @param control_matrix Control matrix B (optional)
     * @param dt Time step
     */
    // Convenience overload (no control input)
    void predict(const StateMatrix& transition_matrix,
                 const StateMatrix& process_noise);

    virtual void predict(const StateMatrix& transition_matrix,
                        const StateMatrix& process_noise,
                        const StateVector& control_input,
                        const StateMatrix& control_matrix,
                        double dt = 1.0);

    /**
     * @brief Update step with Cartesian measurements
     * @param measurement Measurement vector
     * @param measurement_covariance Measurement noise covariance
     * @param measurement_matrix Measurement matrix H
     */

    virtual void update(const VectorXd& measurement,
                       const MatrixXd& measurement_covariance,
                       const MatrixXd& measurement_matrix);

    /**
     * @brief Get current state estimate
     * @return Current state vector
     */
    const StateVector& getState() const { return state_; }

    /**
     * @brief Get current state covariance
     * @return Current state covariance matrix
     */
    const StateMatrix& getCovariance() const { return covariance_; }

    /**
     * @brief Get predicted state
     * @return Predicted state vector
     */
    const StateVector& getPredictedState() const { return predicted_state_; }

    /**
     * @brief Get predicted covariance
     * @return Predicted covariance matrix
     */
    const StateMatrix& getPredictedCovariance() const { return predicted_covariance_; }

    /**
     * @brief Get the last innovation (measurement residual)
     * @return Innovation vector
     */
    const VectorXd& getInnovation() const { return innovation_; }

    /**
     * @brief Get the innovation covariance
     * @return Innovation covariance matrix
     */
    const MatrixXd& getInnovationCovariance() const { return innovation_covariance_; }

    /**
     * @brief Get the Kalman gain from last update
     * @return Kalman gain matrix
     */
    const MatrixXd& getKalmanGain() const { return kalman_gain_; }

    /**
     * @brief Check if filter is initialized
     * @return True if initialized
     */
    bool isInitialized() const { return initialized_; }

    /**
     * @brief Get state dimension
     * @return State vector dimension
     */
    int getStateDimension() const { return state_dim_; }

    /**
     * @brief Get measurement dimension
     * @return Measurement vector dimension
     */
    int getMeasurementDimension() const { return measurement_dim_; }

    /**
     * @brief Calculate log-likelihood of measurement
     * @param measurement Measurement vector
     * @param measurement_covariance Measurement covariance
     * @param measurement_matrix Measurement matrix
     * @return Log-likelihood value
     */
    double calculateLogLikelihood(const VectorXd& measurement,
                                 const MatrixXd& measurement_covariance,
                                 const MatrixXd& measurement_matrix) const;

    /**
     * @brief Calculate Normalized Estimation Error Squared (NEES)
     * @return NEES value for filter consistency check
     */
    double calculateNEES(const StateVector& true_state) const;

protected:
    // Filter dimensions
    int state_dim_;
    int measurement_dim_;

    // State and covariance
    StateVector state_;
    StateMatrix covariance_;
    StateVector predicted_state_;
    StateMatrix predicted_covariance_;

    // Innovation and gain
    VectorXd innovation_;
    MatrixXd innovation_covariance_;
    MatrixXd kalman_gain_;

    // Filter status
    bool initialized_;

    /**
     * @brief Ensure matrix is positive definite
     * @param matrix Matrix to check and fix
     * @param min_eigenvalue Minimum eigenvalue to enforce
     */
    void ensurePositiveDefinite(Eigen::Ref<MatrixXd> matrix, double min_eigenvalue = 1e-9);
};

} // namespace aimm_cs_ducmkf

#endif // AIMM_CS_DUCMKF_KALMAN_FILTER_H
