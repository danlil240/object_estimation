#ifndef AIMM_CS_DUCMKF_TYPES_H
#define AIMM_CS_DUCMKF_TYPES_H

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "logger.h"

namespace aimm_cs_ducmkf {

// Type definitions for Eigen matrices and vectors
using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;
using Matrix2d = Eigen::Matrix2d;
using Vector2d = Eigen::Vector2d;
using Matrix3d = Eigen::Matrix3d;
using Vector3d = Eigen::Vector3d;
using Vector4d = Eigen::Vector4d;
using Matrix4d = Eigen::Matrix4d;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

// State vector - dynamic size to support different models:
// - 6-state for CV: [x, y, z, vx, vy, vz] 
// - 9-state for CA: [x, y, z, vx, vy, vz, ax, ay, az]
using StateVector = VectorXd;
using StateMatrix = MatrixXd;

// Measurement vector [range, azimuth, elevation] for radar measurements
using MeasurementVector = Vector3d;
using MeasurementMatrix = Matrix3d;

// Observation in Cartesian coordinates [x, y, z]
using ObservationVector = Vector3d;

// Wind disturbance vector [wx, wy, wz]
using WindVector = Vector3d;

// Model probabilities vector
using ModelProbabilities = VectorXd;

// Transition probability matrix
using TransitionMatrix = MatrixXd;

// Filter model types
enum class FilterModel {
    CONSTANT_VELOCITY,
    CONSTANT_ACCELERATION,
    CURRENT_STATISTICAL,
    COORDINATED_TURN
};

// Measurement types
enum class MeasurementType {
    CARTESIAN,
    POLAR,
    RADAR
};

// Structure for filter hypothesis in IMM
struct FilterHypothesis {
    StateVector state;
    StateMatrix covariance;
    StateVector predicted_state;
    StateMatrix predicted_covariance;
    double probability;
    double likelihood;
    FilterModel model_type;

    FilterHypothesis() : probability(0.0), likelihood(0.0), model_type(FilterModel::CONSTANT_VELOCITY) {}
};

// Structure for measurement data
struct Measurement {
    MeasurementVector measurement;
    MeasurementMatrix noise_covariance;
    MeasurementType type;
    double timestamp;

    Measurement() : timestamp(0.0), type(MeasurementType::CARTESIAN) {}
};

// Structure for wind gust data
struct WindGust {
    WindVector velocity;
    double timestamp;
    double duration;

    WindGust() : timestamp(0.0), duration(0.0) {}
};

} // namespace aimm_cs_ducmkf

#endif // AIMM_CS_DUCMKF_TYPES_H
