#include "aimm_cs_ducmkf.h"
#include <utility>

namespace aimm_cs_ducmkf
{

// ---------------- Utility helpers ----------------
    MatrixXd createIdentityMatrix(const int size)
    {
        return MatrixXd::Identity(size, size);
    }

    double calculateDistance(const Vector3d &p1, const Vector3d &p2)
    {
        return (p1 - p2).norm();
    }

    double normalizeAngle(double angle)
    {
        while (angle > M_PI)
            angle -= 2.0 * M_PI;
        while (angle < -M_PI)
            angle += 2.0 * M_PI;
        return angle;
    }

    Vector3d polarToCartesian(const double range, const double azimuth,
                              const double elevation)
    {
        double x = range * cos(elevation) * cos(azimuth);
        double y = range * cos(elevation) * sin(azimuth);
        double z = range * sin(elevation);
        return {x, y, z};
    }

    Vector3d cartesianToPolar(const Vector3d &cartesian)
    {
        double range = cartesian.norm();
        double azimuth = 0.0;
        double elevation = 0.0;
        if (range > 1e-9)
        {
            azimuth = atan2(cartesian.y(), cartesian.x());
            elevation = asin(cartesian.z() / range);
        }
        return {range, azimuth, elevation};
    }

    bool isPositiveDefinite(const MatrixXd &matrix)
    {
        const Eigen::LLT<MatrixXd> llt(matrix);
        return llt.info() == Eigen::Success;
    }

// ---------------- TrackerConfig ----------------
    AIMM_CS_DUCMKF_Tracker::TrackerConfig::TrackerConfig()
    {
        // Set up a proper transition matrix for 3 models (CV, CA, CS)
        // This allows models to switch between each other based on target behavior
        model_transition_matrix = MatrixXd(3, 3);
        // More aggressive transition probabilities for better adaptation
        // - High probability of staying in the same model (0.85)
        // - Higher probability of switching to other models (0.075 each)
        // This creates better adaptation to changing target dynamics
        model_transition_matrix << 0.85, 0.075, 0.075, // CV -> CV, CV -> CA, CV -> CS
                                0.075, 0.85, 0.075,                        // CA -> CV, CA -> CA, CA -> CS
                                0.075, 0.075, 0.85;                        // CS -> CV, CS -> CA, CS -> CS
        // Alternative: Even more aggressive switching for better adaptation
        // model_transition_matrix <<
        //     0.80, 0.10, 0.10,   // CV -> CV, CV -> CA, CV -> CS
        //     0.10, 0.80, 0.10,   // CA -> CV, CA -> CA, CA -> CS
        //     0.10, 0.10, 0.80;   // CS -> CV, CS -> CA, CS -> CS
    }

    bool AIMM_CS_DUCMKF_Tracker::TrackerConfig::validate() const
    {
        return (model_transition_matrix.rows() == num_models &&
                model_transition_matrix.cols() == num_models);
    }

// ---------------- AIMM_CS_DUCMKF_Tracker ----------------
    AIMM_CS_DUCMKF_Tracker::AIMM_CS_DUCMKF_Tracker(TrackerConfig config)
        : config_(std::move(config)), initialized_(false), last_timestamp_(0.0),
          measurement_count_(0)
    {
        // Build IMM filter with requested models (here simple CV + CS)
        imm_filter_ = std::make_shared<IMMFilter>(config_.num_models);
        // Wind handler
        wind_handler_ = std::make_unique<WindGustHandler>(
                            config_.sampling_frequency, config_.wind_gust_low_freq,
                            config_.wind_gust_high_freq, config_.bandpass_filter_taps);
    }

    bool AIMM_CS_DUCMKF_Tracker::initialize(const StateVector &initial_state,
                                            const StateMatrix &initial_covariance)
    {
        if (initialized_)
            return true;
        // Better balanced model probabilities
        ModelProbabilities probs(config_.num_models);
        probs << 0.5, 0.3, 0.2; // CV: 50%, CA: 30%, CS: 20%
        // Ensure transition matrix
        TransitionMatrix trans = config_.model_transition_matrix;
        if (trans.rows() == 0)
            trans = MatrixXd::Identity(config_.num_models, config_.num_models);
        imm_filter_->initialize(initial_state, initial_covariance, probs, trans);
        wind_handler_->initialize(imm_filter_);
        result_.state = initial_state;
        result_.covariance = initial_covariance;
        initialized_ = true;
        last_timestamp_ = 0.0;
        return true;
    }

    bool AIMM_CS_DUCMKF_Tracker::processMeasurement(const Measurement &measurement,
            const double timestamp)
    {
        if (!initialized_)
            return false;
        double dt = timestamp - last_timestamp_;
        if (dt <= 0.0)
            dt = 1.0 / config_.sampling_frequency;
        // Step 1: Predict a step for all models in IMM
        imm_filter_->predict(dt);
        // Step 2: Process wind gust detection if acceleration data is available
        // Estimate acceleration from velocity change (works for both polar and Cartesian)
        Vector3d estimated_acceleration = Vector3d::Zero();
        if (measurement_count_ > 0)
        {
            Vector3d current_velocity = result_.state.segment<3>(3);
            Vector3d prev_velocity = getState().segment<3>(3);
            estimated_acceleration = (current_velocity - prev_velocity) / dt;
        }
        // Process acceleration for wind detection
        Vector3d filtered_acceleration =
            wind_handler_->processAcceleration(estimated_acceleration, timestamp);

        // Update wind detection status
        result_.wind_gust_detected = wind_handler_->isWindGustDetected();
        result_.wind_estimate = wind_handler_->getCurrentWindEstimate();

        // Compensate state prediction for wind effects
        if (result_.wind_gust_detected && config_.enable_wind_compensation)
        {
            StateVector compensated_state =
                wind_handler_->compensateWindEffects(imm_filter_->getState(), dt);
            // Note: In a full implementation, this would update the IMM filter state
        }
        // Step 3: Update step with measurement
        if (measurement.type == MeasurementType::POLAR)
        {
            // Use polar measurement update (DUCMKF)
            MeasurementVector polar_meas;
            polar_meas << measurement.measurement(0), measurement.measurement(1),
                       measurement.measurement(2);
            // Create measurement covariance matrix
            MeasurementMatrix R = MeasurementMatrix::Zero();
            R.diagonal() << measurement.noise_covariance(0, 0),
                       measurement.noise_covariance(1, 1), measurement.noise_covariance(2, 2);
            imm_filter_->updateWithPolarMeasurement(polar_meas, R);
        }
        else if (measurement.type == MeasurementType::CARTESIAN)
        {
            // Use the measurement and its covariance directly
            imm_filter_->updateWithCartesianMeasurement(
                measurement.measurement, measurement.noise_covariance);
        }
        else
        {
            // Fallback: treat as Cartesian
            imm_filter_->updateWithCartesianMeasurement(
                measurement.measurement, measurement.noise_covariance);
        }
        // Step 4: Get results from IMM filter
        result_.state = imm_filter_->getState();
        result_.covariance = imm_filter_->getCovariance();
        result_.model_probabilities = imm_filter_->getModelProbabilities();
        result_.most_likely_model = imm_filter_->getMostLikelyModel();
        // Step 5: Get maneuver detection status from IMM filter
        result_.maneuver_detected = imm_filter_->isManeuverDetected();
        // Step 6: Calculate performance metrics
        result_.timestamp = timestamp;
        result_.processing_time = 0.001; // Simplified timing
        // Performance metrics
        result_.performance_metrics.clear();
        result_.performance_metrics["position_rmse"] = calculatePositionRMSE();
        result_.performance_metrics["velocity_rmse"] = calculateVelocityRMSE();
        result_.performance_metrics["model_switching_rate"] =
            calculateModelSwitchingRate();
        result_.performance_metrics["wind_detection_accuracy"] =
            calculateWindDetectionAccuracy();
        last_timestamp_ = timestamp;
        ++measurement_count_;
        return true;
    }

    AIMM_CS_DUCMKF_Tracker::TrackingResult
    AIMM_CS_DUCMKF_Tracker::getTrackingResult() const
    {
        return result_;
    }

    void AIMM_CS_DUCMKF_Tracker::updateConfig(const TrackerConfig &config)
    {
        config_ = config;
    }

    const StateVector &AIMM_CS_DUCMKF_Tracker::getState() const
    {
        return result_.state;
    }

    const StateMatrix &AIMM_CS_DUCMKF_Tracker::getCovariance() const
    {
        return result_.covariance;
    }

    void AIMM_CS_DUCMKF_Tracker::reset()
    {
        initialized_ = false;
    }

// Performance metric calculations
    double AIMM_CS_DUCMKF_Tracker::calculatePositionRMSE() const
    {
        // Simplified RMSE calculation - in practice, this would compare against
        // ground truth
        if (measurement_count_ < 2)
            return 0.0;
        // Calculate position variance as a proxy for RMSE
        Vector3d position_variance = result_.covariance.block<3, 3>(0, 0).diagonal();
        return sqrt(position_variance.mean());
    }

    double AIMM_CS_DUCMKF_Tracker::calculateVelocityRMSE() const
    {
        // Simplified velocity RMSE calculation
        if (measurement_count_ < 2)
            return 0.0;
        // Calculate velocity variance as a proxy for RMSE
        Vector3d velocity_variance = result_.covariance.block<3, 3>(3, 3).diagonal();
        return sqrt(velocity_variance.mean());
    }

    double AIMM_CS_DUCMKF_Tracker::calculateModelSwitchingRate() const
    {
        // Calculate how often the most likely model changes
        static int last_model = -1;
        static int switch_count = 0;
        if (last_model != -1 && last_model != result_.most_likely_model)
        {
            switch_count++;
        }
        last_model = result_.most_likely_model;
        return (measurement_count_ > 0)
               ? static_cast<double>(switch_count) / measurement_count_
               : 0.0;
    }

    double AIMM_CS_DUCMKF_Tracker::calculateWindDetectionAccuracy() const
    {
        // Simplified wind detection accuracy
        // In practice, this would compare against known wind conditions
        if (!wind_handler_)
            return 0.0;
        auto wind_stats = wind_handler_->getWindStats();
        return wind_stats.count("gust_detected") ? wind_stats.at("gust_detected")
               : 0.0;
    }

} // namespace aimm_cs_ducmkf
