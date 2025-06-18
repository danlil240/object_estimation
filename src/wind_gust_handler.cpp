#include "wind_gust_handler.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace aimm_cs_ducmkf {

WindGustHandler::WindGustHandler(double sampling_freq, double gust_low_freq, 
                                double gust_high_freq, int filter_taps)
    : sampling_freq_(sampling_freq), gust_low_freq_(gust_low_freq), 
      gust_high_freq_(gust_high_freq), gust_threshold_(3.0), max_wind_speed_(50.0),
      gust_detected_(false), gust_magnitude_(0.0), gust_start_time_(0.0), 
      gust_duration_(0.0), wind_compensation_enabled_(true), adaptive_threshold_(true),
      initialized_(false), buffer_size_(100), detection_rate_(0.0), 
      false_positive_count_(0), true_positive_count_(0) {

    // Create bandpass filter for 3D wind detection
    bandpass_filter_ = std::make_unique<MultiChannelBandpassFilter>(
        sampling_freq_, gust_low_freq_, gust_high_freq_, 3, filter_taps);

    // Initialize state variables
    current_wind_estimate_.setZero();
    filtered_wind_.setZero();
    gust_direction_.setZero();
    mean_acceleration_.setZero();
    acceleration_variance_.setOnes();

    // Reserve buffer space
    acceleration_history_.resize(buffer_size_);
    filtered_acceleration_history_.resize(buffer_size_);
    timestamp_history_.resize(buffer_size_);
    wind_history_.resize(buffer_size_);
}

void WindGustHandler::initialize(std::shared_ptr<IMMFilter> imm_filter) {
    imm_filter_ = imm_filter;
    initialized_ = true;
}

Vector3d WindGustHandler::processAcceleration(const Vector3d& acceleration, double timestamp) {
    if (!initialized_) {
        throw std::runtime_error("Wind gust handler not initialized");
    }

    // Add to history
    addToHistory(acceleration, timestamp);

    // Apply bandpass filtering to detect wind gusts
    VectorXd accel_vector(3);
    accel_vector << acceleration(0), acceleration(1), acceleration(2);

    VectorXd filtered_accel = bandpass_filter_->filter(accel_vector);
    Vector3d filtered_acceleration(filtered_accel(0), filtered_accel(1), filtered_accel(2));

    // Store filtered acceleration
    filtered_acceleration_history_.push_back(filtered_acceleration);
    if (filtered_acceleration_history_.size() > buffer_size_) {
        filtered_acceleration_history_.pop_front();
    }

    // Update wind statistics
    updateWindStatistics(acceleration, timestamp);

    // Detect wind gust
    bool gust_detected = detectWindGust(acceleration, timestamp);

    // Update wind model if gust detected
    if (gust_detected) {
        updateWindModel();
    }

    // Return compensated acceleration only if wind compensation is enabled AND gust is detected
    // AND the wind estimate is reasonable
    if (wind_compensation_enabled_ && gust_detected_ && validateWindEstimate()) {
        return acceleration - current_wind_estimate_;
    }

    return acceleration;
}

bool WindGustHandler::detectWindGust(const Vector3d& acceleration, double timestamp) {
    if (filtered_acceleration_history_.size() < 3) {
        return false;
    }

    // Get current filtered acceleration
    Vector3d filtered_accel = filtered_acceleration_history_.back();
    double filtered_magnitude = filtered_accel.norm();

    // Calculate adaptive threshold
    double threshold = calculateAdaptiveThreshold();

    // Primary detection: magnitude exceeds threshold
    bool magnitude_detection = (filtered_magnitude > threshold);

    // Secondary detection: pattern analysis
    double pattern_confidence = checkGustPattern();
    bool pattern_detection = (pattern_confidence > 0.6);

    // Combined detection
    bool current_detection = magnitude_detection || pattern_detection;

    // Update gust state
    if (current_detection && !gust_detected_) {
        // Start of new gust
        gust_detected_ = true;
        gust_start_time_ = timestamp;
        gust_magnitude_ = filtered_magnitude;
        gust_direction_ = estimateWindDirection(filtered_accel);
        true_positive_count_++;
    } else if (!current_detection && gust_detected_) {
        // End of gust
        gust_detected_ = false;
        gust_duration_ = timestamp - gust_start_time_;
    } else if (current_detection && gust_detected_) {
        // Continuing gust - update magnitude and direction
        gust_magnitude_ = std::max(gust_magnitude_, filtered_magnitude);
        Vector3d new_direction = estimateWindDirection(filtered_accel);
        if (new_direction.norm() > 0.1) {
            gust_direction_ = 0.8 * gust_direction_ + 0.2 * new_direction;
            gust_direction_.normalize();
        }
    }

    return gust_detected_;
}

WindVector WindGustHandler::estimateWindDisturbance(const StateVector& state, 
                                                   const Vector3d& measurement) {
    WindVector wind_disturbance = WindVector::Zero();

    if (!gust_detected_ || filtered_acceleration_history_.size() < 3) {
        return wind_disturbance;
    }

    // Estimate wind disturbance from filtered acceleration
    Vector3d filtered_accel = filtered_acceleration_history_.back();

    // Convert acceleration to wind velocity estimate
    // Assuming wind affects acceleration linearly
    double time_constant = 2.0;  // seconds
    wind_disturbance = filtered_accel * time_constant;

    // Apply direction weighting
    if (gust_direction_.norm() > 0.1) {
        double directional_component = wind_disturbance.dot(gust_direction_);
        wind_disturbance = gust_direction_ * directional_component;
    }

    // Limit wind speed
    double wind_speed = wind_disturbance.norm();
    if (wind_speed > max_wind_speed_) {
        wind_disturbance = wind_disturbance * (max_wind_speed_ / wind_speed);
    }

    current_wind_estimate_ = wind_disturbance;

    // Add to history
    wind_history_.push_back(current_wind_estimate_);
    if (wind_history_.size() > buffer_size_) {
        wind_history_.pop_front();
    }

    return wind_disturbance;
}

StateVector WindGustHandler::compensateWindEffects(const StateVector& predicted_state, double dt) {
    if (!wind_compensation_enabled_ || !gust_detected_) {
        return predicted_state;
    }

    StateVector compensated_state = predicted_state;

    // Apply wind compensation to velocity components
    compensated_state.segment<3>(3) += current_wind_estimate_ * dt;

    // Apply second-order correction to position
    compensated_state.head<3>() += 0.5 * current_wind_estimate_ * dt * dt;

    return compensated_state;
}

void WindGustHandler::updateWindStatistics(const Vector3d& acceleration, double timestamp) {
    // Update mean acceleration
    if (acceleration_history_.size() > 0) {
        double alpha = 0.95;  // Exponential smoothing factor
        mean_acceleration_ = alpha * mean_acceleration_ + (1.0 - alpha) * acceleration;
    } else {
        mean_acceleration_ = acceleration;
    }

    // Update acceleration variance
    if (acceleration_history_.size() > 1) {
        Vector3d diff = acceleration - mean_acceleration_;
        acceleration_variance_ = 0.95 * acceleration_variance_ + 0.05 * diff.cwiseProduct(diff);
    }

    // Calculate noise statistics periodically
    if (acceleration_history_.size() % 10 == 0) {
        calculateNoiseStatistics();
    }
}

double WindGustHandler::calculateAdaptiveThreshold() const {
    if (!adaptive_threshold_ || acceleration_variance_.norm() < 1e-6) {
        return gust_threshold_;
    }

    // Adaptive threshold based on noise statistics
    double noise_level = sqrt(acceleration_variance_.norm());
    
    // Use a conservative threshold that scales with noise
    double adaptive_threshold = gust_threshold_ * noise_level;
    
    // Bound the threshold to reasonable values
    return std::max(0.5, std::min(10.0, adaptive_threshold));
}

double WindGustHandler::estimateWindPersistence() const {
    if (!gust_detected_) {
        return 0.0;
    }

    // Estimate persistence based on gust duration and magnitude
    double persistence = std::min(1.0, gust_duration_ / 5.0);  // 5 second max
    persistence *= std::min(1.0, gust_magnitude_ / 5.0);       // magnitude factor

    return persistence;
}

void WindGustHandler::updateWindModel() {
    // Update wind model parameters based on recent observations
    if (wind_history_.size() < 5) {
        return;
    }

    // Calculate wind persistence and direction stability
    double persistence = estimateWindPersistence();

    // Update detection rate
    int recent_detections = 0;
    int recent_samples = std::min(20, (int)acceleration_history_.size());

    for (int i = acceleration_history_.size() - recent_samples; i < acceleration_history_.size(); ++i) {
        if (i >= 0 && filtered_acceleration_history_[i].norm() > calculateAdaptiveThreshold()) {
            recent_detections++;
        }
    }

    detection_rate_ = (double)recent_detections / recent_samples;
}

void WindGustHandler::addToHistory(const Vector3d& acceleration, double timestamp) {
    acceleration_history_.push_back(acceleration);
    timestamp_history_.push_back(timestamp);

    // Maintain buffer size
    if (acceleration_history_.size() > buffer_size_) {
        acceleration_history_.pop_front();
        timestamp_history_.pop_front();
    }
}

void WindGustHandler::calculateNoiseStatistics() {
    if (acceleration_history_.size() < 10) {
        return;
    }

    // Calculate statistics from unfiltered acceleration
    Vector3d sum = Vector3d::Zero();
    for (const auto& accel : acceleration_history_) {
        sum += accel;
    }
    Vector3d mean = sum / acceleration_history_.size();

    Vector3d variance_sum = Vector3d::Zero();
    for (const auto& accel : acceleration_history_) {
        Vector3d diff = accel - mean;
        variance_sum += diff.cwiseProduct(diff);
    }

    acceleration_variance_ = variance_sum / (acceleration_history_.size() - 1);
}

Vector3d WindGustHandler::applyMedianFilter(const std::deque<Vector3d>& data, int window_size) const {
    if (data.size() < window_size) {
        return data.empty() ? Vector3d::Zero() : data.back();
    }

    Vector3d result = Vector3d::Zero();
    for (int dim = 0; dim < 3; ++dim) {
        std::vector<double> values;
        int start = std::max(0, (int)data.size() - window_size);

        for (int i = start; i < data.size(); ++i) {
            values.push_back(data[i](dim));
        }

        std::sort(values.begin(), values.end());
        int median_idx = values.size() / 2;
        result(dim) = values[median_idx];
    }

    return result;
}

Vector3d WindGustHandler::estimateWindDirection(const Vector3d& filtered_accel) const {
    if (filtered_accel.norm() < 1e-6) {
        return Vector3d::Zero();
    }

    return filtered_accel.normalized();
}

double WindGustHandler::checkGustPattern() const {
    if (filtered_acceleration_history_.size() < 5) {
        return 0.0;
    }

    // Look for characteristic gust pattern: sudden increase followed by decrease
    int n = filtered_acceleration_history_.size();
    Vector3d current = filtered_acceleration_history_[n-1];
    Vector3d prev1 = filtered_acceleration_history_[n-2];
    Vector3d prev2 = filtered_acceleration_history_[n-3];

    double current_mag = current.norm();
    double prev1_mag = prev1.norm();
    double prev2_mag = prev2.norm();

    // Check for sudden increase
    double increase_factor = (prev1_mag > 1e-6) ? (current_mag / prev1_mag) : 1.0;
    double pattern_confidence = 0.0;

    if (increase_factor > 2.0) {
        pattern_confidence += 0.5;
    }

    // Check for sustained elevation
    if (current_mag > 2.0 * mean_acceleration_.norm()) {
        pattern_confidence += 0.3;
    }

    // Check for directional consistency
    if (current.norm() > 1e-6 && prev1.norm() > 1e-6) {
        double directional_similarity = current.normalized().dot(prev1.normalized());
        if (directional_similarity > 0.7) {
            pattern_confidence += 0.2;
        }
    }

    return std::min(1.0, pattern_confidence);
}

VectorXd WindGustHandler::getFilterResponse(const VectorXd& frequencies) const {
    if (bandpass_filter_->getNumChannels() > 0) {
        // Get response from first channel (assuming all channels are identical)
        return VectorXd::Zero(frequencies.size());  // Placeholder
    }
    return VectorXd::Zero(frequencies.size());
}

void WindGustHandler::reset() {
    acceleration_history_.clear();
    filtered_acceleration_history_.clear();
    timestamp_history_.clear();
    wind_history_.clear();

    bandpass_filter_->reset();

    current_wind_estimate_.setZero();
    filtered_wind_.setZero();
    gust_detected_ = false;
    gust_magnitude_ = 0.0;
    gust_direction_.setZero();
    gust_start_time_ = 0.0;
    gust_duration_ = 0.0;

    mean_acceleration_.setZero();
    acceleration_variance_.setOnes();
    detection_rate_ = 0.0;
    false_positive_count_ = 0;
    true_positive_count_ = 0;
}

std::map<std::string, double> WindGustHandler::getWindStats() const {
    std::map<std::string, double> stats;

    stats["gust_detected"] = gust_detected_ ? 1.0 : 0.0;
    stats["gust_magnitude"] = gust_magnitude_;
    stats["wind_speed"] = current_wind_estimate_.norm();
    stats["detection_rate"] = detection_rate_;
    stats["gust_duration"] = gust_duration_;
    stats["noise_level"] = sqrt(acceleration_variance_.norm());
    stats["adaptive_threshold"] = calculateAdaptiveThreshold();

    if (true_positive_count_ + false_positive_count_ > 0) {
        stats["detection_accuracy"] = (double)true_positive_count_ / 
                                     (true_positive_count_ + false_positive_count_);
    } else {
        stats["detection_accuracy"] = 0.0;
    }

    return stats;
}

bool WindGustHandler::validateWindEstimate() const {
    // Check if wind estimate is physically reasonable
    double wind_speed = current_wind_estimate_.norm();

    if (wind_speed > max_wind_speed_) {
        return false;
    }

    // Check for sudden unrealistic changes
    if (wind_history_.size() >= 2) {
        WindVector prev_wind = wind_history_[wind_history_.size()-2];
        double change_rate = (current_wind_estimate_ - prev_wind).norm();
        if (change_rate > max_wind_speed_) {  // Change rate limit
            return false;
        }
    }

    return true;
}

} // namespace aimm_cs_ducmkf
