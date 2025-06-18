#ifndef AIMM_CS_DUCMKF_WIND_GUST_HANDLER_H
#define AIMM_CS_DUCMKF_WIND_GUST_HANDLER_H

#include "types.h"
#include "bandpass_filter.h"
#include "imm_filter.h"
#include <memory>
#include <deque>

namespace aimm_cs_ducmkf {

/**
 * @brief Wind Gust Handler with Bandpass Filtering
 * 
 * This class handles wind gust detection and compensation using bandpass filtering
 * and disturbance estimation. It integrates with the IMM filter to provide
 * robust tracking in the presence of wind disturbances.
 */
class WindGustHandler {
public:
    /**
     * @brief Constructor
     * @param sampling_freq Sampling frequency in Hz
     * @param gust_low_freq Lower cutoff frequency for gust detection in Hz
     * @param gust_high_freq Upper cutoff frequency for gust detection in Hz
     * @param filter_taps Number of bandpass filter taps
     */
    WindGustHandler(double sampling_freq = 10.0, 
                   double gust_low_freq = 0.1, 
                   double gust_high_freq = 2.0,
                   int filter_taps = 101);

    /**
     * @brief Destructor
     */
    ~WindGustHandler() = default;

    /**
     * @brief Initialize the wind gust handler
     * @param imm_filter Pointer to the IMM filter
     */
    void initialize(std::shared_ptr<IMMFilter> imm_filter);

    /**
     * @brief Process acceleration measurements to detect and filter wind gusts
     * @param acceleration Measured acceleration vector [ax, ay, az]
     * @param timestamp Current timestamp
     * @return Filtered acceleration with wind gust effects removed
     */
    Vector3d processAcceleration(const Vector3d& acceleration, double timestamp);

    /**
     * @brief Detect wind gust events
     * @param acceleration Current acceleration measurement
     * @param timestamp Current timestamp
     * @return True if wind gust is detected
     */
    bool detectWindGust(const Vector3d& acceleration, double timestamp);

    /**
     * @brief Estimate wind disturbance vector
     * @param state Current state estimate
     * @param measurement Current measurement
     * @return Estimated wind disturbance
     */
    WindVector estimateWindDisturbance(const StateVector& state, 
                                      const Vector3d& measurement);

    /**
     * @brief Compensate for wind effects in state prediction
     * @param predicted_state Predicted state without wind compensation
     * @param dt Time step
     * @return Wind-compensated state prediction
     */
    StateVector compensateWindEffects(const StateVector& predicted_state, double dt);

    /**
     * @brief Get current wind estimate
     * @return Current wind velocity estimate
     */
    const WindVector& getCurrentWindEstimate() const { return current_wind_estimate_; }

    /**
     * @brief Get wind gust detection status
     * @return True if wind gust is currently detected
     */
    bool isWindGustDetected() const { return gust_detected_; }

    /**
     * @brief Get wind gust magnitude
     * @return Magnitude of current wind gust
     */
    double getWindGustMagnitude() const { return gust_magnitude_; }

    /**
     * @brief Get wind gust direction
     * @return Unit vector indicating wind gust direction
     */
    Vector3d getWindGustDirection() const { return gust_direction_; }

    /**
     * @brief Set wind gust detection threshold
     * @param threshold Detection threshold (multiple of noise standard deviation)
     */
    void setGustDetectionThreshold(double threshold) { gust_threshold_ = threshold; }

    /**
     * @brief Set maximum wind speed limit
     * @param max_wind Maximum expected wind speed in m/s
     */
    void setMaxWindSpeed(double max_wind) { max_wind_speed_ = max_wind; }

    /**
     * @brief Enable/disable wind compensation
     * @param enable True to enable wind compensation
     */
    void enableWindCompensation(bool enable) { wind_compensation_enabled_ = enable; }

    /**
     * @brief Get bandpass filter frequency response
     * @param frequencies Vector of frequencies to evaluate
     * @return Magnitude response at given frequencies
     */
    VectorXd getFilterResponse(const VectorXd& frequencies) const;

    /**
     * @brief Reset wind gust handler state
     */
    void reset();

    /**
     * @brief Get performance statistics
     * @return Map of performance metrics
     */
    std::map<std::string, double> getWindStats() const;

protected:
    /**
     * @brief Update wind statistics
     * @param acceleration Current acceleration
     * @param timestamp Current timestamp
     */
    void updateWindStatistics(const Vector3d& acceleration, double timestamp);

    /**
     * @brief Calculate adaptive detection threshold
     * @return Adaptive threshold based on recent noise statistics
     */
    double calculateAdaptiveThreshold() const;

    /**
     * @brief Estimate wind persistence
     * @return Estimated duration of current wind condition
     */
    double estimateWindPersistence() const;

    /**
     * @brief Update wind model parameters
     */
    void updateWindModel();

    /**
     * @brief Validate wind estimates
     * @return True if wind estimates are physically reasonable
     */
    bool validateWindEstimate() const;

private:
    // Filter components
    std::unique_ptr<MultiChannelBandpassFilter> bandpass_filter_;
    std::shared_ptr<IMMFilter> imm_filter_;

    // Wind detection parameters
    double sampling_freq_;
    double gust_low_freq_;
    double gust_high_freq_;
    double gust_threshold_;
    double max_wind_speed_;

    // Wind state variables
    WindVector current_wind_estimate_;
    WindVector filtered_wind_;
    bool gust_detected_;
    double gust_magnitude_;
    Vector3d gust_direction_;
    double gust_start_time_;
    double gust_duration_;

    // Data buffers
    std::deque<Vector3d> acceleration_history_;
    std::deque<Vector3d> filtered_acceleration_history_;
    std::deque<double> timestamp_history_;
    std::deque<WindVector> wind_history_;

    // Configuration flags
    bool wind_compensation_enabled_;
    bool adaptive_threshold_;
    bool initialized_;

    // Statistics
    int buffer_size_;
    Vector3d mean_acceleration_;
    Vector3d acceleration_variance_;
    double detection_rate_;
    int false_positive_count_;
    int true_positive_count_;

    /**
     * @brief Add acceleration sample to history buffer
     * @param acceleration Acceleration vector
     * @param timestamp Time stamp
     */
    void addToHistory(const Vector3d& acceleration, double timestamp);

    /**
     * @brief Calculate noise statistics from acceleration history
     */
    void calculateNoiseStatistics();

    /**
     * @brief Apply median filter to reduce impulse noise
     * @param data Input data vector
     * @param window_size Median filter window size
     * @return Median filtered data
     */
    Vector3d applyMedianFilter(const std::deque<Vector3d>& data, int window_size) const;

    /**
     * @brief Estimate wind direction from filtered acceleration
     * @param filtered_accel Filtered acceleration vector
     * @return Estimated wind direction unit vector
     */
    Vector3d estimateWindDirection(const Vector3d& filtered_accel) const;

    /**
     * @brief Check for wind gust patterns
     * @return Confidence level of wind gust pattern (0-1)
     */
    double checkGustPattern() const;
};

} // namespace aimm_cs_ducmkf

#endif // AIMM_CS_DUCMKF_WIND_GUST_HANDLER_H
