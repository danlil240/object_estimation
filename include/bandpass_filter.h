#ifndef AIMM_CS_DUCMKF_BANDPASS_FILTER_H
#define AIMM_CS_DUCMKF_BANDPASS_FILTER_H

#include "types.h"
#include <vector>
#include <cmath>

namespace aimm_cs_ducmkf {

/**
 * @brief Digital Bandpass Filter for wind gust detection and filtering
 * 
 * This class implements a FIR bandpass filter to isolate wind gust frequencies
 * from the target motion signal. It combines high-pass and low-pass filtering
 * to create a band-pass effect that highlights periodic disturbances.
 */
class BandpassFilter {
public:
    /**
     * @brief Constructor for bandpass filter
     * @param sampling_freq Sampling frequency in Hz
     * @param low_cutoff Lower cutoff frequency in Hz
     * @param high_cutoff Upper cutoff frequency in Hz
     * @param num_taps Number of filter taps (should be odd for symmetric response)
     * @param window_type Windowing function type (default: Hamming)
     */
    BandpassFilter(double sampling_freq, double low_cutoff, double high_cutoff, 
                   int num_taps = 101, const std::string& window_type = "hamming");

    /**
     * @brief Destructor
     */
    ~BandpassFilter() = default;

    /**
     * @brief Filter a single sample
     * @param input Input sample
     * @return Filtered output sample
     */
    double filter(double input);

    /**
     * @brief Filter a vector of samples
     * @param input Input vector
     * @return Filtered output vector
     */
    VectorXd filter(const VectorXd& input);

    /**
     * @brief Reset filter state (clear delay line)
     */
    void reset();

    /**
     * @brief Get the filter coefficients
     * @return Vector of filter coefficients
     */
    const VectorXd& getCoefficients() const { return coefficients_; }

    /**
     * @brief Get the filter's frequency response at given frequencies
     * @param frequencies Vector of frequencies to evaluate
     * @return Vector of magnitude responses
     */
    VectorXd getFrequencyResponse(const VectorXd& frequencies) const;

    /**
     * @brief Check if input frequency is within the pass band
     * @param frequency Frequency to check in Hz
     * @return True if frequency is in pass band
     */
    bool isInPassBand(double frequency) const;

private:
    double sampling_freq_;
    double low_cutoff_;
    double high_cutoff_;
    int num_taps_;
    VectorXd coefficients_;
    std::vector<double> delay_line_;
    int delay_index_;

    /**
     * @brief Design the bandpass filter coefficients
     */
    void designFilter();

    /**
     * @brief Apply windowing function to filter coefficients
     * @param window_type Type of window ("hamming", "hanning", "blackman")
     */
    void applyWindow(const std::string& window_type);

    /**
     * @brief Generate windowing function
     * @param N Length of window
     * @param window_type Type of window
     * @return Window coefficients
     */
    VectorXd generateWindow(int N, const std::string& window_type) const;
};

/**
 * @brief Multi-channel bandpass filter for 3D wind gust filtering
 * 
 * This class applies bandpass filtering to each dimension of wind disturbance
 * separately, allowing for different characteristics in different directions.
 */
class MultiChannelBandpassFilter {
public:
    /**
     * @brief Constructor
     * @param sampling_freq Sampling frequency in Hz
     * @param low_cutoff Lower cutoff frequency in Hz
     * @param high_cutoff Upper cutoff frequency in Hz
     * @param num_channels Number of channels (typically 3 for x,y,z)
     * @param num_taps Number of filter taps
     */
    MultiChannelBandpassFilter(double sampling_freq, double low_cutoff, 
                              double high_cutoff, int num_channels = 3, 
                              int num_taps = 101);

    /**
     * @brief Filter multi-channel input
     * @param input Input vector (length = num_channels)
     * @return Filtered output vector
     */
    VectorXd filter(const VectorXd& input);

    /**
     * @brief Reset all filters
     */
    void reset();

    /**
     * @brief Get number of channels
     */
    int getNumChannels() const { return num_channels_; }

private:
    int num_channels_;
    std::vector<std::unique_ptr<BandpassFilter>> filters_;
};

} // namespace aimm_cs_ducmkf

#endif // AIMM_CS_DUCMKF_BANDPASS_FILTER_H
