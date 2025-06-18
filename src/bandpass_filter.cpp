#include "bandpass_filter.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace aimm_cs_ducmkf {

BandpassFilter::BandpassFilter(double sampling_freq, double low_cutoff, double high_cutoff, 
                               int num_taps, const std::string& window_type)
    : sampling_freq_(sampling_freq), low_cutoff_(low_cutoff), high_cutoff_(high_cutoff),
      num_taps_(num_taps), delay_index_(0) {

    if (sampling_freq <= 0) {
        throw std::invalid_argument("Sampling frequency must be positive");
    }
    if (low_cutoff >= high_cutoff) {
        throw std::invalid_argument("Low cutoff must be less than high cutoff");
    }
    if (high_cutoff >= sampling_freq / 2.0) {
        throw std::invalid_argument("High cutoff must be less than Nyquist frequency");
    }
    if (num_taps % 2 == 0) {
        throw std::invalid_argument("Number of taps should be odd for symmetric response");
    }

    // Initialize delay line
    delay_line_.resize(num_taps_, 0.0);

    // Design the filter
    designFilter();
    applyWindow(window_type);
}

void BandpassFilter::designFilter() {
    coefficients_.resize(num_taps_);

    // Normalized cutoff frequencies
    double w1 = 2.0 * M_PI * low_cutoff_ / sampling_freq_;
    double w2 = 2.0 * M_PI * high_cutoff_ / sampling_freq_;

    int M = num_taps_ - 1;
    int center = M / 2;

    for (int n = 0; n < num_taps_; ++n) {
        if (n == center) {
            // Handle n = M/2 case (avoid division by zero)
            coefficients_(n) = (w2 - w1) / M_PI;
        } else {
            double nn = n - center;
            coefficients_(n) = (sin(w2 * nn) - sin(w1 * nn)) / (M_PI * nn);
        }
    }
}

void BandpassFilter::applyWindow(const std::string& window_type) {
    VectorXd window = generateWindow(num_taps_, window_type);

    // Apply window to coefficients
    for (int i = 0; i < num_taps_; ++i) {
        coefficients_(i) *= window(i);
    }

    // Normalize to unit gain at center frequency
    double center_freq = (low_cutoff_ + high_cutoff_) / 2.0;
    VectorXd test_freq(1);
    test_freq(0) = center_freq;
    VectorXd response = getFrequencyResponse(test_freq);
    double gain = response(0);

    if (gain > 1e-10) {
        coefficients_ /= gain;
    }
}

VectorXd BandpassFilter::generateWindow(int N, const std::string& window_type) const {
    VectorXd window(N);

    if (window_type == "hamming") {
        for (int n = 0; n < N; ++n) {
            window(n) = 0.54 - 0.46 * cos(2.0 * M_PI * n / (N - 1));
        }
    } else if (window_type == "hanning") {
        for (int n = 0; n < N; ++n) {
            window(n) = 0.5 * (1.0 - cos(2.0 * M_PI * n / (N - 1)));
        }
    } else if (window_type == "blackman") {
        for (int n = 0; n < N; ++n) {
            double x = 2.0 * M_PI * n / (N - 1);
            window(n) = 0.42 - 0.5 * cos(x) + 0.08 * cos(2.0 * x);
        }
    } else {
        // Default to rectangular window
        window.setOnes();
    }

    return window;
}

double BandpassFilter::filter(double input) {
    // Shift delay line
    delay_line_[delay_index_] = input;

    // Compute convolution
    double output = 0.0;
    for (int i = 0; i < num_taps_; ++i) {
        int index = (delay_index_ + i) % num_taps_;
        output += coefficients_(num_taps_ - 1 - i) * delay_line_[index];
    }

    // Update delay index
    delay_index_ = (delay_index_ + num_taps_ - 1) % num_taps_;

    return output;
}

VectorXd BandpassFilter::filter(const VectorXd& input) {
    VectorXd output(input.size());

    for (int i = 0; i < input.size(); ++i) {
        output(i) = filter(input(i));
    }

    return output;
}

void BandpassFilter::reset() {
    std::fill(delay_line_.begin(), delay_line_.end(), 0.0);
    delay_index_ = 0;
}

VectorXd BandpassFilter::getFrequencyResponse(const VectorXd& frequencies) const {
    VectorXd response(frequencies.size());

    for (int f = 0; f < frequencies.size(); ++f) {
        double freq = frequencies(f);
        double w = 2.0 * M_PI * freq / sampling_freq_;

        std::complex<double> H(0.0, 0.0);
        for (int n = 0; n < num_taps_; ++n) {
            H += coefficients_(n) * std::exp(std::complex<double>(0.0, -w * n));
        }

        response(f) = std::abs(H);
    }

    return response;
}

bool BandpassFilter::isInPassBand(double frequency) const {
    return (frequency >= low_cutoff_ && frequency <= high_cutoff_);
}

// MultiChannelBandpassFilter implementation

MultiChannelBandpassFilter::MultiChannelBandpassFilter(double sampling_freq, double low_cutoff, 
                                                       double high_cutoff, int num_channels, 
                                                       int num_taps)
    : num_channels_(num_channels) {

    filters_.reserve(num_channels_);
    for (int i = 0; i < num_channels_; ++i) {
        filters_.emplace_back(std::make_unique<BandpassFilter>(sampling_freq, low_cutoff, 
                                                              high_cutoff, num_taps));
    }
}

VectorXd MultiChannelBandpassFilter::filter(const VectorXd& input) {
    if (input.size() != num_channels_) {
        throw std::invalid_argument("Input size must match number of channels");
    }

    VectorXd output(num_channels_);
    for (int i = 0; i < num_channels_; ++i) {
        output(i) = filters_[i]->filter(input(i));
    }

    return output;
}

void MultiChannelBandpassFilter::reset() {
    for (auto& filter : filters_) {
        filter->reset();
    }
}

} // namespace aimm_cs_ducmkf
