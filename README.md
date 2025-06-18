# AIMM-CS-DUCMKF Library

A C++ library implementing Adaptive Interactive Multiple Model with Current Statistical model and Decorrelated Unbiased Conversion Measurement Kalman Filter (AIMM-CS-DUCMKF) for robust object tracking with wind gust handling.

## Features

- **Adaptive Interactive Multiple Model (AIMM)**: Handles multiple motion models with adaptive switching
- **Current Statistical (CS) Model**: Adapts to changing acceleration patterns for maneuvering targets  
- **Decorrelated Unbiased Conversion Measurement Kalman Filter (DUCMKF)**: Handles nonlinear measurements (polar/spherical to Cartesian conversion)
- **Bandpass Filtering**: Detects and filters wind gust disturbances
- **Wind Gust Compensation**: Estimates and compensates for wind effects on target motion

## Requirements

- C++14 or later
- [Eigen3](http://eigen.tuxfamily.org/) library for linear algebra
- CMake 3.10 or later

## Installation

### Ubuntu/Debian
```bash
# Install dependencies
sudo apt-get install libeigen3-dev cmake build-essential

# Build the library
git clone <repository-url>
cd aimm_cs_ducmkf
mkdir build && cd build
cmake ..
make -j4

# Optional: Install system-wide
sudo make install
```

### Using in Your Project

#### CMake Integration
```cmake
find_package(AIMM_CS_DUCMKF REQUIRED)
target_link_libraries(your_target aimm_cs_ducmkf)
```

#### Manual Integration
```cmake
# Add include directory
include_directories(/path/to/aimm_cs_ducmkf/include)

# Link against the library
target_link_libraries(your_target /path/to/libaimm_cs_ducmkf.a)
```

## Usage

### Basic Example

```cpp
#include "aimm_cs_ducmkf.h"
using namespace aimm_cs_ducmkf;

// Create tracker configuration
TrackerConfig config;
config.sampling_frequency = 10.0;  // Hz
config.enable_wind_compensation = true;

// Create and initialize tracker
AIMM_CS_DUCMKF_Tracker tracker(config);
StateVector initial_state = StateVector::Zero();
StateMatrix initial_covariance = StateMatrix::Identity() * 10.0;
tracker.initialize(initial_state, initial_covariance);

// Process measurements
Measurement measurement;
measurement.measurement << range, azimuth, elevation;  // Polar coordinates
measurement.type = MeasurementType::POLAR;

Vector3d acceleration;  // From IMU or estimated
tracker.processMeasurement(measurement, timestamp, acceleration);

// Get results
TrackingResult result = tracker.getTrackingResult();
StateVector estimated_state = result.state;
bool maneuver_detected = result.maneuver_detected;
bool wind_detected = result.wind_gust_detected;
```

### Advanced Usage

#### Custom Model Configuration
```cpp
// Create IMM filter with custom models
IMMFilter imm_filter(3);  // 3 models

// Add custom models
auto ducmkf = std::make_shared<DUCMKF>(6, 3);
imm_filter.addFilterModel(ducmkf, FilterModel::CONSTANT_VELOCITY, 0.6);

auto cs_model = std::make_shared<CurrentStatisticalModel>(6, 3, 10, 3.0);
imm_filter.addFilterModel(cs_model, FilterModel::CURRENT_STATISTICAL, 0.4);

// Set transition probabilities
TransitionMatrix transitions(2, 2);
transitions << 0.95, 0.05,
               0.05, 0.95;
imm_filter.setTransitionProbabilities(transitions);
```

#### Wind Gust Handling
```cpp
// Create wind gust handler
WindGustHandler wind_handler(10.0, 0.1, 2.0, 101);  // fs, low_freq, high_freq, taps
wind_handler.initialize(imm_filter);

// Process acceleration for wind detection
Vector3d filtered_accel = wind_handler.processAcceleration(acceleration, timestamp);
bool wind_detected = wind_handler.isWindGustDetected();
WindVector wind_estimate = wind_handler.getCurrentWindEstimate();

// Compensate state prediction
StateVector compensated_state = wind_handler.compensateWindEffects(predicted_state, dt);
```

#### Bandpass Filter Only
```cpp
// Create standalone bandpass filter
BandpassFilter filter(10.0, 0.1, 2.0, 101);  // fs, low_freq, high_freq, taps

// Filter single samples
double filtered_sample = filter.filter(input_sample);

// Filter vectors
VectorXd input_vector(100);
VectorXd filtered_vector = filter.filter(input_vector);

// Multi-channel filtering
MultiChannelBandpassFilter mc_filter(10.0, 0.1, 2.0, 3, 101);  // 3 channels
Vector3d filtered_3d = mc_filter.filter(input_3d);
```

## Configuration Parameters

### TrackerConfig
- `num_models`: Number of models in IMM (default: 3)
- `sampling_frequency`: Sampling rate in Hz (default: 10.0)
- `wind_gust_low_freq`: Lower cutoff for wind detection in Hz (default: 0.1)
- `wind_gust_high_freq`: Upper cutoff for wind detection in Hz (default: 2.0)
- `enable_wind_compensation`: Enable wind effect compensation (default: true)
- `position_noise`: Process noise for position (default: 0.1)
- `velocity_noise`: Process noise for velocity (default: 0.01)
- `cs_window_size`: Window size for CS model (default: 10)
- `maneuver_threshold`: Threshold for maneuver detection (default: 3.0)

### DUCMKF Parameters
- `enable_second_order_bias`: Use second-order bias correction (default: true)
- `max_range`: Maximum valid range in meters (default: 100000)
- `min_range`: Minimum valid range in meters (default: 1.0)

## API Reference

### Main Classes

#### AIMM_CS_DUCMKF_Tracker
- `initialize(initial_state, initial_covariance)`: Initialize tracker
- `processMeasurement(measurement, timestamp, acceleration)`: Process new measurement
- `getState()`: Get current state estimate
- `getTrackingResult()`: Get complete tracking results

#### IMMFilter  
- `addFilterModel(filter, model_type, probability)`: Add filter model
- `predict(dt)`: Prediction step
- `updateWithPolarMeasurement(measurement, covariance)`: Update with polar measurements
- `getModelProbabilities()`: Get current model probabilities

#### DUCMKF
- `updateWithPolarMeasurement(measurement, covariance)`: Specialized polar update
- `convertPolarToCartesian(polar, covariance, predicted_state)`: Coordinate conversion
- `calculateNEES(innovation, covariance)`: Consistency check

#### WindGustHandler
- `processAcceleration(acceleration, timestamp)`: Process acceleration for wind detection
- `detectWindGust(acceleration, timestamp)`: Detect wind gust events
- `estimateWindDisturbance(state, measurement)`: Estimate wind effects

#### BandpassFilter
- `filter(input)`: Filter single sample or vector
- `getFrequencyResponse(frequencies)`: Get filter frequency response
- `isInPassBand(frequency)`: Check if frequency is in pass band

### Data Types

#### State Representation
- `StateVector`: 6D vector [x, y, z, vx, vy, vz] for 3D position and velocity
- `StateMatrix`: 6x6 covariance matrix

#### Measurements
- `MeasurementVector`: 3D vector for measurements
- `ObservationVector`: 3D vector for Cartesian observations  
- `WindVector`: 3D wind velocity vector

#### Results
- `TrackingResult`: Complete tracking results including state, covariance, model probabilities, and detection flags

## Algorithm Details

### AIMM-CS-DUCMKF Process
1. **Model Mixing**: Mix state estimates from previous time step
2. **Prediction**: Predict each model forward in time
3. **Measurement Update**: Update each model with new measurement using DUCMKF
4. **Model Probability Update**: Calculate likelihood and update model probabilities
5. **State Mixing**: Compute final mixed state estimate
6. **Wind Compensation**: Apply wind effect compensation if detected

### Wind Gust Detection
1. **Bandpass Filtering**: Apply bandpass filter to acceleration measurements
2. **Threshold Detection**: Check if filtered signal exceeds adaptive threshold
3. **Pattern Analysis**: Analyze for characteristic wind gust patterns
4. **Wind Estimation**: Estimate wind velocity and direction
5. **Compensation**: Apply wind compensation to state prediction

### DUCMKF Conversion
1. **Polar to Cartesian**: Convert measurements using spherical coordinates
2. **Bias Removal**: Remove second-order conversion bias (UCM)
3. **Decorrelation**: Use predicted state for covariance decorrelation (DUCM)
4. **Update**: Apply standard Kalman update with converted measurements

## Performance Tips

1. **Sampling Rate**: Use consistent sampling rates for optimal performance
2. **Model Selection**: Choose appropriate number of models based on target dynamics
3. **Noise Tuning**: Tune process and measurement noise based on sensor characteristics
4. **Wind Parameters**: Adjust wind detection frequencies based on expected gust characteristics
5. **Buffer Sizes**: Increase buffer sizes for better wind pattern detection

## Limitations

- Assumes 6-state motion model (position + velocity)
- Wind compensation is approximate for highly nonlinear wind effects
- Performance depends on proper noise parameter tuning
- Memory usage scales with buffer sizes and number of models

## License

[Add your license information here]

## References

1. "Decorrelated Unbiased Converted Measurement for Bistatic Radar Tracking"
2. "The Interacting Multiple Model Algorithm for Accurate State Estimation"
3. "Filtering and Estimation of State and Wind Disturbances"

## Contributing

[Add contribution guidelines here]

## Support

[Add support information here]
