#include "aimm_cs_ducmkf.h"

#include <fstream>
#include <iostream>
#include <random>
#include <vector>

using namespace aimm_cs_ducmkf;

struct SimulationData
{
    std::vector<double> time_points;
    std::vector<Eigen::Vector3d> true_positions;
    std::vector<Eigen::Vector3d> estimated_positions;
    std::vector<double> position_errors;
    std::vector<int> active_models;
    std::vector<bool> maneuver_detected;
    std::vector<bool> wind_detected;
};

class TargetSimulator
{
  public:
    TargetSimulator() : rng_(std::random_device{}()), normal_dist_(0.0, 1.0) {}

    StateVector generateTrueState(const double time,
                                  const StateVector &initial_state,
                                  const double maneuver_time)
    {
        StateVector true_state =
            StateVector::Zero(6); // 6-state vector for position and velocity
        true_state = initial_state;
        const auto &position = initial_state.head<3>();
        const auto &velocity = initial_state.tail<3>();
        // Straight line motion until maneuver time
        if (time < maneuver_time)
        {
            true_state(0) = position(0) + velocity(0) * time; // x position
            true_state(1) = position(1) + velocity(1) * time; // y position
            true_state(2) = position(2) + velocity(2) * time; // z position
            true_state(3) = velocity(0);                      // x velocity
            true_state(4) = velocity(1);                      // y velocity
            true_state(5) = velocity(2);                      // z velocity
        }
        else
        {
            // Coordinated turn maneuver
            double maneuver_time = time - maneuver_time;
            double turn_rate = 0.1; // rad/s
            double radius = velocity.norm() / turn_rate;
            true_state(0) = position(0) + velocity(0) * maneuver_time +
                            radius * sin(turn_rate * maneuver_time);
            true_state(1) = position(1) + velocity(1) * maneuver_time +
                            radius * (1.0 - cos(turn_rate * maneuver_time));
            true_state(2) = position(2) + velocity(2) * maneuver_time;
            true_state(3) = velocity(0) * cos(turn_rate * maneuver_time);
            true_state(4) = velocity(1) * sin(turn_rate * maneuver_time);
            true_state(5) = velocity(2);
        }
        return true_state;
    }

    Measurement generateCartesianMeasurement(const StateVector &true_state,
            const double time,
            const double range_noise_std,
            const double angle_noise_std)
    {
        Measurement measurement;
        const auto &position = true_state.head<3>();
        
        // Use appropriate noise levels for Cartesian coordinates
        // X, Y, Z positions should have similar noise characteristics
        measurement.measurement(0) = position(0) + range_noise_std * normal_dist_(rng_); // x
        measurement.measurement(1) = position(1) + range_noise_std * normal_dist_(rng_); // y  
        measurement.measurement(2) = position(2) + range_noise_std * normal_dist_(rng_); // z
        
        // Set measurement covariance - use consistent noise for Cartesian coordinates
        measurement.noise_covariance.setZero();
        measurement.noise_covariance(0, 0) = range_noise_std * range_noise_std;
        measurement.noise_covariance(1, 1) = range_noise_std * range_noise_std;
        measurement.noise_covariance(2, 2) = range_noise_std * range_noise_std;
        measurement.type = MeasurementType::CARTESIAN;
        measurement.timestamp = time;
        return measurement;
    }

    Measurement generatePolarMeasurement(const StateVector &true_state,
                                         const double time,
                                         const double range_noise_std,
                                         const double angle_noise_std)
    {
        Measurement measurement;
        const auto &position = true_state.head<3>();
        Vector3d polar = cartesianToPolar(position);
        measurement.measurement(0) =
            polar(0) + range_noise_std * normal_dist_(rng_); // range
        measurement.measurement(1) =
            polar(1) + angle_noise_std * normal_dist_(rng_); // azimuth
        measurement.measurement(2) =
            polar(2) + angle_noise_std * normal_dist_(rng_); // elevation
        // Set measurement covariance
        measurement.noise_covariance.setZero();
        measurement.noise_covariance(0, 0) = range_noise_std * range_noise_std;
        measurement.noise_covariance(1, 1) = angle_noise_std * angle_noise_std;
        measurement.noise_covariance(2, 2) = angle_noise_std * angle_noise_std;
        measurement.type = MeasurementType::POLAR;
        measurement.timestamp = time;
        return measurement;
    }

    Vector3d generateWindGust(const double time, const double wind_start_time,
                              const double wind_duration,
                              const double wind_magnitude)
    {
        Vector3d wind = Vector3d::Zero();
        // Generate wind gust
        if (time >= wind_start_time && time <= wind_start_time + wind_duration)
        {
            double gust_time = time - wind_start_time;
            // Downward gust followed by upward correction pattern
            double pattern_phase = 2.0 * M_PI * gust_time / 5.0; // 5 second period
            wind(0) = wind_magnitude * 0.3 * sin(pattern_phase); // x-component
            wind(1) =
                wind_magnitude * 0.2 * sin(pattern_phase + M_PI / 4); // y-component
            wind(2) =
                -wind_magnitude * sin(pattern_phase); // z-component (down then up)
            // Add some noise
            for (int i = 0; i < 3; ++i)
            {
                wind(i) += 0.1 * wind_magnitude * normal_dist_(rng_);
            }
        }
        return wind;
    }

    Vector3d generateAccelerationMeasurement(const StateVector &true_state,
            const double time,
            const double maneuver_time,
            const Vector3d initial_velocity,
            const double wind_start_time,
            const double wind_duration,
            const double wind_magnitude)
    {
        // For this example, we simulate acceleration from numerical differentiation
        // In practice, this would come from an IMU
        Vector3d acceleration = Vector3d::Zero();
        if (time > maneuver_time)
        {
            // During turn maneuver
            double turn_rate = 0.1;
            acceleration(0) = -initial_velocity.norm() * turn_rate *
                              sin(turn_rate * (time - maneuver_time));
            acceleration(1) = initial_velocity.norm() * turn_rate *
                              cos(turn_rate * (time - maneuver_time));
            acceleration(2) = 0.0;
        }
        // Add wind effects
        const Vector3d wind =
            generateWindGust(time, wind_start_time, wind_duration, wind_magnitude);
        acceleration += wind / 2.0; // Convert wind to acceleration effect
        // Add measurement noise
        for (int i = 0; i < 3; ++i)
        {
            acceleration(i) += 0.1 * normal_dist_(rng_);
        }
        return acceleration;
    }

  private:
    std::mt19937 rng_;
    std::normal_distribution<double> normal_dist_;
};