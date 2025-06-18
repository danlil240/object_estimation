#include "aimm_cs_ducmkf.h"

#include "sim.cpp"
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

using namespace aimm_cs_ducmkf;

/**
 * @brief Example demonstrating AIMM-CS-DUCMKF library usage
 *
 * This example simulates a maneuvering target with wind gusts and shows
 * how to use the complete AIMM-CS-DUCMKF tracking system.
 */

// Simulation parameters
constexpr double SIMULATION_TIME = 2.0;  // seconds
constexpr double DT = 0.1;                // time step
constexpr double MEASUREMENT_RATE = 10.0; // Hz

// Target trajectory parameters
Vector3d INITIAL_POSITION = Vector3d(500, 3000, 100); // m
Vector3d INITIAL_VELOCITY = Vector3d(-2.0, -10.0, 0); // m/s
constexpr double MANEUVER_TIME = 20.0;                // start maneuver at 20s
constexpr double WIND_START_TIME = 30.0;              // wind gust at 30s
constexpr double WIND_DURATION = 10.0;                // wind lasts 10s

// Noise parameters - adjusted for Cartesian measurements
constexpr double RANGE_NOISE_STD = 0.5;   // meters - smaller for Cartesian
constexpr double ANGLE_NOISE_STD = 0.005; // radians - not used for Cartesian
constexpr double WIND_MAGNITUDE = 10.0;   // m/s

void printTrackingResults(
    const double time, const Measurement &measurement,
    const StateVector &true_state,
    const AIMM_CS_DUCMKF_Tracker::TrackingResult &result)
{
    Vector3d true_pos = true_state.head<3>();
    Vector3d est_pos = result.state.head<3>();
    const Vector3d pos_error = true_pos - est_pos;
    LOG_DEBUG("Time: {}s | Measurement: [{}, {}, {}] | True Pos: [{}, {}, {}] | "
              "Est Pos: "
              "[{}, {}, {}] | "
              "Error: {}m | Model: {} | Maneuver: {} | Wind: {}",
              fmt_eng(time), fmt_eng(measurement.measurement(0)),
              fmt_eng(measurement.measurement(1)),
              fmt_eng(measurement.measurement(2)), fmt_eng(true_pos(0)),
              fmt_eng(true_pos(1)), fmt_eng(true_pos(2)), fmt_eng(est_pos(0)),
              fmt_eng(est_pos(1)), fmt_eng(est_pos(2)), fmt_eng(pos_error.norm()),
              result.most_likely_model, result.maneuver_detected ? "YES" : "NO",
              result.wind_gust_detected ? "YES" : "NO");
}

void saveSimulationData(const SimulationData &data)
{
    std::ofstream file("simulation_data.csv");
    if (file.is_open())
    {
        file << "Time,True_X,True_Y,True_Z,Est_X,Est_Y,Est_Z,Position_Error,"
             "Active_"
             "Model,Maneuver_Detected,Wind_Detected\n";
        for (size_t i = 0; i < data.time_points.size(); ++i)
        {
            const Vector3d &true_pos = data.true_positions[i];
            const Vector3d &est_pos = data.estimated_positions[i];
            file << data.time_points[i] << "," << true_pos(0) << "," << true_pos(1)
                 << "," << true_pos(2) << "," << est_pos(0) << "," << est_pos(1)
                 << "," << est_pos(2) << "," << data.position_errors[i] << ","
                 << data.active_models[i] << ","
                 << (data.maneuver_detected[i] ? 1 : 0) << ","
                 << (data.wind_detected[i] ? 1 : 0) << "\n";
        }
        file.close();
        LOG_INFO("Simulation data saved to simulation_data.csv");
    }
    else
    {
        LOG_ERROR("Unable to open file for writing simulation data");
    }
}

int main()
{
    // Initialize logger
    Logger::initialize();
    LOG_INFO("Starting AIMM-CS-DUCMKF example simulation");
    LOG_INFO("Simulation Parameters:");
    LOG_INFO("  Duration: {} seconds", SIMULATION_TIME);
    LOG_INFO("  Time step: {} seconds", DT);
    LOG_INFO("  Measurement rate: {} Hz", MEASUREMENT_RATE);
    LOG_INFO("  Initial position: ({}, {}, {}) m", INITIAL_POSITION[0],
             INITIAL_POSITION[1], INITIAL_POSITION[2]);
    LOG_INFO("  Initial velocity: ({}, {}, {}) m/s", INITIAL_VELOCITY[0],
             INITIAL_VELOCITY[1], INITIAL_VELOCITY[2]);
    try
    {
        // Create tracker configuration
        AIMM_CS_DUCMKF_Tracker::TrackerConfig config;
        config.sampling_frequency = MEASUREMENT_RATE;
        config.wind_gust_low_freq = 0.1;  // Hz
        config.wind_gust_high_freq = 2.0; // Hz
        config.enable_wind_compensation = true;
        config.position_noise = 0.1;      // m^2/s^4 (process noise)
        config.velocity_noise = 0.01;     // (m/s)^2/s^2 (process noise)
        config.maneuver_threshold = 2.0;  // Lower threshold for better detection
        config.cs_window_size = 5;        // Smaller window for faster adaptation
        // Create tracker
        AIMM_CS_DUCMKF_Tracker tracker(config);
        // Initialize tracker with a more realistic initial guess
        // Use the first measurement to get a better initial estimate
        StateVector initial_state = StateVector::Zero(6);
        initial_state.head<3>() = INITIAL_POSITION;  // Use true initial position as guess
        initial_state.tail<3>() = INITIAL_VELOCITY;  // Use true initial velocity as guess
        
        StateMatrix initial_covariance = StateMatrix::Identity(6, 6);
        // Set more realistic initial uncertainties
        initial_covariance.block<3, 3>(0, 0) = Matrix3d::Identity() * 100.0; // 10m std dev squared
        initial_covariance.block<3, 3>(3, 3) = Matrix3d::Identity() * 25.0;  // 5 m/s std dev squared
        if (!tracker.initialize(initial_state, initial_covariance))
        {
            LOG_ERROR("Failed to initialize tracker");
            return -1;
        }
        // After tracker initialization in main.cpp
        LOG_DEBUG("Tracker initialized. State:");
        LOG_DEBUG("  Position: ({}, {}, {})", tracker.getState()(0),
                  tracker.getState()(1), tracker.getState()(2));
        LOG_DEBUG("  Velocity: ({}, {}, {})", tracker.getState()(3),
                  tracker.getState()(4), tracker.getState()(5));
        // Create target simulator
        TargetSimulator simulator;
        // Data collection
        SimulationData data;
        // Simulation loop
        LOG_INFO("Starting simulation...");
        double time = 0.0;
        while (time <= SIMULATION_TIME)
        {
            // Generate true target state
            StateVector true_state =
                simulator.generateTrueState(time, initial_state, MANEUVER_TIME);
            // Generate measurements
            Measurement measurement = simulator.generateCartesianMeasurement(
                                          true_state, time, RANGE_NOISE_STD, ANGLE_NOISE_STD);
            LOG_DEBUG("Measurement at time {}s: x={}, y={}, z={}",
                      fmt_eng(time), fmt_eng(measurement.measurement(0)),
                      fmt_eng(measurement.measurement(1)),
                      fmt_eng(measurement.measurement(2)));
            // Process measurement
            bool success = tracker.processMeasurement(measurement, time);
            if (!success)
            {
                LOG_ERROR("Failed to process measurement at time {}", time);
                continue;
            }
            // Collect data
            AIMM_CS_DUCMKF_Tracker::TrackingResult result =
                tracker.getTrackingResult();
            data.time_points.push_back(time);
            data.true_positions.emplace_back(true_state.head<3>());
            data.estimated_positions.emplace_back(result.state.head<3>());
            data.position_errors.push_back(
                (true_state.head<3>() - result.state.head<3>()).norm());
            data.active_models.push_back(result.most_likely_model);
            data.maneuver_detected.push_back(result.maneuver_detected);
            data.wind_detected.push_back(result.wind_gust_detected);
            // Print results every second
            printTrackingResults(time, measurement, true_state, result);
            time += DT;
        }
        // Save simulation data
        saveSimulationData(data);
        // Print final statistics
        LOG_INFO("Simulation completed!");
        AIMM_CS_DUCMKF_Tracker::TrackingResult final_result =
            tracker.getTrackingResult();
        LOG_INFO("Final Performance Metrics:");
        for (const auto &metric : final_result.performance_metrics)
        {
            LOG_INFO("{}: {}", metric.first, metric.second);
        }
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Error: {}", e.what());
        return -1;
    }
    return 0;
}
