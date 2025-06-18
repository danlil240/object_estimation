#include "aimm_cs_ducmkf.h"
#include "sim.cpp"
#include <iostream>
#include <vector>

using namespace aimm_cs_ducmkf;

/**
 * @brief Simple loop test for AIMM-CS-DUCMKF
 * 
 * This test runs the tracker in a loop to validate basic functionality
 * and measure performance over multiple iterations.
 */

int main() {
    // Initialize logger
    Logger::initialize();
    
    LOG_INFO("=== Simple Loop Test for AIMM-CS-DUCMKF ===");
    
    const int NUM_ITERATIONS = 50;
    const double DT = 0.1;
    const double SIMULATION_TIME = 10.0; // 10 seconds per iteration
    
    std::vector<double> position_errors;
    std::vector<double> processing_times;
    int successful_iterations = 0;
    
    LOG_INFO("Running {} iterations, {} seconds each", NUM_ITERATIONS, SIMULATION_TIME);
    
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        LOG_INFO("Starting iteration {}/{}", iter + 1, NUM_ITERATIONS);
        
        // Create tracker
        AIMM_CS_DUCMKF_Tracker::TrackerConfig config;
        config.sampling_frequency = 10.0;
        config.enable_wind_compensation = false;
        config.position_noise = 0.1;
        config.velocity_noise = 0.01;
        
        AIMM_CS_DUCMKF_Tracker tracker(config);
        
        // Initialize with random initial state
        Vector3d initial_pos = Vector3d(500, 3000, 100);
        Vector3d initial_vel = Vector3d(-2, -10, 0);
        
        StateVector initial_state = StateVector::Zero(6);
        initial_state.head<3>() = initial_pos;
        initial_state.tail<3>() = initial_vel;
        
        StateMatrix initial_covariance = StateMatrix::Identity(6, 6);
        initial_covariance.block<3, 3>(0, 0) = Matrix3d::Identity() * 100.0;
        initial_covariance.block<3, 3>(3, 3) = Matrix3d::Identity() * 25.0;
        
        if (!tracker.initialize(initial_state, initial_covariance)) {
            LOG_ERROR("Failed to initialize tracker in iteration {}", iter + 1);
            continue;
        }
        
        // Create simulator
        TargetSimulator simulator;
        
        // Run simulation
        double time = 0.0;
        std::vector<double> iteration_errors;
        bool iteration_success = true;
        
        while (time <= SIMULATION_TIME) {
            // Generate true state
            StateVector true_state = simulator.generateTrueState(time, initial_state, 1000.0); // No maneuver
            
            // Generate measurement
            Measurement measurement = simulator.generateCartesianMeasurement(
                true_state, time, 0.5, 0.005);
            
            // Process measurement
            bool success = tracker.processMeasurement(measurement, time);
            if (!success) {
                LOG_ERROR("Failed to process measurement at time {} in iteration {}", time, iter + 1);
                iteration_success = false;
                break;
            }
            
            // Get tracking result
            AIMM_CS_DUCMKF_Tracker::TrackingResult result = tracker.getTrackingResult();
            
            // Calculate position error
            double pos_error = (true_state.head<3>() - result.state.head<3>()).norm();
            iteration_errors.push_back(pos_error);
            
            time += DT;
        }
        
        if (iteration_success) {
            // Calculate average error for this iteration (excluding first second for convergence)
            double avg_error = 0.0;
            int count = 0;
            for (size_t i = 10; i < iteration_errors.size(); ++i) { // Skip first 10 samples (1 second)
                avg_error += iteration_errors[i];
                count++;
            }
            if (count > 0) {
                avg_error /= count;
                position_errors.push_back(avg_error);
                successful_iterations++;
            }
            
            LOG_INFO("  Iteration {} completed - Avg Position Error: {:.3f}m", iter + 1, avg_error);
        } else {
            LOG_ERROR("  Iteration {} failed", iter + 1);
        }
    }
    
    // Calculate overall statistics
    if (successful_iterations > 0) {
        double avg_position_error = 0.0;
        for (double error : position_errors) {
            avg_position_error += error;
        }
        avg_position_error /= position_errors.size();
        
        double min_error = *std::min_element(position_errors.begin(), position_errors.end());
        double max_error = *std::max_element(position_errors.begin(), position_errors.end());
        
        LOG_INFO("=== Loop Test Results ===");
        LOG_INFO("Successful iterations: {}/{}", successful_iterations, NUM_ITERATIONS);
        LOG_INFO("Average position error: {:.3f}m", avg_position_error);
        LOG_INFO("Min position error: {:.3f}m", min_error);
        LOG_INFO("Max position error: {:.3f}m", max_error);
        
        // Performance assessment
        bool performance_ok = avg_position_error < 2.0 && successful_iterations >= NUM_ITERATIONS * 0.9;
        LOG_INFO("Performance assessment: {}", performance_ok ? "PASSED" : "FAILED");
        
        if (performance_ok) {
            LOG_INFO("✅ Loop test completed successfully!");
        } else {
            LOG_ERROR("❌ Loop test failed performance criteria");
        }
    } else {
        LOG_ERROR("No successful iterations completed");
    }
    
    return 0;
} 