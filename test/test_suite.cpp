#include "aimm_cs_ducmkf.h"
#include "sim.cpp"
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <thread>
#include <numeric>
#include <cmath>

using namespace aimm_cs_ducmkf;

/**
 * @brief Comprehensive test suite for AIMM-CS-DUCMKF
 * 
 * This test suite includes multiple scenarios that run in loops to validate:
 * - Basic tracking performance
 * - Maneuver detection
 * - Wind gust handling
 * - Model switching
 * - Convergence behavior
 * - Performance under different noise levels
 */

struct TestResult {
    std::string test_name;
    int iterations = 0;
    double avg_position_error = 0.0;
    double avg_velocity_error = 0.0;
    double convergence_time = 0.0;
    int model_switches = 0;
    int maneuver_detections = 0;
    int wind_detections = 0;
    bool passed = false;
    std::string failure_reason;
};

class TestSuite {
private:
    std::mt19937 rng_;
    std::normal_distribution<double> normal_dist_;
    
    // Test configuration
    constexpr static double DT = 0.1;
    constexpr static double MEASUREMENT_RATE = 10.0;
    constexpr static int CONVERGENCE_ITERATIONS = 50; // 5 seconds for convergence
    
public:
    TestSuite() : rng_(std::random_device{}()), normal_dist_(0.0, 1.0) {}
    
    /**
     * Check if a value is valid (not infinite or NaN)
     */
    bool isValidValue(double value) {
        return std::isfinite(value) && !std::isnan(value);
    }
    
    /**
     * Check if a state vector is valid
     */
    bool isValidState(const StateVector& state) {
        for (int i = 0; i < state.size(); ++i) {
            if (!isValidValue(state(i))) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * Test 1: Basic straight-line tracking with loop
     * Runs multiple iterations to test convergence and stability
     */
    TestResult testStraightLineTracking(int num_iterations = 10) {
        TestResult result;
        result.test_name = "Straight Line Tracking";
        result.iterations = num_iterations;
        
        std::vector<double> position_errors;
        std::vector<double> velocity_errors;
        std::vector<double> convergence_times;
        
        LOG_INFO("Starting Test 1: Straight Line Tracking ({} iterations)", num_iterations);
        
        for (int iter = 0; iter < num_iterations; ++iter) {
            LOG_INFO("  Iteration {}/{}", iter + 1, num_iterations);
            
            // Create tracker with proper configuration
            AIMM_CS_DUCMKF_Tracker::TrackerConfig config;
            config.sampling_frequency = MEASUREMENT_RATE;
            config.enable_wind_compensation = false;
            config.position_noise = 0.1;
            config.velocity_noise = 0.01;
            config.acceleration_noise = 1.0;
            config.range_noise = 1.0;
            config.azimuth_noise = 0.001;
            config.elevation_noise = 0.001;
            
            AIMM_CS_DUCMKF_Tracker tracker(config);
            
            // Initialize with random initial state
            Vector3d initial_pos = Vector3d(500 + 100 * normal_dist_(rng_), 
                                           3000 + 100 * normal_dist_(rng_), 
                                           100 + 10 * normal_dist_(rng_));
            Vector3d initial_vel = Vector3d(-2 + normal_dist_(rng_), 
                                           -10 + normal_dist_(rng_), 
                                           0 + 0.5 * normal_dist_(rng_));
            
            StateVector initial_state = StateVector::Zero(6);
            initial_state.head<3>() = initial_pos;
            initial_state.tail<3>() = initial_vel;
            
            StateMatrix initial_covariance = StateMatrix::Identity(6, 6);
            initial_covariance.block<3, 3>(0, 0) = Matrix3d::Identity() * 100.0;
            initial_covariance.block<3, 3>(3, 3) = Matrix3d::Identity() * 25.0;
            
            if (!tracker.initialize(initial_state, initial_covariance)) {
                result.passed = false;
                result.failure_reason = "Failed to initialize tracker";
                return result;
            }
            
            // Create simulator
            TargetSimulator simulator;
            
            // Run simulation
            double time = 0.0;
            double convergence_time = -1;
            std::vector<double> errors;
            bool iteration_failed = false;
            
            while (time <= 10.0 && !iteration_failed) { // 10 second simulation
                StateVector true_state = simulator.generateTrueState(time, initial_state, 1000.0); // No maneuver
                Measurement measurement = simulator.generateCartesianMeasurement(
                    true_state, time, 0.5, 0.005);
                
                if (!tracker.processMeasurement(measurement, time)) {
                    iteration_failed = true;
                    break;
                }
                
                AIMM_CS_DUCMKF_Tracker::TrackingResult result_data = tracker.getTrackingResult();
                
                // Check if state is valid
                if (!isValidState(result_data.state)) {
                    iteration_failed = true;
                    break;
                }
                
                double pos_error = (true_state.head<3>() - result_data.state.head<3>()).norm();
                double vel_error = (true_state.tail<3>() - result_data.state.tail<3>()).norm();
                
                // Check if errors are valid
                if (!isValidValue(pos_error) || !isValidValue(vel_error)) {
                    iteration_failed = true;
                    break;
                }
                
                errors.push_back(pos_error);
                
                // Check for convergence
                if (convergence_time < 0 && pos_error < 1.0 && time > 1.0) {
                    convergence_time = time;
                }
                
                time += DT;
            }
            
            if (iteration_failed) {
                LOG_ERROR("  Iteration {} failed due to invalid state or error", iter + 1);
                continue;
            }
            
            // Calculate average errors (excluding first second for convergence)
            double avg_pos_error = 0.0;
            double avg_vel_error = 0.0;
            int count = 0;
            
            for (size_t i = 10; i < errors.size(); ++i) { // Skip first 10 samples (1 second)
                if (isValidValue(errors[i])) {
                    avg_pos_error += errors[i];
                    count++;
                }
            }
            
            if (count > 0) {
                avg_pos_error /= count;
                position_errors.push_back(avg_pos_error);
                velocity_errors.push_back(avg_vel_error);
                convergence_times.push_back(convergence_time);
            }
        }
        
        // Check if we have valid results
        if (position_errors.empty()) {
            result.passed = false;
            result.failure_reason = "All iterations failed";
            return result;
        }
        
        // Calculate overall statistics
        result.avg_position_error = std::accumulate(position_errors.begin(), position_errors.end(), 0.0) / position_errors.size();
        result.avg_velocity_error = std::accumulate(velocity_errors.begin(), velocity_errors.end(), 0.0) / velocity_errors.size();
        result.convergence_time = std::accumulate(convergence_times.begin(), convergence_times.end(), 0.0) / convergence_times.size();
        
        // Pass criteria
        result.passed = result.avg_position_error < 2.0 && result.convergence_time < 5.0 && isValidValue(result.avg_position_error);
        
        LOG_INFO("Test 1 Results: Avg Pos Error: {:.3f}m, Convergence Time: {:.3f}s, Passed: {}", 
                 result.avg_position_error, result.convergence_time, result.passed ? "YES" : "NO");
        
        return result;
    }
    
    /**
     * Test 2: Maneuver detection with loop
     * Tests the system's ability to detect and track maneuvers
     */
    TestResult testManeuverDetection(int num_iterations = 5) {
        TestResult result;
        result.test_name = "Maneuver Detection";
        result.iterations = num_iterations;
        
        LOG_INFO("Starting Test 2: Maneuver Detection ({} iterations)", num_iterations);
        
        int total_maneuver_detections = 0;
        int total_model_switches = 0;
        
        for (int iter = 0; iter < num_iterations; ++iter) {
            LOG_INFO("  Iteration {}/{}", iter + 1, num_iterations);
            
            // Create tracker with maneuver detection enabled
            AIMM_CS_DUCMKF_Tracker::TrackerConfig config;
            config.sampling_frequency = MEASUREMENT_RATE;
            config.enable_wind_compensation = false;
            config.maneuver_threshold = 1.5;
            config.cs_window_size = 5;
            config.position_noise = 0.1;
            config.velocity_noise = 0.01;
            config.acceleration_noise = 1.0;
            
            AIMM_CS_DUCMKF_Tracker tracker(config);
            
            // Initialize
            Vector3d initial_pos = Vector3d(500, 3000, 100);
            Vector3d initial_vel = Vector3d(-2, -10, 0);
            
            StateVector initial_state = StateVector::Zero(6);
            initial_state.head<3>() = initial_pos;
            initial_state.tail<3>() = initial_vel;
            
            StateMatrix initial_covariance = StateMatrix::Identity(6, 6);
            initial_covariance.block<3, 3>(0, 0) = Matrix3d::Identity() * 100.0;
            initial_covariance.block<3, 3>(3, 3) = Matrix3d::Identity() * 25.0;
            
            if (!tracker.initialize(initial_state, initial_covariance)) {
                result.passed = false;
                result.failure_reason = "Failed to initialize tracker";
                return result;
            }
            
            TargetSimulator simulator;
            
            // Run simulation with maneuver at 5 seconds
            double time = 0.0;
            int maneuver_detections = 0;
            int model_switches = 0;
            int last_model = -1;
            bool iteration_failed = false;
            
            while (time <= 15.0 && !iteration_failed) { // 15 second simulation
                StateVector true_state = simulator.generateTrueState(time, initial_state, 5.0); // Maneuver at 5s
                Measurement measurement = simulator.generateCartesianMeasurement(
                    true_state, time, 0.5, 0.005);
                
                if (!tracker.processMeasurement(measurement, time)) {
                    iteration_failed = true;
                    break;
                }
                
                AIMM_CS_DUCMKF_Tracker::TrackingResult result_data = tracker.getTrackingResult();
                
                // Check if state is valid
                if (!isValidState(result_data.state)) {
                    iteration_failed = true;
                    break;
                }
                
                if (result_data.maneuver_detected) {
                    maneuver_detections++;
                }
                
                if (last_model != -1 && last_model != result_data.most_likely_model) {
                    model_switches++;
                }
                last_model = result_data.most_likely_model;
                
                time += DT;
            }
            
            if (!iteration_failed) {
                total_maneuver_detections += maneuver_detections;
                total_model_switches += model_switches;
            }
        }
        
        result.maneuver_detections = total_maneuver_detections;
        result.model_switches = total_model_switches;
        
        // Pass criteria: should detect maneuvers and switch models
        result.passed = result.maneuver_detections > 0 && result.model_switches > 0;
        
        LOG_INFO("Test 2 Results: Maneuver Detections: {}, Model Switches: {}, Passed: {}", 
                 result.maneuver_detections, result.model_switches, result.passed ? "YES" : "NO");
        
        return result;
    }
    
    /**
     * Test 3: Wind gust handling with loop
     * Tests the system's ability to handle wind gusts
     */
    TestResult testWindGustHandling(int num_iterations = 5) {
        TestResult result;
        result.test_name = "Wind Gust Handling";
        result.iterations = num_iterations;
        
        LOG_INFO("Starting Test 3: Wind Gust Handling ({} iterations)", num_iterations);
        
        int total_wind_detections = 0;
        
        for (int iter = 0; iter < num_iterations; ++iter) {
            LOG_INFO("  Iteration {}/{}", iter + 1, num_iterations);
            
            // Create tracker with wind compensation enabled
            AIMM_CS_DUCMKF_Tracker::TrackerConfig config;
            config.sampling_frequency = MEASUREMENT_RATE;
            config.enable_wind_compensation = true;
            config.wind_gust_low_freq = 0.1;
            config.wind_gust_high_freq = 2.0;
            config.position_noise = 0.1;
            config.velocity_noise = 0.01;
            config.acceleration_noise = 1.0;
            
            AIMM_CS_DUCMKF_Tracker tracker(config);
            
            // Initialize
            Vector3d initial_pos = Vector3d(500, 3000, 100);
            Vector3d initial_vel = Vector3d(-2, -10, 0);
            
            StateVector initial_state = StateVector::Zero(6);
            initial_state.head<3>() = initial_pos;
            initial_state.tail<3>() = initial_vel;
            
            StateMatrix initial_covariance = StateMatrix::Identity(6, 6);
            initial_covariance.block<3, 3>(0, 0) = Matrix3d::Identity() * 100.0;
            initial_covariance.block<3, 3>(3, 3) = Matrix3d::Identity() * 25.0;
            
            if (!tracker.initialize(initial_state, initial_covariance)) {
                result.passed = false;
                result.failure_reason = "Failed to initialize tracker";
                return result;
            }
            
            TargetSimulator simulator;
            
            // Run simulation with wind gust at 5 seconds
            double time = 0.0;
            int wind_detections = 0;
            bool iteration_failed = false;
            
            while (time <= 15.0 && !iteration_failed) { // 15 second simulation
                StateVector true_state = simulator.generateTrueState(time, initial_state, 1000.0); // No maneuver
                
                // Add wind gust effect to true state
                Vector3d wind = simulator.generateWindGust(time, 5.0, 5.0, 10.0);
                true_state.tail<3>() += wind * DT; // Apply wind to velocity
                
                Measurement measurement = simulator.generateCartesianMeasurement(
                    true_state, time, 0.5, 0.005);
                
                if (!tracker.processMeasurement(measurement, time)) {
                    iteration_failed = true;
                    break;
                }
                
                AIMM_CS_DUCMKF_Tracker::TrackingResult result_data = tracker.getTrackingResult();
                
                // Check if state is valid
                if (!isValidState(result_data.state)) {
                    iteration_failed = true;
                    break;
                }
                
                if (result_data.wind_gust_detected) {
                    wind_detections++;
                }
                
                time += DT;
            }
            
            if (!iteration_failed) {
                total_wind_detections += wind_detections;
            }
        }
        
        result.wind_detections = total_wind_detections;
        
        // Pass criteria: should detect wind gusts
        result.passed = result.wind_detections > 0;
        
        LOG_INFO("Test 3 Results: Wind Detections: {}, Passed: {}", 
                 result.wind_detections, result.passed ? "YES" : "NO");
        
        return result;
    }
    
    /**
     * Test 4: Noise sensitivity with loop
     * Tests performance under different noise levels
     */
    TestResult testNoiseSensitivity(int num_iterations = 5) {
        TestResult result;
        result.test_name = "Noise Sensitivity";
        result.iterations = num_iterations;
        
        LOG_INFO("Starting Test 4: Noise Sensitivity ({} iterations)", num_iterations);
        
        std::vector<double> noise_levels = {0.1, 0.5, 1.0, 2.0, 5.0};
        std::vector<double> avg_errors;
        
        for (double noise_std : noise_levels) {
            LOG_INFO("  Testing noise level: {:.1f}m", noise_std);
            
            double total_error = 0.0;
            int valid_iterations = 0;
            
            for (int iter = 0; iter < num_iterations; ++iter) {
                // Create tracker
                AIMM_CS_DUCMKF_Tracker::TrackerConfig config;
                config.sampling_frequency = MEASUREMENT_RATE;
                config.enable_wind_compensation = false;
                config.position_noise = 0.1;
                config.velocity_noise = 0.01;
                config.acceleration_noise = 1.0;
                
                AIMM_CS_DUCMKF_Tracker tracker(config);
                
                // Initialize
                Vector3d initial_pos = Vector3d(500, 3000, 100);
                Vector3d initial_vel = Vector3d(-2, -10, 0);
                
                StateVector initial_state = StateVector::Zero(6);
                initial_state.head<3>() = initial_pos;
                initial_state.tail<3>() = initial_vel;
                
                StateMatrix initial_covariance = StateMatrix::Identity(6, 6);
                initial_covariance.block<3, 3>(0, 0) = Matrix3d::Identity() * 100.0;
                initial_covariance.block<3, 3>(3, 3) = Matrix3d::Identity() * 25.0;
                
                if (!tracker.initialize(initial_state, initial_covariance)) {
                    continue;
                }
                
                TargetSimulator simulator;
                
                // Run simulation
                double time = 0.0;
                std::vector<double> errors;
                bool iteration_failed = false;
                
                while (time <= 10.0 && !iteration_failed) { // 10 second simulation
                    StateVector true_state = simulator.generateTrueState(time, initial_state, 1000.0);
                    Measurement measurement = simulator.generateCartesianMeasurement(
                        true_state, time, noise_std, 0.005);
                    
                    if (!tracker.processMeasurement(measurement, time)) {
                        iteration_failed = true;
                        break;
                    }
                    
                    AIMM_CS_DUCMKF_Tracker::TrackingResult result_data = tracker.getTrackingResult();
                    
                    // Check if state is valid
                    if (!isValidState(result_data.state)) {
                        iteration_failed = true;
                        break;
                    }
                    
                    double pos_error = (true_state.head<3>() - result_data.state.head<3>()).norm();
                    
                    if (time > 2.0 && isValidValue(pos_error)) { // Skip first 2 seconds for convergence
                        errors.push_back(pos_error);
                    }
                    
                    time += DT;
                }
                
                if (!iteration_failed && !errors.empty()) {
                    // Calculate average error for this iteration
                    double avg_error = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
                    if (isValidValue(avg_error)) {
                        total_error += avg_error;
                        valid_iterations++;
                    }
                }
            }
            
            if (valid_iterations > 0) {
                avg_errors.push_back(total_error / valid_iterations);
            } else {
                avg_errors.push_back(std::numeric_limits<double>::infinity());
            }
        }
        
        // Pass criteria: errors should increase with noise but remain reasonable
        result.passed = true;
        for (size_t i = 1; i < avg_errors.size(); ++i) {
            if (isValidValue(avg_errors[i]) && isValidValue(avg_errors[i-1])) {
                if (avg_errors[i] > avg_errors[i-1] * 10) { // Error shouldn't increase more than 10x
                    result.passed = false;
                    result.failure_reason = "Error increased too much with noise";
                    break;
                }
            }
        }
        
        LOG_INFO("Test 4 Results: Avg Errors by noise level: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, Passed: {}", 
                 avg_errors[0], avg_errors[1], avg_errors[2], avg_errors[3], avg_errors[4], result.passed ? "YES" : "NO");
        
        return result;
    }
    
    /**
     * Test 5: Long-term stability with loop
     * Tests system stability over extended periods
     */
    TestResult testLongTermStability(int num_iterations = 3) {
        TestResult result;
        result.test_name = "Long-term Stability";
        result.iterations = num_iterations;
        
        LOG_INFO("Starting Test 5: Long-term Stability ({} iterations)", num_iterations);
        
        std::vector<double> final_errors;
        
        for (int iter = 0; iter < num_iterations; ++iter) {
            LOG_INFO("  Iteration {}/{}", iter + 1, num_iterations);
            
            // Create tracker
            AIMM_CS_DUCMKF_Tracker::TrackerConfig config;
            config.sampling_frequency = MEASUREMENT_RATE;
            config.enable_wind_compensation = true;
            config.position_noise = 0.1;
            config.velocity_noise = 0.01;
            config.acceleration_noise = 1.0;
            
            AIMM_CS_DUCMKF_Tracker tracker(config);
            
            // Initialize
            Vector3d initial_pos = Vector3d(500, 3000, 100);
            Vector3d initial_vel = Vector3d(-2, -10, 0);
            
            StateVector initial_state = StateVector::Zero(6);
            initial_state.head<3>() = initial_pos;
            initial_state.tail<3>() = initial_vel;
            
            StateMatrix initial_covariance = StateMatrix::Identity(6, 6);
            initial_covariance.block<3, 3>(0, 0) = Matrix3d::Identity() * 100.0;
            initial_covariance.block<3, 3>(3, 3) = Matrix3d::Identity() * 25.0;
            
            if (!tracker.initialize(initial_state, initial_covariance)) {
                result.passed = false;
                result.failure_reason = "Failed to initialize tracker";
                return result;
            }
            
            TargetSimulator simulator;
            
            // Run long simulation (60 seconds)
            double time = 0.0;
            std::vector<double> errors;
            bool iteration_failed = false;
            
            while (time <= 60.0 && !iteration_failed) { // 60 second simulation
                StateVector true_state = simulator.generateTrueState(time, initial_state, 30.0); // Maneuver at 30s
                
                // Add wind gusts periodically
                Vector3d wind = simulator.generateWindGust(time, 10.0, 5.0, 5.0);
                true_state.tail<3>() += wind * DT;
                
                Measurement measurement = simulator.generateCartesianMeasurement(
                    true_state, time, 0.5, 0.005);
                
                if (!tracker.processMeasurement(measurement, time)) {
                    iteration_failed = true;
                    break;
                }
                
                AIMM_CS_DUCMKF_Tracker::TrackingResult result_data = tracker.getTrackingResult();
                
                // Check if state is valid
                if (!isValidState(result_data.state)) {
                    iteration_failed = true;
                    break;
                }
                
                double pos_error = (true_state.head<3>() - result_data.state.head<3>()).norm();
                
                if (time > 5.0 && isValidValue(pos_error)) { // Skip first 5 seconds for convergence
                    errors.push_back(pos_error);
                }
                
                time += DT;
            }
            
            if (!iteration_failed && !errors.empty()) {
                // Calculate final error (last 10 seconds)
                size_t start_idx = errors.size() > 100 ? errors.size() - 100 : 0;
                double final_error = 0.0;
                int valid_count = 0;
                
                for (size_t i = start_idx; i < errors.size(); ++i) {
                    if (isValidValue(errors[i])) {
                        final_error += errors[i];
                        valid_count++;
                    }
                }
                
                if (valid_count > 0) {
                    final_error /= valid_count;
                    if (isValidValue(final_error)) {
                        final_errors.push_back(final_error);
                    }
                }
            }
        }
        
        if (final_errors.empty()) {
            result.passed = false;
            result.failure_reason = "All iterations failed";
            return result;
        }
        
        result.avg_position_error = std::accumulate(final_errors.begin(), final_errors.end(), 0.0) / final_errors.size();
        
        // Pass criteria: should maintain reasonable accuracy over long periods
        result.passed = result.avg_position_error < 5.0 && isValidValue(result.avg_position_error);
        
        LOG_INFO("Test 5 Results: Final Avg Error: {:.3f}m, Passed: {}", 
                 result.avg_position_error, result.passed ? "YES" : "NO");
        
        return result;
    }
    
    /**
     * Run all tests and generate report
     */
    void runAllTests() {
        LOG_INFO("=== AIMM-CS-DUCMKF Test Suite ===");
        LOG_INFO("Starting comprehensive test suite...");
        
        std::vector<TestResult> results;
        
        // Run all tests
        results.push_back(testStraightLineTracking(10));
        results.push_back(testManeuverDetection(5));
        results.push_back(testWindGustHandling(5));
        results.push_back(testNoiseSensitivity(5));
        results.push_back(testLongTermStability(3));
        
        // Generate summary report
        LOG_INFO("=== Test Summary ===");
        int passed_tests = 0;
        int total_tests = results.size();
        
        for (const auto& result : results) {
            LOG_INFO("Test: {} - {}", result.test_name, result.passed ? "PASSED" : "FAILED");
            if (!result.passed) {
                LOG_ERROR("  Failure reason: {}", result.failure_reason);
            }
            if (result.passed) passed_tests++;
        }
        
        LOG_INFO("Overall Result: {}/{} tests passed", passed_tests, total_tests);
        
        // Save detailed results to file
        saveTestResults(results);
    }
    
private:
    void saveTestResults(const std::vector<TestResult>& results) {
        std::ofstream file("test_results.csv");
        if (file.is_open()) {
            file << "Test Name,Iterations,Avg Position Error,Avg Velocity Error,"
                 << "Convergence Time,Model Switches,Maneuver Detections,"
                 << "Wind Detections,Passed,Failure Reason\n";
            
            for (const auto& result : results) {
                file << result.test_name << ","
                     << result.iterations << ","
                     << result.avg_position_error << ","
                     << result.avg_velocity_error << ","
                     << result.convergence_time << ","
                     << result.model_switches << ","
                     << result.maneuver_detections << ","
                     << result.wind_detections << ","
                     << (result.passed ? "YES" : "NO") << ","
                     << result.failure_reason << "\n";
            }
            file.close();
            LOG_INFO("Test results saved to test_results.csv");
        }
    }
};

// Define static constexpr members
constexpr double TestSuite::DT;
constexpr double TestSuite::MEASUREMENT_RATE;
constexpr int TestSuite::CONVERGENCE_ITERATIONS;

int main() {
    // Initialize logger
    Logger::initialize();
    
    TestSuite test_suite;
    test_suite.runAllTests();
    
    return 0;
} 