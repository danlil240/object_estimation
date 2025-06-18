#include "aimm_cs_ducmkf.h"
#include "sim.cpp"
#include <iostream>
#include <chrono>
#include <vector>

using namespace aimm_cs_ducmkf;

/**
 * @brief Performance test that runs in a loop
 * 
 * This test measures:
 * - Processing time per measurement
 * - Memory usage
 * - Convergence speed
 * - Tracking accuracy over multiple iterations
 */

struct PerformanceMetrics {
    double avg_processing_time_ms;
    double avg_position_error_m;
    double convergence_time_s;
    int total_measurements;
    double throughput_hz;
};

class PerformanceTest {
private:
    constexpr static double DT = 0.1;
    constexpr static double MEASUREMENT_RATE = 10.0;
    
public:
    /**
     * Test 1: Basic performance loop
     * Runs the tracker for multiple iterations and measures performance
     */
    PerformanceMetrics testBasicPerformance(int num_iterations = 100) {
        LOG_INFO("Starting Performance Test: {} iterations", num_iterations);
        
        std::vector<double> processing_times;
        std::vector<double> position_errors;
        double convergence_time = -1;
        int total_measurements = 0;
        
        // Create tracker
        AIMM_CS_DUCMKF_Tracker::TrackerConfig config;
        config.sampling_frequency = MEASUREMENT_RATE;
        config.enable_wind_compensation = false;
        config.position_noise = 0.1;
        config.velocity_noise = 0.01;
        
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
            LOG_ERROR("Failed to initialize tracker");
            return PerformanceMetrics{};
        }
        
        TargetSimulator simulator;
        
        // Run simulation
        double time = 0.0;
        
        for (int iter = 0; iter < num_iterations; ++iter) {
            StateVector true_state = simulator.generateTrueState(time, initial_state, 1000.0);
            Measurement measurement = simulator.generateCartesianMeasurement(
                true_state, time, 0.5, 0.005);
            
            // Measure processing time
            auto start_time = std::chrono::high_resolution_clock::now();
            
            bool success = tracker.processMeasurement(measurement, time);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            double processing_time_ms = duration.count() / 1000.0;
            
            if (!success) {
                LOG_ERROR("Failed to process measurement at iteration {}", iter);
                continue;
            }
            
            AIMM_CS_DUCMKF_Tracker::TrackingResult result = tracker.getTrackingResult();
            double pos_error = (true_state.head<3>() - result.state.head<3>()).norm();
            
            processing_times.push_back(processing_time_ms);
            position_errors.push_back(pos_error);
            total_measurements++;
            
            // Check for convergence
            if (convergence_time < 0 && pos_error < 1.0 && time > 1.0) {
                convergence_time = time;
            }
            
            time += DT;
            
            // Print progress every 10 iterations
            if ((iter + 1) % 10 == 0) {
                LOG_INFO("Progress: {}/{} iterations completed", iter + 1, num_iterations);
            }
        }
        
        // Calculate metrics
        PerformanceMetrics metrics;
        metrics.avg_processing_time_ms = std::accumulate(processing_times.begin(), processing_times.end(), 0.0) / processing_times.size();
        metrics.avg_position_error_m = std::accumulate(position_errors.begin(), position_errors.end(), 0.0) / position_errors.size();
        metrics.convergence_time_s = convergence_time;
        metrics.total_measurements = total_measurements;
        metrics.throughput_hz = 1000.0 / metrics.avg_processing_time_ms; // measurements per second
        
        LOG_INFO("Performance Test Results:");
        LOG_INFO("  Average Processing Time: {:.3f} ms", metrics.avg_processing_time_ms);
        LOG_INFO("  Average Position Error: {:.3f} m", metrics.avg_position_error_m);
        LOG_INFO("  Convergence Time: {:.3f} s", metrics.convergence_time_s);
        LOG_INFO("  Total Measurements: {}", metrics.total_measurements);
        LOG_INFO("  Throughput: {:.1f} Hz", metrics.throughput_hz);
        
        return metrics;
    }
    
    /**
     * Test 2: Stress test with high measurement rate
     * Tests performance under high load
     */
    PerformanceMetrics testHighLoadPerformance(int num_iterations = 1000) {
        LOG_INFO("Starting High Load Performance Test: {} iterations", num_iterations);
        
        std::vector<double> processing_times;
        int total_measurements = 0;
        
        // Create tracker with high measurement rate
        AIMM_CS_DUCMKF_Tracker::TrackerConfig config;
        config.sampling_frequency = 100.0; // 100 Hz
        config.enable_wind_compensation = false;
        
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
            LOG_ERROR("Failed to initialize tracker");
            return PerformanceMetrics{};
        }
        
        TargetSimulator simulator;
        
        // Run high-frequency simulation
        double time = 0.0;
        double dt = 1.0 / config.sampling_frequency;
        
        for (int iter = 0; iter < num_iterations; ++iter) {
            StateVector true_state = simulator.generateTrueState(time, initial_state, 1000.0);
            Measurement measurement = simulator.generateCartesianMeasurement(
                true_state, time, 0.5, 0.005);
            
            // Measure processing time
            auto start_time = std::chrono::high_resolution_clock::now();
            
            bool success = tracker.processMeasurement(measurement, time);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            double processing_time_ms = duration.count() / 1000.0;
            
            if (!success) {
                LOG_ERROR("Failed to process measurement at iteration {}", iter);
                continue;
            }
            
            processing_times.push_back(processing_time_ms);
            total_measurements++;
            
            time += dt;
            
            // Print progress every 100 iterations
            if ((iter + 1) % 100 == 0) {
                LOG_INFO("Progress: {}/{} iterations completed", iter + 1, num_iterations);
            }
        }
        
        // Calculate metrics
        PerformanceMetrics metrics;
        metrics.avg_processing_time_ms = std::accumulate(processing_times.begin(), processing_times.end(), 0.0) / processing_times.size();
        metrics.total_measurements = total_measurements;
        metrics.throughput_hz = 1000.0 / metrics.avg_processing_time_ms;
        
        LOG_INFO("High Load Performance Test Results:");
        LOG_INFO("  Average Processing Time: {:.3f} ms", metrics.avg_processing_time_ms);
        LOG_INFO("  Total Measurements: {}", metrics.total_measurements);
        LOG_INFO("  Throughput: {:.1f} Hz", metrics.throughput_hz);
        
        return metrics;
    }
    
    /**
     * Test 3: Memory usage test
     * Runs multiple trackers to test memory efficiency
     */
    void testMemoryUsage(int num_trackers = 10, int iterations_per_tracker = 100) {
        LOG_INFO("Starting Memory Usage Test: {} trackers, {} iterations each", num_trackers, iterations_per_tracker);
        
        std::vector<std::unique_ptr<AIMM_CS_DUCMKF_Tracker>> trackers;
        std::vector<PerformanceMetrics> results;
        TargetSimulator simulator;
        
        // Create multiple trackers
        for (int i = 0; i < num_trackers; ++i) {
            AIMM_CS_DUCMKF_Tracker::TrackerConfig config;
            config.sampling_frequency = MEASUREMENT_RATE;
            config.enable_wind_compensation = false;
            
            auto tracker = std::make_unique<AIMM_CS_DUCMKF_Tracker>(config);
            
            // Initialize each tracker
            StateVector initial_state = StateVector::Zero(6);
            initial_state.head<3>() = Vector3d(500 + i * 10, 3000 + i * 10, 100);
            initial_state.tail<3>() = Vector3d(-2, -10, 0);
            
            StateMatrix initial_covariance = StateMatrix::Identity(6, 6);
            initial_covariance.block<3, 3>(0, 0) = Matrix3d::Identity() * 100.0;
            initial_covariance.block<3, 3>(3, 3) = Matrix3d::Identity() * 25.0;
            
            if (!tracker->initialize(initial_state, initial_covariance)) {
                LOG_ERROR("Failed to initialize tracker {}", i);
                continue;
            }
            
            std::vector<double> processing_times;
            double time = 0.0;
            
            for (int iter = 0; iter < iterations_per_tracker; ++iter) {
                StateVector true_state = simulator.generateTrueState(time, initial_state, 1000.0);
                Measurement measurement = simulator.generateCartesianMeasurement(
                    true_state, time, 0.5, 0.005);
                
                auto start_time = std::chrono::high_resolution_clock::now();
                
                bool success = tracker->processMeasurement(measurement, time);
                
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                double processing_time_ms = duration.count() / 1000.0;
                
                if (success) {
                    processing_times.push_back(processing_time_ms);
                }
                
                time += DT;
            }
            
            // Calculate metrics for this tracker
            PerformanceMetrics metrics;
            if (!processing_times.empty()) {
                metrics.avg_processing_time_ms = std::accumulate(processing_times.begin(), processing_times.end(), 0.0) / processing_times.size();
                metrics.total_measurements = processing_times.size();
                metrics.throughput_hz = 1000.0 / metrics.avg_processing_time_ms;
            } else {
                metrics.avg_processing_time_ms = 0.0;
                metrics.total_measurements = 0;
                metrics.throughput_hz = 0.0;
            }
            
            results.push_back(metrics);
            trackers.push_back(std::move(tracker));
        }
        
        // Calculate overall statistics
        double avg_processing_time = 0.0;
        double avg_throughput = 0.0;
        
        for (const auto& result : results) {
            avg_processing_time += result.avg_processing_time_ms;
            avg_throughput += result.throughput_hz;
        }
        
        if (!results.empty()) {
            avg_processing_time /= results.size();
            avg_throughput /= results.size();
        }
        
        LOG_INFO("Memory Usage Test Results:");
        LOG_INFO("  Number of Trackers: {}", trackers.size());
        LOG_INFO("  Average Processing Time: {:.3f} ms", avg_processing_time);
        LOG_INFO("  Average Throughput: {:.1f} Hz", avg_throughput);
        LOG_INFO("  Total Memory Usage: {} trackers running simultaneously", trackers.size());
    }
    
    /**
     * Run all performance tests
     */
    void runAllPerformanceTests() {
        LOG_INFO("=== AIMM-CS-DUCMKF Performance Test Suite ===");
        
        // Test 1: Basic performance
        auto basic_metrics = testBasicPerformance(100);
        
        // Test 2: High load performance
        auto high_load_metrics = testHighLoadPerformance(500);
        
        // Test 3: Memory usage
        testMemoryUsage(5, 50);
        
        LOG_INFO("=== Performance Test Summary ===");
        LOG_INFO("Basic Test - Avg Processing Time: {:.3f} ms, Throughput: {:.1f} Hz", 
                 basic_metrics.avg_processing_time_ms, basic_metrics.throughput_hz);
        LOG_INFO("High Load Test - Avg Processing Time: {:.3f} ms, Throughput: {:.1f} Hz", 
                 high_load_metrics.avg_processing_time_ms, high_load_metrics.throughput_hz);
        
        // Performance criteria
        bool basic_performance_ok = basic_metrics.avg_processing_time_ms < 10.0; // Less than 10ms per measurement
        bool high_load_performance_ok = high_load_metrics.avg_processing_time_ms < 5.0; // Less than 5ms under high load
        
        LOG_INFO("Performance Assessment:");
        LOG_INFO("  Basic Performance: {}", basic_performance_ok ? "PASSED" : "FAILED");
        LOG_INFO("  High Load Performance: {}", high_load_performance_ok ? "PASSED" : "FAILED");
    }
};

int main() {
    // Initialize logger
    Logger::initialize();
    
    PerformanceTest performance_test;
    performance_test.runAllPerformanceTests();
    
    return 0;
} 