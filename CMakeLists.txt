cmake_minimum_required(VERSION 3.10)
project(AIMM_CS_DUCMKF VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Eigen package
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/test)


# Add library
add_library(aimm_cs_ducmkf
    src/ducmkf.cpp
    src/kalman_filter.cpp
    src/current_statistical_model.cpp
    src/imm_filter.cpp
    src/bandpass_filter.cpp
    src/wind_gust_handler.cpp
    src/aimm_cs_ducmkf.cpp
    src/logger.cpp
)

target_include_directories(aimm_cs_ducmkf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(aimm_cs_ducmkf_example test/main.cpp)
target_link_libraries(aimm_cs_ducmkf_example aimm_cs_ducmkf spdlog::spdlog)

# Add test suite executable
add_executable(test_suite
    test/test_suite.cpp
    src/aimm_cs_ducmkf.cpp
    src/imm_filter.cpp
    src/ducmkf.cpp
    src/kalman_filter.cpp
    src/current_statistical_model.cpp
    src/bandpass_filter.cpp
    src/wind_gust_handler.cpp
    src/logger.cpp
)

target_link_libraries(test_suite ${EIGEN3_LIBRARIES} spdlog::spdlog)
target_include_directories(test_suite PRIVATE include)

# Add performance test executable
add_executable(performance_test
    test/performance_test.cpp
    src/aimm_cs_ducmkf.cpp
    src/imm_filter.cpp
    src/ducmkf.cpp
    src/kalman_filter.cpp
    src/current_statistical_model.cpp
    src/bandpass_filter.cpp
    src/wind_gust_handler.cpp
    src/logger.cpp
)

target_link_libraries(performance_test ${EIGEN3_LIBRARIES})
target_include_directories(performance_test PRIVATE include)

# Add simple loop test executable
add_executable(simple_loop_test
    test/simple_loop_test.cpp
    src/aimm_cs_ducmkf.cpp
    src/imm_filter.cpp
    src/ducmkf.cpp
    src/kalman_filter.cpp
    src/current_statistical_model.cpp
    src/bandpass_filter.cpp
    src/wind_gust_handler.cpp
    src/logger.cpp
)

target_link_libraries(simple_loop_test ${EIGEN3_LIBRARIES})
target_include_directories(simple_loop_test PRIVATE include)




# Installation rules
#install(TARGETS aimm_cs_ducmkf
#    ARCHIVE DESTINATION lib
#    LIBRARY DESTINATION lib
#    RUNTIME DESTINATION bin
#)
#install(DIRECTORY include/ DESTINATION include/aimm_cs_ducmkf)
