#include <memory>
#include <spdlog/spdlog.h>
#include "logger.h"

using namespace aimm_cs_ducmkf;

std::shared_ptr<spdlog::logger> aimm_cs_ducmkf::Logger::logger_ = nullptr;

bool aimm_cs_ducmkf::Logger::initialized_ = false;
