#ifndef LOGGER_H
#define LOGGER_H

#include <Eigen/Dense>
#include <cstdio>
#include <memory>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

using namespace Eigen;

namespace aimm_cs_ducmkf
{

    inline std::string fmt_eng(double x)
    {
        return fmt::format(
                   (std::abs(x) < 1e-2 || std::abs(x) >= 1e4) ? "{:.2e}" : "{:.2f}", x);
    }

    inline std::string fmt_mat(const MatrixXd &mat)
    {
        std::string res = "";
        for (int i = 0; i < mat.rows(); i++)
        {
            for (int j = 0; j < mat.cols(); j++)
            {
                res += fmt::format("{} ", mat(i, j));
            }
            res += "\n";
        }
        return res;
    }

    class Logger
    {
      public:
        static void initialize()
        {
            if (initialized_)
                return;
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
            logger_ = std::make_shared<spdlog::logger>("main_logger", console_sink);
            logger_->set_level(spdlog::level::debug);
            console_sink->set_color(spdlog::level::debug, console_sink->yellow);
            console_sink->set_color(spdlog::level::info, console_sink->green);
            console_sink->set_color(spdlog::level::warn, console_sink->red);
            console_sink->set_color(spdlog::level::err, console_sink->red);
            console_sink->set_color(spdlog::level::critical, console_sink->red);
            spdlog::register_logger(logger_);
            initialized_ = true;
        }

        static std::shared_ptr<spdlog::logger> get()
        {
            if (!initialized_)
                initialize();
            return logger_;
        }

      private:
        static std::shared_ptr<spdlog::logger> logger_;
        static bool initialized_;
    };

// Logging macros for convenience
#define LOG_TRACE(...) Logger::get()->trace(__VA_ARGS__)
#define LOG_DEBUG(...) Logger::get()->debug(__VA_ARGS__)
#define LOG_INFO(...) Logger::get()->info(__VA_ARGS__)
#define LOG_WARN(...) Logger::get()->warn(__VA_ARGS__)
#define LOG_ERROR(...) Logger::get()->error(__VA_ARGS__)
#define LOG_CRITICAL(...) Logger::get()->critical(__VA_ARGS__)

} // namespace aimm_cs_ducmkf

#endif // LOGGER_H