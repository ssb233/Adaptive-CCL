#ifndef AMPCCL_COMMON_LOG_H_
#define AMPCCL_COMMON_LOG_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ampccl {

// Log levels: higher value = more verbose.
// Set via AMPCCL_LOG_LEVEL env (0-4 or off/error/warn/info/debug) or SetLogLevel().
enum class LogLevel : int {
    OFF = 0,
    ERROR = 1,
    WARN = 2,
    INFO = 3,
    DEBUG = 4
};

// Global log level. Can be set by env AMPCCL_LOG_LEVEL or by SetLogLevel() at runtime.
inline int& GetLogLevelRef() {
    static int level = -1;
    if (level < 0) {
        const char* env = std::getenv("AMPCCL_LOG_LEVEL");
        if (env == nullptr) {
            level = static_cast<int>(LogLevel::OFF);
        } else if (std::strcmp(env, "off") == 0 || std::strcmp(env, "0") == 0) {
            level = static_cast<int>(LogLevel::OFF);
        } else if (std::strcmp(env, "error") == 0 || std::strcmp(env, "1") == 0) {
            level = static_cast<int>(LogLevel::ERROR);
        } else if (std::strcmp(env, "warn") == 0 || std::strcmp(env, "2") == 0) {
            level = static_cast<int>(LogLevel::WARN);
        } else if (std::strcmp(env, "info") == 0 || std::strcmp(env, "3") == 0) {
            level = static_cast<int>(LogLevel::INFO);
        } else if (std::strcmp(env, "debug") == 0 || std::strcmp(env, "4") == 0) {
            level = static_cast<int>(LogLevel::DEBUG);
        } else {
            level = static_cast<int>(LogLevel::OFF);
        }
    }
    return level;
}

inline LogLevel GetLogLevel() {
    return static_cast<LogLevel>(GetLogLevelRef());
}

inline void SetLogLevel(LogLevel l) {
    GetLogLevelRef() = static_cast<int>(l);
}

inline void SetLogLevel(int l) {
    if (l < 0) l = 0;
    if (l > 4) l = 4;
    GetLogLevelRef() = l;
}

inline const char* LogLevelName(LogLevel l) {
    switch (l) {
        case LogLevel::OFF:   return "OFF";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::WARN:  return "WARN";
        case LogLevel::INFO:  return "INFO";
        case LogLevel::DEBUG: return "DEBUG";
        default: return "?";
    }
}

#define AMPCCL_LOG(level, ...) do { \
    if (static_cast<int>(ampccl::LogLevel::level) <= ampccl::GetLogLevelRef()) { \
        std::fprintf(stderr, "[AMPCCL][%s] ", ampccl::LogLevelName(ampccl::LogLevel::level)); \
        std::fprintf(stderr, __VA_ARGS__); \
        std::fprintf(stderr, "\n"); \
        std::fflush(stderr); \
    } \
} while (0)

}  // namespace ampccl

#endif  // AMPCCL_COMMON_LOG_H_
