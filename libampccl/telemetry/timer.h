#ifndef AMPCCL_TELEMETRY_TIMER_H_
#define AMPCCL_TELEMETRY_TIMER_H_

#include <chrono>

// Timer backend: one of AMPCCL_USE_CUDA_TIMER or AMPCCL_USE_ACL_TIMER (CMake).
// Record start/end events on stream; Stop() only records (no sync). Call
// Synchronize() after all timers have recorded so parallel ops (e.g. fast + PCIe)
// are not serialized; then ElapsedSeconds() is valid.

#if defined(AMPCCL_USE_CUDA_TIMER)
#include <cuda_runtime.h>
#elif defined(AMPCCL_USE_ACL_TIMER)
#include <acl/acl_rt.h>
#ifndef ACL_EVENT_TIME_LINE
#define ACL_EVENT_TIME_LINE 0x00000008u
#endif
#endif

namespace ampccl {

class Timer {
public:
    Timer() : use_device_events_(false) {
#if defined(AMPCCL_USE_CUDA_TIMER)
        if (cudaEventCreate(&start_event_) == cudaSuccess &&
            cudaEventCreate(&end_event_) == cudaSuccess) {
            use_device_events_ = true;
        }
#elif defined(AMPCCL_USE_ACL_TIMER)
        if (aclrtCreateEventWithFlag(&start_event_, ACL_EVENT_TIME_LINE) == ACL_SUCCESS &&
            aclrtCreateEventWithFlag(&end_event_, ACL_EVENT_TIME_LINE) == ACL_SUCCESS) {
            use_device_events_ = true;
        }
#endif
    }

    ~Timer() {
#if defined(AMPCCL_USE_CUDA_TIMER)
        if (use_device_events_) {
            cudaEventDestroy(start_event_);
            cudaEventDestroy(end_event_);
        }
#elif defined(AMPCCL_USE_ACL_TIMER)
        if (use_device_events_) {
            aclrtDestroyEvent(start_event_);
            aclrtDestroyEvent(end_event_);
        }
#endif
    }

    // Record start event on stream (NCCL: cudaStream_t, HCCL: aclrtStream).
    // stream may be nullptr for CPU fallback.
    void Start(void* stream) {
#if defined(AMPCCL_USE_CUDA_TIMER)
        if (use_device_events_ && stream) {
            cudaEventRecord(start_event_, static_cast<cudaStream_t>(stream));
            return;
        }
#elif defined(AMPCCL_USE_ACL_TIMER)
        if (use_device_events_ && stream) {
            aclrtRecordEvent(start_event_, static_cast<aclrtStream>(stream));
            return;
        }
#endif
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    // Record end event on stream only (no sync). Call Synchronize() later so
    // ElapsedSeconds() is valid. This allows parallel timing: record both
    // timers' end events, then sync both, then read both times.
    void Stop(void* stream) {
#if defined(AMPCCL_USE_CUDA_TIMER)
        if (use_device_events_ && stream) {
            cudaEventRecord(end_event_, static_cast<cudaStream_t>(stream));
            return;
        }
#elif defined(AMPCCL_USE_ACL_TIMER)
        if (use_device_events_ && stream) {
            aclrtRecordEvent(end_event_, static_cast<aclrtStream>(stream));
            return;
        }
#endif
        end_time_ = std::chrono::high_resolution_clock::now();
    }

    // Wait for the end event so ElapsedSeconds() is valid. Call after all
    // timers have recorded their end events to preserve parallelism.
    void Synchronize() {
#if defined(AMPCCL_USE_CUDA_TIMER)
        if (use_device_events_) {
            cudaEventSynchronize(end_event_);
        }
#elif defined(AMPCCL_USE_ACL_TIMER)
        if (use_device_events_) {
            aclrtSynchronizeEvent(end_event_);
        }
#endif
    }

    double ElapsedSeconds() const {
#if defined(AMPCCL_USE_CUDA_TIMER)
        if (use_device_events_) {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start_event_, end_event_);
            return ms / 1000.0;
        }
#elif defined(AMPCCL_USE_ACL_TIMER)
        if (use_device_events_) {
            float ms = 0.0f;
            if (aclrtEventElapsedTime(&ms, start_event_, end_event_) == ACL_SUCCESS) {
                return ms / 1000.0;
            }
            return 0.0;
        }
#endif
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time_ - start_time_);
        return duration.count() / 1000000.0;
    }

    double ElapsedMilliseconds() const {
        return ElapsedSeconds() * 1000.0;
    }

private:
    bool use_device_events_;
#if defined(AMPCCL_USE_CUDA_TIMER)
    cudaEvent_t start_event_;
    cudaEvent_t end_event_;
#elif defined(AMPCCL_USE_ACL_TIMER)
    aclrtEvent start_event_;
    aclrtEvent end_event_;
#endif
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
};

}  // namespace ampccl

#endif  // AMPCCL_TELEMETRY_TIMER_H_
