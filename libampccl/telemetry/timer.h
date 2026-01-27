#ifndef AMPCCL_TELEMETRY_TIMER_H_
#define AMPCCL_TELEMETRY_TIMER_H_

#include <chrono>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace ampccl {

class Timer {
public:
    Timer() : start_time_(0), end_time_(0), use_cuda_(false) {
        // Try to use CUDA events if available
        #ifdef __CUDACC__
        cudaError_t err = cudaEventCreate(&start_event_);
        if (err == cudaSuccess) {
            err = cudaEventCreate(&end_event_);
            if (err == cudaSuccess) {
                use_cuda_ = true;
            }
        }
        #endif
    }

    ~Timer() {
        #ifdef __CUDACC__
        if (use_cuda_) {
            cudaEventDestroy(start_event_);
            cudaEventDestroy(end_event_);
        }
        #endif
    }

    void Start() {
        if (use_cuda_) {
            #ifdef __CUDACC__
            cudaEventRecord(start_event_);
            #endif
        } else {
            start_time_ = std::chrono::high_resolution_clock::now();
        }
    }

    void Stop() {
        if (use_cuda_) {
            #ifdef __CUDACC__
            cudaEventRecord(end_event_);
            cudaEventSynchronize(end_event_);
            #endif
        } else {
            end_time_ = std::chrono::high_resolution_clock::now();
        }
    }

    double ElapsedSeconds() const {
        if (use_cuda_) {
            #ifdef __CUDACC__
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start_event_, end_event_);
            return ms / 1000.0;
            #else
            return 0.0;
            #endif
        } else {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time_ - start_time_);
            return duration.count() / 1000000.0;
        }
    }

    double ElapsedMilliseconds() const {
        return ElapsedSeconds() * 1000.0;
    }

private:
    bool use_cuda_;
    #ifdef __CUDACC__
    cudaEvent_t start_event_;
    cudaEvent_t end_event_;
    #endif
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
};

}  // namespace ampccl

#endif  // AMPCCL_TELEMETRY_TIMER_H_
