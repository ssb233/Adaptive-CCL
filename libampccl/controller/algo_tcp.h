#ifndef AMPCCL_CONTROLLER_ALGO_TCP_H_
#define AMPCCL_CONTROLLER_ALGO_TCP_H_

#include "algo_base.h"
#include "telemetry/stats.h"
#include "cache/param_cache.h"

namespace ampccl {

// TCP-style AIMD (Additive Increase Multiplicative Decrease) algorithm
class TCPAlgo : public AdaptiveAlgo {
public:
    TCPAlgo()
        : alpha_(0.5),
          increase_factor_(0.01),  // Additive increase step
          decrease_factor_(0.5),   // Multiplicative decrease factor
          min_alpha_(0.1),
          max_alpha_(0.9) {}

    double Suggest(const ParamValue& current) override {
        // Use current alpha, clamped to valid range
        alpha_ = current.alpha;
        if (alpha_ < min_alpha_) alpha_ = min_alpha_;
        if (alpha_ > max_alpha_) alpha_ = max_alpha_;
        return alpha_;
    }

    void Update(const ExecStat& stat) override {
        if (!stat.fast_success || !stat.pcie_success) {
            // If either backend failed, decrease alpha (use more fast backend)
            alpha_ *= decrease_factor_;
            if (alpha_ < min_alpha_) alpha_ = min_alpha_;
            return;
        }

        // Compare completion times
        double fast_time = stat.fast_time;
        double pcie_time = stat.pcie_time;
        double total_time = stat.GetTotalTime();

        // If PCIe is slower, decrease alpha (use more fast backend)
        if (pcie_time > fast_time * 1.1) {  // 10% threshold
            alpha_ *= decrease_factor_;
            if (alpha_ < min_alpha_) alpha_ = min_alpha_;
        } else if (pcie_time < fast_time * 0.9) {
            // If PCIe is faster, increase alpha (use more PCIe)
            alpha_ += increase_factor_;
            if (alpha_ > max_alpha_) alpha_ = max_alpha_;
        } else {
            // Balanced, slight increase
            alpha_ += increase_factor_ * 0.5;
            if (alpha_ > max_alpha_) alpha_ = max_alpha_;
        }
    }

    void Reset() override {
        alpha_ = 0.5;
    }

private:
    double alpha_;
    double increase_factor_;
    double decrease_factor_;
    double min_alpha_;
    double max_alpha_;
};

}  // namespace ampccl

#endif  // AMPCCL_CONTROLLER_ALGO_TCP_H_
