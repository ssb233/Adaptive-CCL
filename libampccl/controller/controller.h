#ifndef AMPCCL_CONTROLLER_CONTROLLER_H_
#define AMPCCL_CONTROLLER_CONTROLLER_H_

#include "algo_base.h"
#include "algo_factory.h"
#include "cache/param_cache.h"
#include "telemetry/stats.h"
#include "common/op_key.h"
#include "common/config.h"
#include <memory>

namespace ampccl {

// Adaptive controller that manages algorithm and parameter cache
class AdaptiveController {
public:
    explicit AdaptiveController(std::unique_ptr<AdaptiveAlgo> algo)
        : algo_(std::move(algo)) {}

    // Get suggested alpha for an operation
    double SuggestAlpha(const OpKey& op_key, const ParamCache& cache) {
        ParamValue current = cache.Lookup(op_key);
        return algo_->Suggest(current);
    }

    // Update controller state based on execution statistics
    void Update(const OpKey& op_key, const ExecStat& stat, ParamCache& cache) {
        // Update algorithm
        algo_->Update(stat);

        // Get current parameters
        ParamValue current = cache.Lookup(op_key);

        // Update parameters based on algorithm suggestion
        double new_alpha = algo_->Suggest(current);

        // Update bandwidth estimates
        double fast_bw = stat.GetFastBandwidth();
        double pcie_bw = stat.GetPCIeBandwidth();

        // Decide whether to use PCIe
        bool use_pcie = Config::IsPCIeEnabled() &&
                       stat.pcie_success &&
                       pcie_bw > 0.0 //&&
                    //    (pcie_bw > fast_bw * 0.3);  // PCIe must be at least 30% of fast BW

        // Create updated parameter value
        ParamValue updated(new_alpha, use_pcie, fast_bw, pcie_bw);

        // Update cache
        cache.Update(op_key, updated);
    }

    void Reset() {
        algo_->Reset();
    }

private:
    std::unique_ptr<AdaptiveAlgo> algo_;
};

}  // namespace ampccl

#endif  // AMPCCL_CONTROLLER_CONTROLLER_H_
