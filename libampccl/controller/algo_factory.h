#ifndef AMPCCL_CONTROLLER_ALGO_FACTORY_H_
#define AMPCCL_CONTROLLER_ALGO_FACTORY_H_

#include "algo_base.h"
#include "algo_tcp.h"
#include "algo_dcqcn.h"
#include "common/config.h"
#include "core/domain.h"
#include <memory>

namespace ampccl {

class AlgoFactory {
public:
    // Create adaptive algorithm based on environment variable and domain
    static std::unique_ptr<AdaptiveAlgo> Create(const CommDomain& domain) {
        AdaptiveAlgorithm algo_type = Config::GetAlgorithm();

        switch (algo_type) {
            case AdaptiveAlgorithm::TCP:
                return std::make_unique<TCPAlgo>();

            case AdaptiveAlgorithm::DCQCN:
                return std::make_unique<DCQCNAlgo>();

            case AdaptiveAlgorithm::STATIC:
                return std::make_unique<StaticAlgo>();

            default:
                return std::make_unique<TCPAlgo>();  // Default to TCP
        }
    }
};

// Forward declaration - will be defined in algo_static.h or here
class StaticAlgo : public AdaptiveAlgo {
public:
    StaticAlgo() : alpha_(0.5) {}

    double Suggest(const ParamValue& current) override {
        return alpha_;  // Fixed ratio
    }

    void Update(const ExecStat& stat) override {
        // Static algorithm doesn't adapt
    }

    void Reset() override {
        alpha_ = 0.5;
    }

private:
    double alpha_;
};

}  // namespace ampccl

#endif  // AMPCCL_CONTROLLER_ALGO_FACTORY_H_
