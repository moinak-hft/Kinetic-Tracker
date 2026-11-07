#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace kinetic {

struct SchedulingConfig {
    std::string mode = "every";
    int interval = 4;
    int max_gap = 8;
    int min_gap = 1;
    int total_frames = 0;
    int uniform_calls = 0;
    int seed = 42;
    double risk_threshold = .18;
    double novelty_threshold = .00008;
    double target_rate = 1.0;
};

class Scheduler {
public:
    explicit Scheduler(SchedulingConfig config)
        : config_(std::move(config)), random_(config_.seed) {
        const auto& mode = config_.mode;
        if (mode != "every" && mode != "fixed" && mode != "age" && mode != "random" &&
            mode != "covariance" && mode != "risk" && mode != "adaptive" && mode != "uniform")
            throw std::invalid_argument("unsupported detector schedule");
        if (config_.interval < 1 || config_.min_gap < 1 || config_.max_gap < config_.min_gap ||
            config_.total_frames < 0 || config_.uniform_calls < 0 ||
            !std::isfinite(config_.target_rate) || config_.target_rate <= 0 || config_.target_rate > 1 ||
            !std::isfinite(config_.risk_threshold) || config_.risk_threshold < 0 ||
            !std::isfinite(config_.novelty_threshold) || config_.novelty_threshold < 0 || config_.novelty_threshold > 1)
            throw std::invalid_argument("invalid detector scheduling configuration");
        budgeted_ = mode == "covariance" || mode == "risk" || mode == "adaptive";
        if (budgeted_ && static_cast<double>(config_.max_gap) + 1e-12 < std::ceil(1.0 / config_.target_rate))
            throw std::invalid_argument("max_gap must be at least ceil(1 / target_rate)");
        if (mode == "uniform" && (config_.total_frames < 1 || config_.uniform_calls < 1 ||
                                   config_.uniform_calls > config_.total_frames))
            throw std::invalid_argument("uniform schedule requires 0 < uniform_calls <= total_frames");
    }

    // Inputs contain current/past measurements only. Every call represents the
    // next consecutive source frame; detector skips cannot hide elapsed budget.
    bool decide(int frame, double uncertainty, double ambiguity, double novelty) {
        if (static_cast<std::int64_t>(frame) != static_cast<std::int64_t>(last_frame_) + 1 ||
            (config_.total_frames > 0 && frame > config_.total_frames))
            throw std::invalid_argument("scheduler needs consecutive frame indices starting at 1");
        if (!std::isfinite(uncertainty) || !std::isfinite(ambiguity) || !std::isfinite(novelty))
            throw std::invalid_argument("scheduler signals must be finite");

        if (budgeted_ && frame > 1)
            tokens_ = std::min<long double>(2.0L, tokens_ + config_.target_rate);
        const int gap = frame - last_detection_;
        bool requested = frame == 1;
        const auto& mode = config_.mode;
        if (mode == "uniform") {
            requested = uniform_emitted_ < config_.uniform_calls && frame == uniform_frame(uniform_emitted_);
        } else if (mode == "every") requested = true;
        else if (frame > 1) {
            if (mode == "fixed" || mode == "age") requested = gap >= config_.interval;
            else if (mode == "random") requested = std::bernoulli_distribution(1.0 / config_.interval)(random_);
            else {
                const bool covariance_risk = uncertainty >= config_.risk_threshold;
                const bool association_risk = mode != "covariance" && ambiguity >= config_.risk_threshold;
                const bool new_arrival = mode == "adaptive" && novelty >= config_.novelty_threshold;
                requested = gap >= config_.max_gap ||
                            (gap >= config_.min_gap && (covariance_risk || association_risk || new_arrival));
            }
        }

        bool run = requested;
        if (budgeted_) {
            // A hard cap wins over all risk/max-gap requests. There is no token
            // borrowing. The tiny tolerance only compensates floating-point
            // sums at mathematically integral refill boundaries (e.g. 10*.1).
            const auto available_calls = static_cast<std::int64_t>(
                std::floor(1.0L + static_cast<long double>(config_.target_rate) * (frame - 1) + 1e-12L));
            run = requested && tokens_ + 1e-12L >= 1.0L && calls_ < available_calls;
            if (run) tokens_ = std::max(0.0L, tokens_ - 1.0L);
        }
        last_frame_ = frame;
        if (run) {
            last_detection_ = frame;
            ++calls_;
            if (mode == "uniform") ++uniform_emitted_;
        }
        return run;
    }

private:
    int uniform_frame(int index) const {
        if (config_.uniform_calls == 1) return 1;
        return 1 + static_cast<int>(static_cast<std::int64_t>(index) * (config_.total_frames - 1) /
                                    (config_.uniform_calls - 1));
    }

    SchedulingConfig config_;
    std::mt19937 random_;
    bool budgeted_ = false;
    int last_frame_ = 0;
    int last_detection_ = 0;
    int calls_ = 0;
    int uniform_emitted_ = 0;
    long double tokens_ = 1.0L;
};

} // namespace kinetic
