#include "kinetic/scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Action> void rejects(Action action, const std::string& message) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, message);
}

void uniform_tests() {
    for (int frames : {1, 2, 7, 31}) {
        for (int calls = 1; calls <= frames; ++calls) {
            kinetic::SchedulingConfig config;
            config.mode = "uniform";
            config.total_frames = frames;
            config.uniform_calls = calls;
            kinetic::Scheduler first(config), changed_signals(config);
            std::vector<int> selected;
            for (int frame = 1; frame <= frames; ++frame) {
                bool run = first.decide(frame, 0, 0, 0);
                require(run == changed_signals.decide(frame, frame % 2, frame % 3, frame % 5),
                        "uniform decisions must ignore content");
                if (run) selected.push_back(frame);
            }
            require(static_cast<int>(selected.size()) == calls && selected.front() == 1,
                    "uniform schedule has exact cardinality and includes frame 1");
            if (calls > 1) {
                require(selected.back() == frames, "uniform schedule covers the final frame");
                int smallest = frames, largest = 0;
                for (std::size_t index = 1; index < selected.size(); ++index) {
                    int gap = selected[index] - selected[index - 1];
                    require(gap >= 1, "uniform frames must be unique");
                    smallest = std::min(smallest, gap); largest = std::max(largest, gap);
                }
                require(largest - smallest <= 1, "uniform spacing differs by at most one frame");
            }
        }
    }
}

void hard_budget_and_maximum_gap_tests() {
    for (const std::string mode : {"covariance", "risk", "adaptive"}) {
        for (double rate : {.1, .25, .3, .5, 1.0}) {
            for (bool high_risk : {false, true}) {
                kinetic::SchedulingConfig config;
                config.mode = mode;
                config.target_rate = rate;
                config.max_gap = std::max(8, static_cast<int>(std::ceil(1.0 / rate)));
                kinetic::Scheduler scheduler(config);
                int calls = 0, last_detection = 0;
                for (int frame = 1; frame <= 160; ++frame) {
                    if (scheduler.decide(frame, high_risk ? 1 : 0, 0, 0)) {
                        ++calls;
                        require(frame == 1 || frame - last_detection <= config.max_gap,
                                "detector call spacing exceeds configured maximum gap");
                        last_detection = frame;
                    }
                    require(calls <= static_cast<int>(std::floor(1 + rate * (frame - 1) + 1e-9)),
                            "hard detector budget must hold at every prefix");
                    require(frame - last_detection < config.max_gap,
                            "fallback must acquire tokens by the maximum gap");
                    if (frame == 1) require(calls == 1, "initial frame always runs");
                }
                require(calls > 1, "fallback does not starve indefinitely");
            }
        }
    }
}

void signal_ablation_tests() {
    const auto after_first = [](const std::string& mode, double uncertainty, double ambiguity, double novelty) {
        kinetic::SchedulingConfig config; config.mode = mode;
        kinetic::Scheduler scheduler(config);
        require(scheduler.decide(1, 0, 0, 0), "first detector frame mandatory");
        return scheduler.decide(2, uncertainty, ambiguity, novelty);
    };
    require(!after_first("covariance", 0, 1, 0), "covariance-only excludes ambiguity");
    require(after_first("risk", 0, 1, 0), "risk includes association ambiguity");
    require(after_first("covariance", 1, 0, 0), "covariance responds to uncertainty");
    require(!after_first("risk", 0, 0, 1), "risk excludes image novelty");
    require(!after_first("covariance", 0, 0, 1), "covariance excludes image novelty");
    require(after_first("adaptive", 0, 0, 1), "adaptive responds to image novelty");
    require(after_first("adaptive", 0, 1, 0), "adaptive retains association risk");

    kinetic::SchedulingConfig config;
    config.mode = "adaptive"; config.min_gap = 3; config.max_gap = 8;
    kinetic::Scheduler minimum_gap(config);
    require(minimum_gap.decide(1, 1, 1, 1), "first frame ignores minimum gap");
    require(!minimum_gap.decide(2, 1, 1, 1) && !minimum_gap.decide(3, 1, 1, 1),
            "minimum gap suppresses too-early risk requests");
    require(minimum_gap.decide(4, 1, 1, 1), "risk is eligible after minimum gap");
}

void bucket_capacity_and_default_tests() {
    kinetic::SchedulingConfig config;
    config.mode = "risk"; config.target_rate = .5; config.max_gap = 20;
    kinetic::Scheduler scheduler(config);
    require(scheduler.decide(1, 0, 0, 0), "initial token is available");
    for (int frame = 2; frame <= 9; ++frame)
        require(!scheduler.decide(frame, 0, 0, 0), "low risk can save tokens");
    int burst = 0;
    for (int frame = 10; frame <= 14; ++frame)
        burst += scheduler.decide(frame, 1, 0, 0);
    require(burst == 4, "capacity-two bucket allows bounded bursts without unlimited saved credit");

    kinetic::SchedulingConfig defaults;
    kinetic::Scheduler every(defaults);
    for (int frame = 1; frame <= 12; ++frame)
        require(every.decide(frame, 0, 0, 0), "default every mode remains every frame");
    defaults.mode = "fixed"; defaults.interval = 4;
    kinetic::Scheduler fixed(defaults);
    defaults.mode = "age";
    kinetic::Scheduler alias(defaults);
    for (int frame = 1; frame <= 16; ++frame) {
        const bool expected = (frame - 1) % 4 == 0;
        require(fixed.decide(frame, 1, 1, 1) == expected, "fixed timing is content independent");
        require(alias.decide(frame, 0, 0, 0) == expected, "age is a documented fixed alias");
    }
}

void random_tests() {
    kinetic::SchedulingConfig config;
    config.mode = "random"; config.seed = 17;
    kinetic::Scheduler first(config), repeated(config);
    config.seed = 18; kinetic::Scheduler different(config);
    bool any_difference = false; int calls = 0;
    for (int frame = 1; frame <= 200; ++frame) {
        const bool run = first.decide(frame, 0, 0, 0);
        require(run == repeated.decide(frame, 1, 1, 1), "seeded random schedule is reproducible/content independent");
        any_difference = different.decide(frame, 0, 0, 0) != run || any_difference;
        calls += run;
    }
    require(any_difference && calls > 10 && calls < 190, "random control produces a nondegenerate seeded schedule");
}

void invalid_tests() {
    kinetic::SchedulingConfig config;
    config.mode = "unknown";
    rejects([&] { kinetic::Scheduler scheduler(config); }, "unknown mode rejected");
    config = {}; config.target_rate = 0;
    rejects([&] { kinetic::Scheduler scheduler(config); }, "zero target rate rejected");
    config = {}; config.target_rate = 1.1;
    rejects([&] { kinetic::Scheduler scheduler(config); }, "excess target rate rejected");
    config = {}; config.target_rate = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { kinetic::Scheduler scheduler(config); }, "NaN target rate rejected");
    config = {}; config.mode = "adaptive"; config.target_rate = .1; config.max_gap = 8;
    rejects([&] { kinetic::Scheduler scheduler(config); }, "infeasible maximum gap rejected");
    config = {}; config.mode = "uniform";
    rejects([&] { kinetic::Scheduler scheduler(config); }, "uniform requires known duration and count");
    config.total_frames = 3; config.uniform_calls = 4;
    rejects([&] { kinetic::Scheduler scheduler(config); }, "uniform count cannot exceed frames");
    config = {}; kinetic::Scheduler scheduler(config);
    rejects([&] { scheduler.decide(2, 0, 0, 0); }, "first call must be frame 1");
    require(scheduler.decide(1, 0, 0, 0), "invalid call must not mutate frame state");
    rejects([&] { scheduler.decide(1, 0, 0, 0); }, "duplicate frame rejected");
    rejects([&] { scheduler.decide(3, 0, 0, 0); }, "skipped frame index rejected");
    rejects([&] { scheduler.decide(2, 0, std::numeric_limits<double>::infinity(), 0); },
            "nonfinite signal rejected");
    require(scheduler.decide(2, 0, 0, 0), "invalid signals must not mutate schedule state");
}
} // namespace

int main() {
    try {
        uniform_tests();
        hard_budget_and_maximum_gap_tests();
        signal_ablation_tests();
        bucket_capacity_and_default_tests();
        random_tests();
        invalid_tests();
        std::cout << "Scheduler tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Scheduler test failure: " << error.what() << '\n';
        return 1;
    }
}
