#include "kinetic/tracker.hpp"

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
    bool failed = false;
    try { action(); } catch (const std::invalid_argument&) { failed = true; }
    require(failed, message);
}

kinetic::Detection detection(float x, float y = 0, float score = 0.9f, int class_id = 0) {
    return {cv::Rect2f(x, y, 20, 12), score, class_id};
}

kinetic::TrackerConfig immediate() {
    kinetic::TrackerConfig config;
    config.min_hits = 1;
    config.max_age_seconds = 1.0;
    config.prediction_output_seconds = 0.5;
    return config;
}

void assignment_tests() {
    const double forbidden = std::numeric_limits<double>::infinity();
    require(kinetic::hungarian({}).empty(), "empty assignment");
    require(kinetic::hungarian({{}, {}}) == std::vector<int>({-1, -1}), "zero-column assignment");
    require(kinetic::hungarian({{4, 1, 8}, {1, 5, 9}}) == std::vector<int>({1, 0}), "wide assignment");
    require(kinetic::hungarian({{0, 9}, {9, 0}, {4, 4}}) == std::vector<int>({0, 1, -1}), "tall assignment");
    require(kinetic::hungarian({{forbidden, forbidden}, {forbidden, forbidden}}) ==
            std::vector<int>({-1, -1}), "forbidden assignment remains unmatched");
    // Gating before optimization preserves both valid alternatives. A post-gate
    // solve on [[0,.69],[.69,.71]] would select the invalid diagonal and lose one.
    require(kinetic::hungarian({{0, .69}, {.69, forbidden}}) == std::vector<int>({1, 0}),
            "gated alternative assignment maximizes valid matches");
    require(kinetic::hungarian({{-4, -2}, {-1, -5}}) == std::vector<int>({0, 1}), "negative finite costs");
    rejects([] { kinetic::hungarian({{1}, {2, 3}}); }, "ragged costs rejected");
    rejects([] { kinetic::hungarian({{std::numeric_limits<double>::quiet_NaN()}}); }, "NaN costs rejected");
}

void geometry_tests() {
    require(std::abs(kinetic::iou({0, 0, 10, 10}, {0, 0, 10, 10}) - 1) < 1e-6,
            "identical boxes");
    require(kinetic::iou({0, 0, 10, 10}, {20, 20, 10, 10}) == 0, "disjoint boxes");
    require(kinetic::iou({0, 0, -1, 10}, {0, 0, 10, 10}) == 0, "invalid boxes");
    require(std::abs(kinetic::iou({0, 0, 10, 10}, {5, 0, 10, 10}) - 1.0f / 3) < 1e-6,
            "fractional IoU");
}

void confirmation_and_skips() {
    kinetic::Tracker tracker;
    tracker.predict(0);
    auto first = tracker.update({detection(0)});
    require(first.size() == 1 && first[0].state == "tentative" && first[0].observed,
            "new track is tentative");
    const int identity = first[0].id;
    tracker.predict(.05);
    auto skipped = tracker.update({}, false);
    require(skipped.size() == 1 && skipped[0].id == identity &&
            skipped[0].state == "tentative" && !skipped[0].observed,
            "skip preserves tentative track without confirming it");
    tracker.predict(.1);
    auto confirmed = tracker.update({detection(2)});
    require(confirmed.size() == 1 && confirmed[0].state == "confirmed" && confirmed[0].id == identity,
            "second observed hit confirms identity across skip");
    tracker.predict(.2);
    const auto predicted = tracker.update({}, false);
    require(predicted.size() == 1 && !predicted[0].observed && predicted[0].state == "confirmed",
            "deliberate skip is not a detector miss");
    require(predicted[0].box.x > confirmed[0].box.x, "prediction advances learned velocity");
    tracker.predict(.3);
    const auto lost = tracker.update({}, true);
    require(lost.size() == 1 && lost[0].state == "lost" && !lost[0].observed,
            "a detector miss marks a confirmed track lost");
    tracker.predict(.35);
    auto recovered = tracker.update({detection(7)});
    require(recovered.size() == 1 && recovered[0].id == identity && recovered[0].state == "confirmed",
            "lost track reactivates with original identity");
}

void timestamp_and_expiry_tests() {
    auto config = immediate();
    kinetic::Tracker tracker(config);
    tracker.predict(0);
    const auto original = tracker.update({detection(0)});
    tracker.predict(.1);
    tracker.update({detection(4)});
    tracker.predict(.2);
    const auto early = tracker.update({}, false);
    tracker.predict(.4);
    const auto later = tracker.update({}, false);
    require(early.size() == 1 && later.size() == 1 && later[0].box.x > early[0].box.x,
            "variable timestamp intervals advance motion");
    tracker.predict(.61);
    require(tracker.update({}, false).empty(), "stale predictions hidden before deletion");
    tracker.predict(.7);
    auto recovered = tracker.update({detection(28)});
    require(recovered.size() == 1 && recovered[0].id == original[0].id,
            "hidden tracks can reactivate before observation-based expiry");
    tracker.predict(1.71);
    require(tracker.update({}, false).empty(), "skipped frames still expire stale state");
    tracker.predict(1.72);
    auto replacement = tracker.update({detection(28)});
    require(replacement.size() == 1 && replacement[0].id != original[0].id,
            "expired identity is not resurrected");
    rejects([&] { tracker.predict(1.0); }, "backwards timestamp rejected");
    rejects([&] { tracker.predict(std::numeric_limits<double>::infinity()); }, "infinite timestamp rejected");
}

void dt_scaling_test() {
    auto config = immediate();
    config.prediction_output_seconds = 1;
    kinetic::Tracker short_step(config), long_step(config);
    for (auto* tracker : {&short_step, &long_step}) {
        tracker->predict(0); tracker->update({detection(0)});
        tracker->predict(.1); tracker->update({detection(4)});
    }
    const double origin = short_step.tracks()[0].box.x;
    short_step.predict(.2);
    long_step.predict(.3);
    const double first_distance = short_step.update({}, false)[0].box.x - origin;
    const double second_distance = long_step.update({}, false)[0].box.x - origin;
    require(first_distance > 0.1 && std::abs(second_distance - 2 * first_distance) < 1e-3,
            "Kalman transition uses elapsed seconds, not a fixed frame step");
    require(long_step.normalized_uncertainty() > short_step.normalized_uncertainty(),
            "longer prediction interval increases uncertainty");
}

void low_confidence_test() {
    auto sort_config = immediate();
    auto two_stage_config = sort_config;
    two_stage_config.association = "two_stage";
    kinetic::Tracker sort(sort_config), two_stage(two_stage_config);
    for (auto* tracker : {&sort, &two_stage}) {
        tracker->predict(0); tracker->update({detection(0)});
        tracker->predict(.1);
    }
    const auto sort_result = sort.update({detection(2, 0, .15f)});
    const auto recovered = two_stage.update({detection(2, 0, .15f)});
    require(sort_result.size() == 1 && !sort_result[0].observed && sort_result[0].state == "lost",
            "SORT does not use sub-high-threshold detections");
    require(recovered.size() == 1 && recovered[0].observed && recovered[0].state == "confirmed",
            "second stage recovers a confirmed track from low-score evidence");
    kinetic::Tracker no_birth(two_stage_config);
    no_birth.predict(0);
    require(no_birth.update({detection(0, 0, .15f)}).empty(), "low-score boxes do not create identities");
}

void crossing_test() {
    auto config = immediate();
    kinetic::Tracker tracker(config);
    tracker.predict(0);
    auto first = tracker.update({detection(0), detection(60)});
    const int rightward_id = first[0].id;
    const int leftward_id = first[1].id;
    for (int step = 1; step <= 10; ++step) {
        tracker.predict(step * .1);
        auto rightward = detection(step * 6.0f);
        auto leftward = detection(60 - step * 6.0f);
        const auto results = tracker.update(step % 2 ? std::vector<kinetic::Detection>{leftward, rightward}
                                                    : std::vector<kinetic::Detection>{rightward, leftward});
        require(results.size() == 2, "crossing does not spawn extra identities");
    }
    const auto results = tracker.tracks();
    const auto rightward = std::find_if(results.begin(), results.end(), [&](const auto& t) { return t.id == rightward_id; });
    const auto leftward = std::find_if(results.begin(), results.end(), [&](const auto& t) { return t.id == leftward_id; });
    require(rightward != results.end() && leftward != results.end() &&
            rightward->box.x > 55 && leftward->box.x < 5,
            "constant-velocity crossing preserves direction-linked identities");
}

void invalid_and_class_tests() {
    kinetic::Tracker tracker(immediate());
    tracker.predict(0);
    auto bad = detection(0);
    bad.box.width = 0;
    auto nan = detection(0);
    nan.box.x = std::numeric_limits<float>::quiet_NaN();
    auto score = detection(0); score.score = 1.1f;
    require(tracker.update({bad, nan, score}).empty(), "invalid detections are ignored");
    auto first = tracker.update({detection(0, 0, .9f, 1)});
    tracker.predict(.1);
    const auto other_class = tracker.update({detection(0, 0, .9f, 2)});
    require(other_class.size() == 2 && other_class[0].id == first[0].id && !other_class[0].observed &&
            other_class[1].class_id == 2, "cross-class matching is forbidden");
    require(tracker.ambiguity() == 0, "different classes do not create association ambiguity");
    require(tracker.normalized_uncertainty() >= 0 && tracker.normalized_uncertainty() <= 1,
            "normalized uncertainty is bounded");
    kinetic::Tracker overlap(immediate());
    overlap.predict(0); overlap.update({detection(0), detection(2)});
    require(overlap.ambiguity() > .7 && overlap.ambiguity() <= 1, "overlap ambiguity is exposed");
    auto invalid_config = immediate(); invalid_config.min_hits = 0;
    rejects([&] { kinetic::Tracker invalid(invalid_config); }, "invalid configuration rejected");
}
} // namespace

int main() {
    try {
        assignment_tests();
        geometry_tests();
        confirmation_and_skips();
        timestamp_and_expiry_tests();
        dt_scaling_test();
        low_confidence_test();
        crossing_test();
        invalid_and_class_tests();
        std::cout << "Tracker tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Tracker test failure: " << error.what() << '\n';
        return 1;
    }
}
