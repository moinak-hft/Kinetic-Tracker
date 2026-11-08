#include "kinetic/tracker.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
}
template <class Action> void rejects(Action action, const std::string &message) {
    bool rejected = false;
    try {
        action();
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, message);
}
kinetic::Detection detection(float x, float score = .9f, int category = 0) {
    return {{x, 0, 20, 12}, score, category};
}
kinetic::TrackerConfig motion_config() {
    kinetic::TrackerConfig config;
    config.association = "motion";
    config.min_hits = 1;
    config.max_age_seconds = 2;
    config.prediction_output_seconds = 1;
    return config;
}
const kinetic::TrackResult &identity(const std::vector<kinetic::TrackResult> &tracks, int id) {
    const auto found = std::find_if(tracks.begin(), tracks.end(),
                                    [id](const auto &track) { return track.id == id; });
    if (found == tracks.end())
        throw std::runtime_error("expected track identity absent");
    return *found;
}

void nonoverlap_and_confidence_recovery() {
    auto config = motion_config();
    kinetic::Tracker motion(config);
    config.association = "two_stage";
    kinetic::Tracker overlap(config);
    for (auto *tracker : {&motion, &overlap}) {
        tracker->predict(0);
        tracker->update({detection(0)});
        tracker->predict(.25);
    }
    const auto recovered = motion.update({detection(35)});
    const auto fragmented = overlap.update({detection(35)});
    require(recovered.size() == 1 && recovered[0].id == 1 && recovered[0].observed,
            "uncertainty gate recovers a plausible disjoint detection");
    require(fragmented.size() == 2 && !identity(fragmented, 1).observed,
            "IoU-only control fragments the disjoint example");
    motion.predict(.5);
    const auto low = motion.update({detection(60, .15f)});
    require(low.size() == 1 && low[0].id == 1 && low[0].observed &&
                std::abs(low[0].confidence - .15f) < 1e-6,
            "motion backend retains confidence-based second-stage association");
}

void implausible_class_and_ambiguity_gates() {
    for (bool wrong_class : {false, true}) {
        kinetic::Tracker tracker(motion_config());
        tracker.predict(0);
        tracker.update({detection(0)});
        tracker.predict(.25);
        const auto result =
            tracker.update({detection(wrong_class ? 1 : 400, .9f, wrong_class ? 1 : 0)});
        require(result.size() == 2 && !identity(result, 1).observed,
                "implausible or cross-class observation cannot steal an identity");
    }
    kinetic::Tracker ambiguous(motion_config());
    ambiguous.predict(0);
    ambiguous.update({detection(0)});
    ambiguous.predict(.25);
    const auto result = ambiguous.update({detection(35), detection(35)});
    require(!identity(result, 1).observed,
            "indistinguishable nonoverlap recovery candidates are rejected");

    auto config = motion_config();
    config.motion_mahalanobis_weight = 0;
    kinetic::Tracker gate_only(config);
    gate_only.predict(0);
    gate_only.update({detection(0)});
    gate_only.predict(.01);
    const auto gated = gate_only.update({detection(30)});
    require(!identity(gated, 1).observed,
            "zero Mahalanobis cost weight keeps the hard uncertainty gate");
}

void direction_cost_and_zero_weight_ablation() {
    auto directed_config = motion_config();
    directed_config.motion_gate = 100; // Isolate the direction cost inside valid gates.
    directed_config.motion_direction_weight = .4;
    kinetic::Tracker directed(directed_config);
    directed_config.motion_direction_weight = 0;
    kinetic::Tracker ablated(directed_config);
    for (auto *tracker : {&directed, &ablated}) {
        tracker->predict(0);
        tracker->update({detection(0)});
        tracker->predict(.1);
        tracker->update({detection(4)});
        tracker->predict(.2);
    }
    const std::vector<kinetic::Detection> alternatives{detection(2, .8f), detection(14, .95f)};
    const auto with_direction = directed.update(alternatives);
    const auto without_direction = ablated.update(alternatives);
    require(identity(with_direction, 1).confidence == .95f,
            "real-observation direction prefers the consistent forward candidate");
    require(identity(without_direction, 1).confidence == .8f,
            "zero direction weight exposes the controlled geometric alternative");
}

void hybrid_overlap_fallback() {
    auto strict_config = motion_config();
    require(!strict_config.motion_overlap_fallback, "strict attempt remains the default");
    auto hybrid_config = strict_config;
    hybrid_config.motion_overlap_fallback = true;
    // An abrupt observed width change preserves IoU=.5 but violates the
    // heuristic width covariance gate. It must not destroy a reliable overlap.
    for (bool size_change : {false, true}) {
        kinetic::Tracker strict(strict_config), hybrid(hybrid_config);
        for (auto* tracker : {&strict, &hybrid}) {
            tracker->predict(0);
            tracker->update({detection(0)});
            tracker->predict(.001);
        }
        auto observation = detection(13);
        if (size_change) observation.box = {-10, 0, 40, 12};
        require(kinetic::iou({0, 0, 20, 12}, observation.box) >= strict_config.iou_threshold,
                "fallback fixture has sufficient genuine IoU");
        const auto rejected = strict.update({observation});
        const auto accepted = hybrid.update({observation});
        require(!identity(rejected, 1).observed,
                "strict covariance attempt rejects the controlled innovation");
        require(accepted.size() == 1 && identity(accepted, 1).observed,
                "hybrid preserves the reliable-IoU identity despite uncertain motion calibration");
    }
    // In hybrid mode cost_limit bounds nonoverlap recovery, not reliable IoU.
    hybrid_config.motion_cost_limit = .01;
    kinetic::Tracker low_limit(hybrid_config);
    low_limit.predict(0); low_limit.update({detection(0)});
    low_limit.predict(.001);
    require(identity(low_limit.update({detection(13)}), 1).observed,
            "hybrid cost limit does not discard a reliable-overlap assignment");
    for (bool far : {false, true}) {
        kinetic::Tracker bounded(hybrid_config);
        bounded.predict(0); bounded.update({detection(0)});
        bounded.predict(.001);
        const auto result = bounded.update({detection(far ? 400 : 35)});
        require(!identity(result, 1).observed,
                "hybrid retains strict gates for implausible/nonoverlap recovery");
    }
    hybrid_config = motion_config();
    hybrid_config.motion_overlap_fallback = true;
    hybrid_config.motion_max_center_distance = .01;
    kinetic::Tracker center_bound(hybrid_config);
    center_bound.predict(0); center_bound.update({detection(0)});
    center_bound.predict(.1);
    require(!identity(center_bound.update({detection(2)}), 1).observed,
            "reliable overlap still obeys the configured center-distance safety bound");
    kinetic::Tracker classes(hybrid_config);
    classes.predict(0); classes.update({detection(0)});
    classes.predict(.1);
    require(!identity(classes.update({detection(0, .9f, 1)}), 1).observed,
            "hybrid never bypasses class consistency");
}

void crossing_trajectories() {
    kinetic::Tracker tracker(motion_config());
    tracker.predict(0);
    tracker.update({detection(0), detection(60)});
    for (int step = 1; step <= 10; ++step) {
        tracker.predict(step * .1);
        auto forward = detection(6.0f * step), backward = detection(60.0f - 6.0f * step);
        const auto result =
            tracker.update(step % 2 ? std::vector<kinetic::Detection>{backward, forward}
                                    : std::vector<kinetic::Detection>{forward, backward});
        require(result.size() == 2, "controlled crossing does not create duplicate identities");
    }
    const auto result = tracker.tracks();
    require(identity(result, 1).box.x > 55 && identity(result, 2).box.x < 5,
            "motion-direction trajectories preserve identities through crossing");
}

void post_gap_observation_velocity_correction() {
    for (bool detector_miss : {false, true}) {
        auto config = motion_config();
        kinetic::Tracker recovered(config);
        config.motion_recovery = false;
        kinetic::Tracker control(config);
        std::vector<kinetic::TrackResult> saved_prefix;
        for (auto *tracker : {&recovered, &control}) {
            tracker->predict(0);
            tracker->update({detection(0)});
            tracker->predict(.1);
            tracker->update({detection(1)});
            tracker->predict(.2);
            tracker->update({detection(2)});
            tracker->predict(.3);
            tracker->update({}, detector_miss);
            tracker->predict(.4);
            tracker->update({}, false);
        }
        saved_prefix = recovered.tracks();
        const auto immutable_prefix = saved_prefix;
        recovered.predict(.5);
        control.predict(.5);
        const auto reacquired = recovered.update({detection(20)});
        const auto normal_update = control.update({detection(20)});
        require(identity(reacquired, 1).observed && identity(normal_update, 1).observed,
                "gap is associated before recovery is applied");
        recovered.predict(.6);
        control.predict(.6);
        const double corrected_error =
            std::abs(identity(recovered.update({}, false), 1).box.x - 26);
        const double original_error = std::abs(identity(control.update({}, false), 1).box.x - 26);
        require(corrected_error < original_error,
                "endpoint velocity re-update improves the controlled post-gap secant trajectory");
        require(saved_prefix[0].box == immutable_prefix[0].box &&
                    saved_prefix[0].state == immutable_prefix[0].state,
                "new observations do not rewrite emitted history");
        require(std::isfinite(recovered.normalized_uncertainty()) &&
                    recovered.normalized_uncertainty() >= 0 &&
                    recovered.normalized_uncertainty() <= 1,
                "recovery keeps finite bounded covariance risk");
    }
}

void timestamp_and_invalid_parameters() {
    kinetic::Tracker tracker(motion_config());
    tracker.predict(0);
    const auto first = tracker.update({detection(0)});
    rejects([&] { tracker.update({detection(1)}); }, "duplicate detector timestamp rejected");
    require(tracker.tracks()[0].box == first[0].box && tracker.tracks()[0].observed,
            "duplicate rejection does not alter existing observation");
    tracker.predict(.0000000001);
    tracker.update({detection(0)}); // Tiny positive dt never divides a recovery velocity by zero.
    tracker.predict(.3);
    tracker.update({}, false);
    tracker.predict(.5);
    tracker.update({detection(2)});
    require(std::isfinite(tracker.tracks()[0].box.x),
            "small and irregular positive intervals remain finite");
    auto invalid = motion_config();
    invalid.motion_gate = 0;
    rejects([&] { kinetic::Tracker t(invalid); }, "zero motion gate rejected");
    invalid = motion_config();
    invalid.motion_gate = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { kinetic::Tracker t(invalid); }, "nonfinite gate rejected");
    invalid = motion_config();
    invalid.motion_mahalanobis_weight = 1.01;
    rejects([&] { kinetic::Tracker t(invalid); }, "invalid distance weight rejected");
    invalid = motion_config();
    invalid.motion_direction_weight = -1;
    rejects([&] { kinetic::Tracker t(invalid); }, "invalid direction weight rejected");
    invalid = motion_config();
    invalid.motion_max_center_distance = 0;
    rejects([&] { kinetic::Tracker t(invalid); }, "invalid physical distance bound rejected");
    invalid = motion_config();
    invalid.motion_cost_limit = std::numeric_limits<double>::infinity();
    rejects([&] { kinetic::Tracker t(invalid); }, "nonfinite cost bound rejected");
    invalid = motion_config();
    invalid.motion_ambiguity_margin = -.1;
    rejects([&] { kinetic::Tracker t(invalid); }, "negative ambiguity margin rejected");
}

void active_risk_scope() {
    auto config = motion_config();
    config.association = "two_stage";
    kinetic::Tracker tracker(config);
    tracker.predict(0);
    tracker.update({detection(0), detection(10)});
    tracker.predict(.1);
    tracker.update({detection(10)});
    require(tracker.normalized_uncertainty(false) > tracker.normalized_uncertainty(true),
            "active risk excludes a stale lost track's larger covariance");
    require(tracker.ambiguity(false) > 0 && tracker.ambiguity(true) == 0,
            "active ambiguity excludes lost-track overlap");

    config.min_hits = 2;
    kinetic::Tracker tentative(config);
    tentative.predict(0);
    tentative.update({detection(0), detection(10)});
    require(tentative.normalized_uncertainty(true) == 1 && tentative.ambiguity(true) == 0,
            "tentative-only pool has discovery risk and no active ambiguity");
    config.min_hits = 1;
    config.prediction_output_seconds = .05;
    kinetic::Tracker horizon(config);
    horizon.predict(0);
    horizon.update({detection(0), detection(10)});
    horizon.predict(.04);
    horizon.update({}, false);
    require(horizon.ambiguity(true) == horizon.ambiguity(false) && horizon.ambiguity(true) > 0,
            "confirmed deliberate-skip predictions remain eligible before output horizon");
    horizon.predict(.1);
    horizon.update({}, false);
    require(horizon.ambiguity(false) > 0 && horizon.ambiguity(true) == 0 &&
                horizon.normalized_uncertainty(true) == 1,
            "hidden predictions are excluded from active risk");
}

std::string serialized(const std::vector<kinetic::TrackResult>& tracks) {
    std::ostringstream out;
    out << std::setprecision(17);
    for (const auto& track : tracks)
        out << track.id << ',' << track.box.x << ',' << track.box.y << ',' << track.box.width << ','
            << track.box.height << ',' << track.confidence << ',' << track.class_id << ','
            << track.state << ',' << track.observed << '\n';
    return out.str();
}

kinetic::TrackerConfig recovery_config() {
    auto config = motion_config();
    config.association = "recovery";
    return config;
}

void establish_gap(kinetic::Tracker& tracker, bool second_track = false) {
    tracker.predict(0);
    tracker.update(second_track ? std::vector<kinetic::Detection>{detection(0), detection(25)}
                                : std::vector<kinetic::Detection>{detection(0)});
    tracker.predict(.1);
    tracker.update(second_track ? std::vector<kinetic::Detection>{detection(4), detection(29)}
                                : std::vector<kinetic::Detection>{detection(4)});
    tracker.predict(.2); tracker.update({}, false);
    tracker.predict(.3); tracker.update({}, true);
    tracker.predict(.4);
}

void third_pass_preserves_regular_baseline() {
    auto config = recovery_config();
    kinetic::Tracker recovery(config);
    config.association = "two_stage";
    kinetic::Tracker baseline(config);
    for (int frame = 0; frame < 100; ++frame) {
        const double time = frame * .04;
        std::vector<kinetic::Detection> detections;
        for (int target = 0; target < 3; ++target)
            detections.push_back(detection(target * 100 + frame * 1.5f,
                                           (frame + target) % 5 == 0 ? .15f : .9f));
        recovery.predict(time); baseline.predict(time);
        const bool ran = frame % 7 != 3;
        require(serialized(recovery.update(detections, ran)) == serialized(baseline.update(detections, ran)),
                "regular high/low IoU passes preserve baseline output bytes exactly");
        require(recovery.normalized_uncertainty() == baseline.normalized_uncertainty() &&
                recovery.ambiguity() == baseline.ambiguity(),
                "metadata maintenance does not alter baseline Kalman arithmetic");
    }
}

void third_pass_gap_and_shape_recovery() {
    auto config = recovery_config();
    kinetic::Tracker recovery(config);
    config.association = "two_stage";
    kinetic::Tracker baseline(config);
    establish_gap(recovery); establish_gap(baseline);
    auto observation = detection(30);
    observation.box.width = 30; // Bounded observed shape jitter; center gate is 2D.
    const auto recovered = recovery.update({observation});
    const auto fragmented = baseline.update({observation});
    require(recovered.size() == 1 && identity(recovered, 1).observed,
            "unmatched observed history can recover across a gap with bounded shape jitter");
    require(fragmented.size() == 2 && !identity(fragmented, 1).observed,
            "the IoU baseline cannot bridge the controlled disjoint gap");
    recovery.predict(.5);
    require(std::isfinite(identity(recovery.update({}, false), 1).box.x),
            "third-pass endpoint velocity correction leaves a valid predictive state");
}

void third_pass_cannot_steal_existing_overlap() {
    auto config = recovery_config();
    kinetic::Tracker recovery(config);
    config.association = "two_stage";
    kinetic::Tracker baseline(config);
    establish_gap(recovery, true); establish_gap(baseline, true);
    // A global weak-edge solve could prefer track1->40, track2->65 to gain
    // cardinality. The existing strong track2->40 assignment must remain frozen.
    const std::vector<kinetic::Detection> observations{detection(40, .93f), detection(65, .85f)};
    const auto result = recovery.update(observations);
    const auto reference = baseline.update(observations);
    require(identity(result, 2).observed && identity(result, 2).confidence == .93f,
            "a weak recovery edge cannot steal a committed high-IoU match");
    require(identity(result, 2).box == identity(reference, 2).box,
            "committed overlap retains exact baseline correction geometry");
}

void third_pass_rejects_unproven_or_ambiguous_births() {
    kinetic::Tracker unproven(recovery_config());
    unproven.predict(0); unproven.update({detection(0)});
    unproven.predict(.1); unproven.update({}, false);
    unproven.predict(.2);
    require(!identity(unproven.update({detection(35)}), 1).observed,
            "a track with only one real observation cannot recover a weak edge");
    for (int scenario = 0; scenario < 4; ++scenario) {
        kinetic::Tracker tracker(recovery_config());
        establish_gap(tracker);
        auto observation = detection(scenario == 0 ? 400 : 30);
        if (scenario == 1) observation.box.width = 80;
        if (scenario == 2) observation.class_id = 1;
        std::vector<kinetic::Detection> observations{observation};
        if (scenario == 3) observations.push_back(observation);
        const auto result = tracker.update(observations);
        require(!identity(result, 1).observed,
                "far, oversized, cross-class or ambiguous recovery is rejected");
        require(result.size() > 1, "unmatched high evidence remains eligible for a new birth");
    }
    kinetic::Tracker low_only(recovery_config()); establish_gap(low_only);
    const auto result = low_only.update({detection(30, .15f)});
    require(result.size() == 1 && !result[0].observed,
            "third pass never uses unmatched low-score evidence or invents a low-score birth");
    auto invalid = recovery_config(); invalid.recovery_gate = 0;
    rejects([&] { kinetic::Tracker tracker(invalid); }, "invalid 2D recovery gate rejected");
}
} // namespace

int main() {
    try {
        nonoverlap_and_confidence_recovery();
        implausible_class_and_ambiguity_gates();
        direction_cost_and_zero_weight_ablation();
        hybrid_overlap_fallback();
        crossing_trajectories();
        post_gap_observation_velocity_correction();
        timestamp_and_invalid_parameters();
        active_risk_scope();
        third_pass_preserves_regular_baseline();
        third_pass_gap_and_shape_recovery();
        third_pass_cannot_steal_existing_overlap();
        third_pass_rejects_unproven_or_ambiguous_births();
        std::cout << "Motion association tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Motion test failure: " << error.what() << '\n';
        return 1;
    }
}
