#include "kinetic/scene.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void finite(const kinetic::SceneResult& result) {
    for (double value : {result.novelty, result.raw_novelty, result.global_change,
                         result.brightness_offset, result.translation_x, result.translation_y})
        require(std::isfinite(value), "scene diagnostics must remain finite");
    for (double value : {result.novelty, result.raw_novelty, result.global_change})
        require(value >= 0 && value <= 1, "scene fractions must remain in [0,1]");
}

kinetic::SceneConfig local_config() {
    kinetic::SceneConfig config;
    config.compensate_translation = false;
    return config;
}

cv::Mat background() {
    return cv::Mat(100, 200, CV_8UC1, cv::Scalar(100));
}

cv::Mat recurring(int frame) {
    auto image = background();
    image(cv::Rect(20, 20, 50, 60)).setTo(frame % 2 ? 145 : 100);
    return image;
}

void initialization_and_validation() {
    kinetic::SceneChangeDetector detector(local_config());
    auto frame = background();
    const auto initial = detector.observe(frame, {});
    require(initial.warming_up && !initial.reset && initial.novelty == 0, "cold start must be explicit and quiet");
    const auto stable = detector.observe(frame, {});
    require(!stable.warming_up && stable.novelty == 0 && stable.raw_novelty == 0, "unchanged pixels should be quiet");
    finite(initial); finite(stable);
    bool rejected = false;
    try { detector.observe(cv::Mat(), {}); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "empty frames must be rejected rather than called stable");
    rejected = false;
    try { detector.observe(cv::Mat(20, 20, CV_32F, cv::Scalar(1)), {}); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "non-8-bit inputs must be rejected");
    auto config = local_config(); config.adaptation_rate = 0;
    rejected = false;
    try { kinetic::SceneChangeDetector invalid(config); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid adaptation settings must be rejected");
    // Portrait input must also be bounded by the analysis edge, not just its width.
    cv::Mat portrait(1800, 300, CV_8UC3, cv::Scalar(90, 100, 110));
    kinetic::SceneChangeDetector portrait_detector;
    finite(portrait_detector.observe(portrait, {}));
    require(portrait_detector.observe(portrait, {}).novelty == 0, "portrait resizing must retain a stable causal reference");
}

void recurring_background_and_new_arrival() {
    kinetic::SceneChangeDetector detector(local_config());
    double raw = 0, filtered = 0;
    for (int frame = 0; frame < 50; ++frame) {
        const auto result = detector.observe(recurring(frame), {});
        finite(result);
        if (frame >= 40) { raw += result.raw_novelty; filtered += result.novelty; }
    }
    require(raw / 10 > 0.14, "stimulus must keep changing a substantial background patch");
    require(filtered < raw * 0.10, "prior recurring-motion envelope should reduce repeated nuisance motion");
    auto arrival = recurring(50);
    arrival(cv::Rect(120, 40, 10, 8)).setTo(20);
    const auto result = detector.observe(arrival, {});
    require(result.novelty >= 80.0 / arrival.total() * .95, "a new target in previously stable background must remain visible");
    finite(result);
}

void confirmed_mask_and_ghost_safety() {
    auto first = background();
    auto changed = first.clone(); changed(cv::Rect(90, 40, 20, 12)).setTo(20);
    kinetic::TrackResult track{7, cv::Rect2f(90, 40, 20, 12), .9f, 0, "confirmed", false};
    kinetic::SceneChangeDetector confirmed(local_config());
    confirmed.observe(first, {});
    const auto masked = confirmed.observe(changed, {track});
    require(masked.novelty == 0, "current confirmed prediction region should be masked");
    for (const auto& state : {"lost", "tentative"}) {
        kinetic::SceneChangeDetector unconfirmed(local_config()); unconfirmed.observe(first, {});
        track.state = state;
        const auto result = unconfirmed.observe(changed, {track});
        require(result.novelty > .01, "lost/tentative ghosts must not hide newly appearing pixel evidence");
    }
}

void brightness_compensation() {
    auto first = background();
    cv::RNG random(42); random.fill(first, cv::RNG::UNIFORM, 70, 130);
    cv::Mat brighter; cv::add(first, cv::Scalar(35), brighter);
    kinetic::SceneChangeDetector detector(local_config()); detector.observe(first, {});
    const auto result = detector.observe(brighter, {});
    require(result.raw_novelty > .99, "uniform brightness stimulus must change raw pixels");
    require(std::abs(result.brightness_offset - 35) < .01, "robust exposure offset should be measurable");
    require(result.novelty == 0 && result.global_change == 0, "uniform exposure shift should not be a local arrival cue");
    finite(result);
}

void translation_and_border_safety() {
    cv::Mat first(180, 320, CV_8UC1);
    cv::RNG random(157); random.fill(first, cv::RNG::UNIFORM, 30, 210);
    cv::GaussianBlur(first, first, {3, 3}, .7);
    const cv::Mat affine = (cv::Mat_<double>(2, 3) << 1, 0, 2, 0, 1, -1);
    cv::Mat shifted; cv::warpAffine(first, shifted, affine, first.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT101);
    kinetic::SceneChangeDetector detector; detector.observe(first, {});
    const auto result = detector.observe(shifted, {});
    require(result.translation_compensated, "distributed textured background should support a small camera translation");
    require(std::abs(result.translation_x-2) < .5 && std::abs(result.translation_y+1) < .5,
            "estimated translation must align previous pixels into current coordinates");
    require(result.raw_novelty > .05, "camera jitter must be a meaningful raw-difference nuisance");
    require(result.novelty < result.raw_novelty * .20, "registration should suppress jitter without counting warped borders");
    finite(result);

    // One moving foreground patch cannot justify global camera compensation.
    cv::Mat flat(120, 240, CV_8UC1, cv::Scalar(100));
    auto old_object = flat.clone(); old_object(cv::Rect(80, 50, 14, 10)).setTo(20);
    auto moving_object = flat.clone(); moving_object(cv::Rect(83, 50, 14, 10)).setTo(20);
    kinetic::SceneChangeDetector local; local.observe(old_object, {});
    const auto foreground = local.observe(moving_object, {});
    require(!foreground.translation_compensated && foreground.novelty > 0,
            "local object movement must not be explained away as camera translation");
}

void cuts_and_dimension_changes() {
    cv::Mat first(100, 200, CV_8UC1), second(first.size(), CV_8UC1);
    cv::RNG random(19); random.fill(first, cv::RNG::UNIFORM, 0, 256); random.fill(second, cv::RNG::UNIFORM, 0, 256);
    kinetic::SceneChangeDetector detector(local_config()); detector.observe(first, {});
    const auto cut = detector.observe(second, {});
    require(cut.reset && cut.warming_up && cut.novelty == 1, "strong scene cut must invalidate old histories and request refresh");
    require(detector.observe(second, {}).novelty == 0, "the new scene becomes the causal reference after a cut");
    const auto resized = detector.observe(cv::Mat(60, 100, CV_8UC1, cv::Scalar(90)), {});
    require(resized.reset && resized.warming_up && resized.novelty == 1, "dimension change must be an explicit reset");
    finite(cut); finite(resized);
}

void causality_and_component_area() {
    auto config = local_config(); config.adaptation_rate = 1;
    kinetic::SceneChangeDetector detector(config);
    auto frame = background(); detector.observe(frame, {});
    // Mutating the caller's old matrix must not mutate the stored prior. Even at
    // alpha=1, the current change must be scored before updating the envelope.
    frame(cv::Rect(120, 30, 6, 6)).setTo(20);
    const auto current = detector.observe(frame, {});
    require(current.novelty >= 36.0 / frame.total() * .99, "current evidence must not train away its own novelty");

    for (int area : {1, 2}) {
        kinetic::SceneChangeDetector tiny(local_config());
        cv::Mat base(40, 60, CV_8UC1, cv::Scalar(100)); tiny.observe(base, {});
        auto change = base.clone(); change(cv::Rect(20, 20, area, 1)).setTo(200);
        const auto result = tiny.observe(change, {});
        require(area == 1 ? result.novelty == 0 : result.novelty > 0,
                "component filtering must remove isolated noise while preserving a two-pixel component");
    }
}

void known_recurring_region_tradeoff() {
    kinetic::SceneChangeDetector detector(local_config());
    for (int frame = 0; frame < 60; ++frame) detector.observe(recurring(frame), {});
    auto change = recurring(60);
    change(cv::Rect(30, 40, 8, 8)).setTo(75); // Inside learned moving background.
    change(cv::Rect(130, 40, 8, 8)).setTo(75); // Same modest contrast in stable background.
    const auto result = detector.observe(change, {});
    require(result.novelty > 0 && result.novelty < 128.0/change.total(),
            "recurring-motion suppression can hide a target in a dynamic region; this tradeoff must remain explicit");
}

} // namespace

int main() {
    try {
        initialization_and_validation();
        recurring_background_and_new_arrival();
        confirmed_mask_and_ghost_safety();
        brightness_compensation();
        translation_and_border_safety();
        cuts_and_dimension_changes();
        causality_and_component_area();
        known_recurring_region_tradeoff();
        std::cout << "All causal scene-change tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Scene test failure: " << error.what() << '\n';
        return 1;
    }
}
