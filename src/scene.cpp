#include "kinetic/scene.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kinetic {
namespace {

double fraction(const cv::Mat& binary, const cv::Mat& eligible) {
    const int denominator = cv::countNonZero(eligible);
    if (denominator == 0) return 0.0;
    cv::Mat selected;
    cv::bitwise_and(binary, eligible, selected);
    return static_cast<double>(cv::countNonZero(selected)) / denominator;
}

double median(const cv::Mat& values, const cv::Mat& eligible) {
    // A deterministic spatial sample bounds allocation/sorting cost independently
    // of the original camera resolution. No future frame is consulted.
    const int stride = std::max(1, static_cast<int>(std::sqrt(values.total() / 16000.0)));
    std::vector<float> sample;
    sample.reserve(std::min<std::size_t>(values.total(), 18000));
    for (int y = 0; y < values.rows; y += stride) {
        const auto* row = values.ptr<float>(y);
        const auto* mask = eligible.ptr<unsigned char>(y);
        for (int x = 0; x < values.cols; x += stride)
            if (mask[x]) sample.push_back(row[x]);
    }
    if (sample.empty()) return 0.0;
    auto middle = sample.begin() + sample.size() / 2;
    std::nth_element(sample.begin(), middle, sample.end());
    return *middle;
}

cv::Mat track_mask(cv::Size analysis, cv::Size original, const std::vector<TrackResult>& tracks) {
    cv::Mat eligible(analysis, CV_8U, cv::Scalar(255));
    const double sx = static_cast<double>(analysis.width) / original.width;
    const double sy = static_cast<double>(analysis.height) / original.height;
    for (const auto& track : tracks) {
        const auto& box = track.box;
        if (track.state != "confirmed" || !std::isfinite(box.x) || !std::isfinite(box.y) ||
            !std::isfinite(box.width) || !std::isfinite(box.height) || box.width <= 0 || box.height <= 0)
            continue;
        const double padding = std::max(1.0, std::min(3.0, std::min(box.width * sx, box.height * sy) * 0.08));
        const double left = std::clamp(box.x * sx - padding, 0.0, static_cast<double>(analysis.width));
        const double top = std::clamp(box.y * sy - padding, 0.0, static_cast<double>(analysis.height));
        const double right = std::clamp((box.x + box.width) * sx + padding, 0.0, static_cast<double>(analysis.width));
        const double bottom = std::clamp((box.y + box.height) * sy + padding, 0.0, static_cast<double>(analysis.height));
        if (right > left && bottom > top) {
            const cv::Rect rectangle(static_cast<int>(std::floor(left)), static_cast<int>(std::floor(top)),
                                     static_cast<int>(std::ceil(right)) - static_cast<int>(std::floor(left)),
                                     static_cast<int>(std::ceil(bottom)) - static_cast<int>(std::floor(top)));
            eligible(rectangle).setTo(0);
        }
    }
    return eligible;
}

cv::Mat warp(const cv::Mat& input, const cv::Mat& affine, int interpolation) {
    cv::Mat result;
    cv::warpAffine(input, result, affine, input.size(), interpolation, cv::BORDER_CONSTANT, cv::Scalar(0));
    return result;
}

bool distributed_improvement(const cv::Mat& before, const cv::Mat& after, const cv::Mat& texture,
                             const cv::Mat& eligible) {
    int qualified = 0, improved = 0;
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            const cv::Rect region(column * before.cols / 4, row * before.rows / 4,
                                  (column + 1) * before.cols / 4 - column * before.cols / 4,
                                  (row + 1) * before.rows / 4 - row * before.rows / 4);
            if (region.area() < 16) continue;
            const auto mask = eligible(region);
            if (cv::countNonZero(mask) < region.area() / 2 || fraction(texture(region), mask) < 0.04)
                continue;
            const double old_error = cv::mean(before(region), mask)[0];
            if (old_error < 1.0) continue;
            ++qualified;
            if (cv::mean(after(region), mask)[0] < old_error * 0.92) ++improved;
        }
    }
    // A moving insect or one textured foreground patch must not explain an
    // apparent global camera translation. Require support across the scene.
    return qualified >= 6 && improved >= std::max(4, (qualified + 1) / 2) &&
           cv::mean(after, eligible)[0] < cv::mean(before, eligible)[0] * 0.90;
}

} // namespace

struct SceneChangeDetector::Impl {
    explicit Impl(SceneConfig configuration) : config(std::move(configuration)) {}
    SceneConfig config;
    cv::Size original_size;
    cv::Mat previous, motion_mean, motion_square, motion_frequency, window;

    void initialize(const cv::Mat& current, cv::Size original) {
        original_size = original;
        previous = current.clone();
        motion_mean = cv::Mat::zeros(current.size(), CV_32F);
        motion_square = cv::Mat::zeros(current.size(), CV_32F);
        motion_frequency = cv::Mat::zeros(current.size(), CV_32F);
        if (current.cols > 1 && current.rows > 1)
            cv::createHanningWindow(window, current.size(), CV_32F);
        else window.release();
    }
};

SceneChangeDetector::SceneChangeDetector(SceneConfig config) : impl_(std::make_unique<Impl>(config)) {
    if (config.analysis_width < 16 || config.analysis_width > 1920 ||
        !std::isfinite(config.pixel_threshold) || config.pixel_threshold <= 0 || config.pixel_threshold > 255 ||
        !std::isfinite(config.adaptation_rate) || config.adaptation_rate <= 0 || config.adaptation_rate > 1 ||
        config.min_component_area < 1 || !std::isfinite(config.max_translation) || config.max_translation < 0 ||
        !std::isfinite(config.translation_response) || config.translation_response < 0 || config.translation_response > 1)
        throw std::invalid_argument("invalid causal scene-change configuration");
}

SceneChangeDetector::~SceneChangeDetector() = default;
SceneChangeDetector::SceneChangeDetector(SceneChangeDetector&&) noexcept = default;
SceneChangeDetector& SceneChangeDetector::operator=(SceneChangeDetector&&) noexcept = default;

void SceneChangeDetector::reset() {
    const auto config = impl_->config;
    impl_ = std::make_unique<Impl>(config);
}

SceneResult SceneChangeDetector::observe(const cv::Mat& raw_frame, const std::vector<TrackResult>& tracks) {
    if (raw_frame.empty() || raw_frame.depth() != CV_8U ||
        (raw_frame.channels() != 1 && raw_frame.channels() != 3 && raw_frame.channels() != 4))
        throw std::invalid_argument("scene-change input must be a nonempty 8-bit gray, BGR or BGRA image");
    cv::Mat gray;
    if (raw_frame.channels() == 1) gray = raw_frame;
    else cv::cvtColor(raw_frame, gray, raw_frame.channels() == 3 ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY);
    if (std::max(gray.cols, gray.rows) > impl_->config.analysis_width) {
        const double scale = static_cast<double>(impl_->config.analysis_width) / std::max(gray.cols, gray.rows);
        const int width = std::max(1, static_cast<int>(std::lround(gray.cols * scale)));
        const int height = std::max(1, static_cast<int>(std::lround(gray.rows * scale)));
        cv::resize(gray, gray, {width, height}, 0, 0, cv::INTER_AREA);
    }
    cv::Mat current;
    gray.convertTo(current, CV_32F);
    SceneResult result;
    if (impl_->previous.empty()) {
        impl_->initialize(current, raw_frame.size());
        result.warming_up = true;
        return result;
    }
    if (impl_->original_size != raw_frame.size() || impl_->previous.size() != current.size()) {
        impl_->initialize(current, raw_frame.size());
        result.novelty = result.raw_novelty = result.global_change = 1.0;
        result.warming_up = result.reset = true;
        return result;
    }

    const auto& config = impl_->config;
    cv::Mat eligible = track_mask(current.size(), raw_frame.size(), tracks);
    cv::Mat valid(current.size(), CV_8U, cv::Scalar(255));
    cv::Mat raw_difference, raw_changed;
    cv::absdiff(current, impl_->previous, raw_difference);
    cv::compare(raw_difference, config.pixel_threshold, raw_changed, cv::CMP_GT);
    result.raw_novelty = fraction(raw_changed, eligible);

    cv::Mat previous = impl_->previous;
    cv::Mat motion_mean = impl_->motion_mean;
    cv::Mat motion_square = impl_->motion_square;
    cv::Mat motion_frequency = impl_->motion_frequency;
    cv::Mat signed_difference = current - previous;
    double brightness = median(signed_difference, eligible);
    cv::Mat residual;
    cv::absdiff(signed_difference, cv::Scalar(brightness), residual);
    double phase_response = 0.0;

    if (config.compensate_translation && config.max_translation > 0 && !impl_->window.empty() &&
        cv::countNonZero(eligible) >= static_cast<int>(current.total() / 4)) {
        cv::Mat gx, gy, magnitude, texture;
        cv::Sobel(previous, gx, CV_32F, 1, 0);
        cv::Sobel(previous, gy, CV_32F, 0, 1);
        cv::magnitude(gx, gy, magnitude);
        cv::compare(magnitude, 8.0, texture, cv::CMP_GT);
        if (fraction(texture, eligible) >= 0.05) {
            cv::Mat previous_phase = previous - cv::mean(previous, eligible)[0];
            cv::Mat current_phase = current - cv::mean(current, eligible)[0];
            const auto shift = cv::phaseCorrelate(previous_phase, current_phase, impl_->window, &phase_response);
            const double distance = std::hypot(shift.x, shift.y);
            if (std::isfinite(shift.x) && std::isfinite(shift.y) && std::isfinite(phase_response) &&
                phase_response >= config.translation_response && distance >= 0.15 && distance <= config.max_translation) {
                const cv::Mat affine = (cv::Mat_<double>(2, 3) << 1, 0, shift.x, 0, 1, shift.y);
                cv::Mat candidate_previous = warp(previous, affine, cv::INTER_LINEAR);
                cv::Mat candidate_valid = warp(valid, affine, cv::INTER_NEAREST);
                cv::erode(candidate_valid, candidate_valid, cv::Mat::ones(3, 3, CV_8U));
                cv::Mat candidate_eligible;
                cv::bitwise_and(eligible, candidate_valid, candidate_eligible);
                cv::Mat candidate_difference = current - candidate_previous;
                const double candidate_brightness = median(candidate_difference, candidate_eligible);
                cv::Mat candidate_residual;
                cv::absdiff(candidate_difference, cv::Scalar(candidate_brightness), candidate_residual);
                if (distributed_improvement(residual, candidate_residual, texture, candidate_eligible)) {
                    previous = candidate_previous; valid = candidate_valid; eligible = candidate_eligible;
                    signed_difference = candidate_difference; residual = candidate_residual; brightness = candidate_brightness;
                    motion_mean = warp(motion_mean, affine, cv::INTER_LINEAR);
                    motion_square = warp(motion_square, affine, cv::INTER_LINEAR);
                    motion_frequency = warp(motion_frequency, affine, cv::INTER_LINEAR);
                    result.translation_x = shift.x; result.translation_y = shift.y;
                    result.translation_compensated = true;
                }
            }
        }
    }
    result.brightness_offset = brightness;
    cv::Mat changed;
    cv::compare(residual, config.pixel_threshold, changed, cv::CMP_GT);
    result.global_change = fraction(changed, valid);

    // A strong, poorly registered scene discontinuity invalidates prior histories.
    // This deliberately requests detection rather than treating a cut as stability.
    if (result.global_change > 0.70 && cv::mean(residual, valid)[0] > 42.0 &&
        !result.translation_compensated && phase_response < 0.12) {
        impl_->initialize(current, raw_frame.size());
        result.novelty = 1.0;
        result.warming_up = result.reset = true;
        return result;
    }

    cv::Mat candidate = cv::Mat::zeros(current.size(), CV_8U);
    for (int y = 0; y < current.rows; ++y) {
        const auto* difference = residual.ptr<float>(y);
        const auto* mean = motion_mean.ptr<float>(y);
        const auto* square = motion_square.ptr<float>(y);
        const auto* frequency = motion_frequency.ptr<float>(y);
        const auto* mask = eligible.ptr<unsigned char>(y);
        auto* output = candidate.ptr<unsigned char>(y);
        for (int x = 0; x < current.cols; ++x) {
            if (!mask[x]) continue;
            double threshold = config.pixel_threshold;
            if (frequency[x] > 0.20f) {
                const double variance = std::max(0.0, static_cast<double>(square[x]) - mean[x] * mean[x]);
                threshold += 1.5 * mean[x] + 0.75 * std::sqrt(variance);
            }
            if (difference[x] > threshold) output[x] = 255;
        }
    }
    cv::Mat labels, statistics, centroids;
    const int components = cv::connectedComponentsWithStats(candidate, labels, statistics, centroids, 8, CV_32S);
    int retained_pixels = 0;
    for (int component = 1; component < components; ++component) {
        const int area = statistics.at<int>(component, cv::CC_STAT_AREA);
        if (area >= config.min_component_area) retained_pixels += area;
    }
    const int available = cv::countNonZero(eligible);
    result.novelty = available ? static_cast<double>(retained_pixels) / available : 0.0;

    // Update only after scoring against the previous model. Confirmed foreground
    // regions do not train the background envelope; tentative/lost ghosts do not
    // hide pixel evidence. One-off targets still affect later histories, a known
    // heuristic tradeoff rather than a guaranteed foreground/background decision.
    const float alpha = static_cast<float>(config.adaptation_rate);
    for (int y = 0; y < current.rows; ++y) {
        const auto* difference = residual.ptr<float>(y);
        const auto* mask = eligible.ptr<unsigned char>(y);
        auto* mean = motion_mean.ptr<float>(y);
        auto* square = motion_square.ptr<float>(y);
        auto* frequency = motion_frequency.ptr<float>(y);
        for (int x = 0; x < current.cols; ++x) {
            if (!mask[x]) continue;
            const float value = difference[x];
            mean[x] = (1-alpha)*mean[x] + alpha*value;
            square[x] = (1-alpha)*square[x] + alpha*value*value;
            frequency[x] = (1-alpha)*frequency[x] + alpha*(value > config.pixel_threshold ? 1.0f : 0.0f);
        }
    }
    impl_->motion_mean = motion_mean; impl_->motion_square = motion_square; impl_->motion_frequency = motion_frequency;
    impl_->previous = current.clone();
    return result;
}

} // namespace kinetic
