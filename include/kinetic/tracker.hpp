#pragma once

#include <opencv2/core/types.hpp>
#include <memory>
#include <string>
#include <vector>

namespace kinetic {

struct Detection {
    cv::Rect2f box;
    float score = 0.0f;
    int class_id = 0;
};

struct TrackResult {
    int id = -1;
    cv::Rect2f box;
    float confidence = 0.0f; // Last detector score, not a calibrated track probability.
    int class_id = 0;
    std::string state;
    bool observed = false;
};

struct TrackerConfig {
    std::string association = "sort";
    float high_threshold = 0.35f;
    float low_threshold = 0.08f;
    float iou_threshold = 0.2f;
    int min_hits = 2;
    double max_age_seconds = 0.75;
    double prediction_output_seconds = 0.4;
};

float iou(const cv::Rect2f& first, const cv::Rect2f& second);

// Minimum-cost maximum-cardinality finite matching. Infinity forbids a pair.
// Supports empty/rectangular matrices; unmatched rows have assignment -1.
// NaN and ragged matrices are rejected.
std::vector<int> hungarian(const std::vector<std::vector<double>>& costs);

class Tracker {
public:
    explicit Tracker(TrackerConfig config = {});
    ~Tracker();
    Tracker(Tracker&&) noexcept;
    Tracker& operator=(Tracker&&) noexcept;
    Tracker(const Tracker&) = delete;
    Tracker& operator=(const Tracker&) = delete;

    // Call once per frame with monotonic video/capture time, in seconds.
    // Expiry is measured from the last observation, even on skipped frames.
    void predict(double timestamp_seconds);
    // A deliberate detector skip does not count as evidence that a track is lost.
    std::vector<TrackResult> update(const std::vector<Detection>& detections,
                                  bool detector_ran = true);
    // Includes tentative tracks, explicitly labelled. Old predictions are hidden
    // after prediction_output_seconds, but retained internally until max_age_seconds.
    std::vector<TrackResult> tracks() const;
    double normalized_uncertainty() const;
    double ambiguity() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kinetic
