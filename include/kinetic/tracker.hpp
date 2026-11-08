#pragma once

#include <memory>
#include <opencv2/core/types.hpp>
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
    // The motion backend adds uncertainty-gated association and a real-endpoint
    // velocity correction after gaps. It is not the complete OC-SORT algorithm.
    double motion_gate = 13.2767;
    double motion_mahalanobis_weight = 0.25;
    double motion_direction_weight = 0.1;
    bool motion_recovery = true;
    // Hybrid candidate: reliable IoU plus the center bound may bypass the
    // heuristic covariance gate. motion_cost_limit then bounds nonoverlap
    // recovery only; finite weighted costs still rank overlapping pairs.
    bool motion_overlap_fallback = false;
    double motion_max_center_distance = 3.0; // Mean box diagonals.
    double motion_cost_limit = 1.0;
    double motion_ambiguity_margin = 0.03;
    // Recovery mode preserves the two-stage IoU passes first. Only remaining
    // confirmed, previously observed gap tracks use this 2D center gate.
    double recovery_gate = 9.2103;
};

float iou(const cv::Rect2f &first, const cv::Rect2f &second);

// Minimum-cost maximum-cardinality finite matching. Infinity forbids a pair.
// Supports empty/rectangular matrices; unmatched rows have assignment -1.
// NaN and ragged matrices are rejected.
std::vector<int> hungarian(const std::vector<std::vector<double>> &costs);

class Tracker {
  public:
    explicit Tracker(TrackerConfig config = {});
    ~Tracker();
    Tracker(Tracker &&) noexcept;
    Tracker &operator=(Tracker &&) noexcept;
    Tracker(const Tracker &) = delete;
    Tracker &operator=(const Tracker &) = delete;

    // Call once per frame with monotonic video/capture time, in seconds.
    // Expiry is measured from the last observation, even on skipped frames.
    void predict(double timestamp_seconds);
    // A deliberate detector skip does not count as evidence that a track is lost.
    std::vector<TrackResult> update(const std::vector<Detection> &detections,
                                    bool detector_ran = true);
    // Includes tentative tracks, explicitly labelled. Old predictions are hidden
    // after prediction_output_seconds, but retained internally until max_age_seconds.
    std::vector<TrackResult> tracks() const;
    // active_only includes confirmed, non-lost estimates inside the output
    // horizon, including deliberate skips. No eligible track gives uncertainty
    // 1 for discovery. false preserves the original all-internal-track policy.
    double normalized_uncertainty(bool active_only = false) const;
    double ambiguity(bool active_only = false) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kinetic
