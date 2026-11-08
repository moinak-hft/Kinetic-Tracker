#pragma once

#include "kinetic/tracker.hpp"
#include <opencv2/core/mat.hpp>
#include <memory>

namespace kinetic {

struct SceneConfig {
    int analysis_width = 640; // Maximum analysis edge; caps both axes for bounded memory.
    float pixel_threshold = 18.0f;
    double adaptation_rate = 0.08;
    int min_component_area = 2;
    bool compensate_translation = true;
    double max_translation = 6.0; // Pixels in the downsampled analysis image.
    double translation_response = 0.15;
};

struct SceneResult {
    // Fraction of eligible pixels in surviving change components. Current detector
    // outputs and ground truth are never inputs to this computation.
    double novelty = 0.0;
    // Uncompensated previous-frame difference, before the causal motion envelope.
    double raw_novelty = 0.0;
    // Corrected global changed fraction, before track masking and local adaptation.
    double global_change = 0.0;
    bool warming_up = false;
    bool reset = false;
    double brightness_offset = 0.0;
    double translation_x = 0.0;
    double translation_y = 0.0;
    bool translation_compensated = false;
};

// Training-free, causal scene-change heuristic. Recurring background motion is
// learned only from preceding frame differences. It cannot guarantee discovery:
// targets entering repeatedly moving regions can also be suppressed, and broad
// nonuniform illumination changes can still trigger detection.
class SceneChangeDetector {
public:
    explicit SceneChangeDetector(SceneConfig config = {});
    ~SceneChangeDetector();
    SceneChangeDetector(SceneChangeDetector&&) noexcept;
    SceneChangeDetector& operator=(SceneChangeDetector&&) noexcept;
    SceneChangeDetector(const SceneChangeDetector&) = delete;
    SceneChangeDetector& operator=(const SceneChangeDetector&) = delete;

    // Call before the current detector decision, with raw pixels and predictions
    // based on earlier observations. Only state="confirmed" regions are masked.
    // First frame returns novelty=0,warming_up=true. A dimension change or strong
    // scene cut explicitly resets the model and returns novelty=1,reset=true so a
    // scheduler can request a fresh detection subject to its compute budget.
    // Empty/non-8-bit/unsupported-channel images are rejected.
    SceneResult observe(const cv::Mat& raw_frame, const std::vector<TrackResult>& tracks);
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kinetic
