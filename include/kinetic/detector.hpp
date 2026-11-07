#pragma once

#include "kinetic/tracker.hpp"
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

namespace kinetic {

struct DetectorConfig {
    int input_size = 640;
    float score_threshold = .05f;
    float nms_threshold = .45f;
};

struct LetterboxGeometry {
    cv::Size original_size;
    int input_size = 0;
    double scale = 0;
    cv::Size resized_size;
    int left = 0;
    int top = 0;
};

enum class YoloLayout { Auto, ChannelsFirst, ChannelsLast };

LetterboxGeometry make_letterbox(cv::Size original_size, int input_size);
cv::Mat letterbox_image(const cv::Mat &frame, const LetterboxGeometry &geometry);
// Input is an xywh rectangle in model-input pixels. Invalid/outside geometry
// returns an empty rectangle; valid results are clipped to the source image.
cv::Rect2f inverse_letterbox(const cv::Rect2f &box, const LetterboxGeometry &geometry);

// Objectness-free YOLOv8 float output. Auto keeps the normal export convention:
// the smaller of the final two dimensions contains 4 + class_count attributes.
// Choose an explicit layout for ambiguous small fabricated/custom outputs.
std::vector<Detection> decode_yolov8(const cv::Mat &output, const LetterboxGeometry &geometry,
                                     const DetectorConfig &config = {},
                                     YoloLayout layout = YoloLayout::Auto);

class YoloDetector {
  public:
    explicit YoloDetector(const std::string &model_path, DetectorConfig config = {});
    std::vector<Detection> run(const cv::Mat &frame, double &preprocess_ms, double &inference_ms,
                               double &postprocess_ms);

  private:
    DetectorConfig config_;
    cv::dnn::Net network_;
};

} // namespace kinetic
