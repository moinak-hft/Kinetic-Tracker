#include "kinetic/detector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace kinetic {
namespace {
using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void validate_config(const DetectorConfig &config) {
    if (config.input_size < 1 || !std::isfinite(config.score_threshold) ||
        !std::isfinite(config.nms_threshold) || config.score_threshold < 0 ||
        config.score_threshold > 1 || config.nms_threshold < 0 || config.nms_threshold > 1)
        throw std::invalid_argument("invalid YOLO detector configuration");
}

void validate_geometry(const LetterboxGeometry &geometry) {
    if (geometry.original_size.width <= 0 || geometry.original_size.height <= 0 ||
        geometry.input_size <= 0 || !std::isfinite(geometry.scale) || geometry.scale <= 0 ||
        geometry.resized_size.width <= 0 || geometry.resized_size.height <= 0 ||
        geometry.left < 0 || geometry.top < 0 ||
        geometry.left + geometry.resized_size.width > geometry.input_size ||
        geometry.top + geometry.resized_size.height > geometry.input_size)
        throw std::invalid_argument("invalid letterbox geometry");
}
} // namespace

LetterboxGeometry make_letterbox(cv::Size original_size, int input_size) {
    if (original_size.width <= 0 || original_size.height <= 0 || input_size <= 0)
        throw std::invalid_argument("letterbox needs positive image and input dimensions");
    const double scale = std::min(static_cast<double>(input_size) / original_size.width,
                                  static_cast<double>(input_size) / original_size.height);
    const int width = static_cast<int>(std::lround(original_size.width * scale));
    const int height = static_cast<int>(std::lround(original_size.height * scale));
    LetterboxGeometry geometry{original_size,
                               input_size,
                               scale,
                               {width, height},
                               (input_size - width) / 2,
                               (input_size - height) / 2};
    validate_geometry(geometry);
    return geometry;
}

cv::Mat letterbox_image(const cv::Mat &frame, const LetterboxGeometry &geometry) {
    validate_geometry(geometry);
    if (frame.empty() || frame.type() != CV_8UC3 || frame.size() != geometry.original_size)
        throw std::invalid_argument("letterbox expects a nonempty 8-bit BGR source image");
    cv::Mat resized, padded;
    cv::resize(frame, resized, geometry.resized_size);
    cv::copyMakeBorder(resized, padded, geometry.top,
                       geometry.input_size - geometry.resized_size.height - geometry.top,
                       geometry.left,
                       geometry.input_size - geometry.resized_size.width - geometry.left,
                       cv::BORDER_CONSTANT, {114, 114, 114});
    return padded;
}

cv::Rect2f inverse_letterbox(const cv::Rect2f &box, const LetterboxGeometry &geometry) {
    validate_geometry(geometry);
    if (!std::isfinite(box.x) || !std::isfinite(box.y) || !std::isfinite(box.width) ||
        !std::isfinite(box.height) || box.width <= 0 || box.height <= 0)
        return {};
    // Keep float subtraction followed by division by the nominal double scale,
    // matching the original inference path and its cached detection coordinates.
    const float x = static_cast<float>((box.x - geometry.left) / geometry.scale);
    const float y = static_cast<float>((box.y - geometry.top) / geometry.scale);
    const float width = static_cast<float>(box.width / geometry.scale);
    const float height = static_cast<float>(box.height / geometry.scale);
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(width) || !std::isfinite(height))
        return {};
    return cv::Rect2f(x, y, width, height) &
           cv::Rect2f(0, 0, static_cast<float>(geometry.original_size.width),
                      static_cast<float>(geometry.original_size.height));
}

std::vector<Detection> decode_yolov8(const cv::Mat &output, const LetterboxGeometry &geometry,
                                     const DetectorConfig &config, YoloLayout layout) {
    validate_geometry(geometry);
    validate_config(config);
    if (output.empty() || output.dims != 3 || output.size[0] != 1 || output.type() != CV_32F)
        throw std::invalid_argument("expected float YOLOv8 output [1,C,N] or [1,N,C]");
    const cv::Mat contiguous = output.isContinuous() ? output : output.clone();
    const bool channels_first = layout == YoloLayout::ChannelsFirst ||
                                (layout == YoloLayout::Auto && output.size[1] < output.size[2]);
    cv::Mat predictions;
    if (channels_first)
        predictions = cv::Mat(output.size[1], output.size[2], CV_32F,
                              const_cast<float *>(contiguous.ptr<float>()))
                          .t();
    else
        predictions = cv::Mat(output.size[1], output.size[2], CV_32F,
                              const_cast<float *>(contiguous.ptr<float>()));
    if (predictions.cols < 5 || predictions.cols > 1000)
        throw std::invalid_argument(
            "unsupported attributes: objectness-free YOLOv8 export required");

    std::vector<Detection> candidates;
    std::map<int, std::vector<int>> by_class;
    for (int row = 0; row < predictions.rows; ++row) {
        const float *values = predictions.ptr<float>(row);
        int class_id = 0;
        float score = values[4];
        for (int attribute = 5; attribute < predictions.cols; ++attribute) {
            if (values[attribute] > score) {
                score = values[attribute];
                class_id = attribute - 4;
            }
        }
        if (!std::isfinite(score) || score < config.score_threshold || score > 1)
            continue;
        const cv::Rect2f network_box(values[0] - .5f * values[2], values[1] - .5f * values[3],
                                     values[2], values[3]);
        const auto box = inverse_letterbox(network_box, geometry);
        if (box.width < 1 || box.height < 1)
            continue;
        by_class[class_id].push_back(static_cast<int>(candidates.size()));
        candidates.push_back({box, score, class_id});
    }

    std::vector<Detection> result;
    for (const auto &item : by_class) {
        std::vector<cv::Rect2d> boxes;
        std::vector<float> scores;
        for (const int index : item.second) {
            const auto &detection = candidates[index];
            boxes.emplace_back(detection.box.x, detection.box.y, detection.box.width,
                               detection.box.height);
            scores.push_back(detection.score);
        }
        std::vector<int> kept;
        cv::dnn::NMSBoxes(boxes, scores, config.score_threshold, config.nms_threshold, kept);
        for (const int index : kept)
            result.push_back(candidates[item.second[index]]);
    }
    return result;
}

YoloDetector::YoloDetector(const std::string &model_path, DetectorConfig config) : config_(config) {
    validate_config(config_);
    if (model_path.empty())
        return; // Cached replay does not initialize a model.
    network_ = cv::dnn::readNetFromONNX(model_path);
    network_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    network_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

std::vector<Detection> YoloDetector::run(const cv::Mat &frame, double &preprocess_ms,
                                         double &inference_ms, double &postprocess_ms) {
    if (network_.empty())
        throw std::runtime_error("detector model is not initialized");
    auto start = Clock::now();
    const auto geometry = make_letterbox(frame.size(), config_.input_size);
    const auto padded = letterbox_image(frame, geometry);
    cv::Mat blob;
    cv::dnn::blobFromImage(padded, blob, 1.0 / 255.0, {config_.input_size, config_.input_size}, {},
                           true, false);
    network_.setInput(blob);
    preprocess_ms = milliseconds(start);

    start = Clock::now();
    std::vector<cv::Mat> outputs;
    network_.forward(outputs, network_.getUnconnectedOutLayersNames());
    inference_ms = milliseconds(start);
    start = Clock::now();
    if (outputs.empty())
        throw std::runtime_error("YOLO network returned no output tensor");
    auto detections = decode_yolov8(outputs.front(), geometry, config_);
    postprocess_ms = milliseconds(start);
    return detections;
}

} // namespace kinetic
