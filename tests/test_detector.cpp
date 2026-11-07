#include "kinetic/detector.hpp"

#include <cmath>
#include <iostream>
#include <limits>
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
bool near(float first, float second) {
    return std::abs(first - second) < 1e-4f;
}
void require_box(const cv::Rect2f &actual, const cv::Rect2f &expected, const std::string &message) {
    require(near(actual.x, expected.x) && near(actual.y, expected.y) &&
                near(actual.width, expected.width) && near(actual.height, expected.height),
            message);
}

cv::Mat tensor(const std::vector<std::vector<float>> &rows, bool channels_first, int attributes = 6,
               int count = 12) {
    const int shape[] = {1, channels_first ? attributes : count,
                         channels_first ? count : attributes};
    cv::Mat result(3, shape, CV_32F, cv::Scalar(0));
    float *values = result.ptr<float>();
    for (std::size_t row = 0; row < rows.size(); ++row)
        for (int attribute = 0; attribute < attributes; ++attribute)
            values[channels_first ? attribute * count + row : row * attributes + attribute] =
                rows[row][attribute];
    return result;
}

void letterbox_tests() {
    const auto geometry = kinetic::make_letterbox({1280, 720}, 640);
    require(geometry.scale == .5 && geometry.left == 0 && geometry.top == 140 &&
                geometry.resized_size == cv::Size(640, 360),
            "landscape letterbox geometry");
    require_box(kinetic::inverse_letterbox({50, 240, 150, 50}, geometry), {100, 200, 300, 100},
                "inverse transform returns source coordinates");
    require_box(kinetic::inverse_letterbox({-20, 130, 80, 80}, geometry), {0, 0, 120, 140},
                "inverse geometry clips all source edges");
    require(kinetic::inverse_letterbox({-100, -100, 10, 10}, geometry).empty(),
            "fully outside box is discarded");
    const auto portrait = kinetic::make_letterbox({720, 1280}, 640);
    require(portrait.left == 140 && portrait.top == 0, "portrait letterbox padding");

    const auto odd = kinetic::make_letterbox({333, 111}, 640);
    require(odd.resized_size == cv::Size(640, 213) && odd.top == 213,
            "odd padding keeps the original rounding rule");
    const cv::Mat source(111, 333, CV_8UC3, cv::Scalar(10, 20, 30));
    const auto padded = kinetic::letterbox_image(source, odd);
    require(padded.size() == cv::Size(640, 640), "letterbox has model input dimensions");
    require(padded.at<cv::Vec3b>(0, 0) == cv::Vec3b(114, 114, 114), "letterbox fill value");
    require(padded.at<cv::Vec3b>(odd.top, 10) == cv::Vec3b(10, 20, 30),
            "letterbox retains BGR source values");
}

void tensor_layout_tests() {
    const auto geometry = kinetic::make_letterbox({1280, 720}, 640);
    const std::vector<std::vector<float>> values = {{125, 265, 150, 50, .9f, .1f}};
    for (bool channels_first : {true, false}) {
        const auto output = tensor(values, channels_first);
        const auto detections = kinetic::decode_yolov8(output, geometry);
        require(detections.size() == 1 && detections[0].class_id == 0 &&
                    near(detections[0].score, .9f),
                "auto layout decodes class and confidence");
        require_box(detections[0].box, {100, 200, 300, 100},
                    "both layouts decode identical geometry");
        // Tiny tensors have ambiguous dimensions, so the public explicit-layout
        // argument makes the intended representation unambiguous.
        const auto small = tensor(values, channels_first, 6, 1);
        const auto layout =
            channels_first ? kinetic::YoloLayout::ChannelsFirst : kinetic::YoloLayout::ChannelsLast;
        const auto decoded = kinetic::decode_yolov8(small, geometry, {}, layout);
        require(decoded.size() == 1, "explicit layout supports small fabricated tensors");
        require_box(decoded[0].box, detections[0].box, "explicit and automatic layouts agree");
    }
    const auto one_class = tensor({{125, 265, 150, 50, .8f}}, true, 5);
    require(kinetic::decode_yolov8(one_class, geometry).size() == 1,
            "single-class pest exports supported");
}

void class_aware_nms_tests() {
    const auto geometry = kinetic::make_letterbox({640, 640}, 640);
    const auto output = tensor(
        {
            {100, 100, 40, 40, .95f, .05f},
            {101, 101, 40, 40, .8f, .1f},
            {100, 100, 40, 40, .05f, .9f},
        },
        true);
    const auto detections = kinetic::decode_yolov8(output, geometry);
    require(detections.size() == 2 && detections[0].class_id == 0 && detections[1].class_id == 1,
            "different classes survive overlapping geometry");
    require(near(detections[0].score, .95f), "same-class NMS retains the strongest box");
    kinetic::DetectorConfig config;
    config.nms_threshold = 1;
    require(kinetic::decode_yolov8(output, geometry, config).size() == 3,
            "NMS threshold controls suppression");
    config = {};
    config.score_threshold = .92f;
    const auto selected = kinetic::decode_yolov8(output, geometry, config);
    require(selected.size() == 1 && near(selected[0].score, .95f),
            "score threshold removes weaker candidates");
}

void invalid_tensor_tests() {
    const auto geometry = kinetic::make_letterbox({640, 640}, 640);
    rejects([&] { kinetic::decode_yolov8(cv::Mat(), geometry); }, "empty tensor rejected");
    rejects([&] { kinetic::decode_yolov8(cv::Mat::zeros(5, 12, CV_32F), geometry); },
            "non-batched shape rejected");
    const int batch_shape[] = {2, 5, 12};
    rejects([&] { kinetic::decode_yolov8(cv::Mat(3, batch_shape, CV_32F), geometry); },
            "batch size greater than one rejected");
    const int valid_shape[] = {1, 5, 12};
    rejects([&] { kinetic::decode_yolov8(cv::Mat(3, valid_shape, CV_64F), geometry); },
            "double output rejected explicitly");
    const int short_shape[] = {1, 4, 12};
    rejects([&] { kinetic::decode_yolov8(cv::Mat(3, short_shape, CV_32F), geometry); },
            "missing class attributes rejected");
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const auto invalid = tensor(
        {
            {nan, 100, 40, 40, .9f, .1f},
            {100, 100, -2, 40, .9f, .1f},
            {100, 100, 40, 40, nan, .1f},
            {100, 100, 40, 40, 1.2f, .1f},
            {100, 100, 40, 40, .9f, .1f},
        },
        true);
    const auto decoded = kinetic::decode_yolov8(invalid, geometry);
    require(decoded.size() == 1 && near(decoded[0].score, .9f),
            "invalid geometry/confidences are discarded");
}

void invalid_configuration_tests() {
    rejects([] { kinetic::make_letterbox({0, 100}, 640); }, "empty image size rejected");
    rejects([] { kinetic::make_letterbox({100, 100}, 0); }, "zero input size rejected");
    const auto geometry = kinetic::make_letterbox({640, 640}, 640);
    const auto output = tensor({{100, 100, 40, 40, .9f, .1f}}, true);
    kinetic::DetectorConfig config;
    config.score_threshold = -1;
    rejects([&] { kinetic::decode_yolov8(output, geometry, config); },
            "negative score threshold rejected");
    config = {};
    config.nms_threshold = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { kinetic::decode_yolov8(output, geometry, config); },
            "NaN NMS threshold rejected");
    auto bad_geometry = geometry;
    bad_geometry.scale = 0;
    rejects([&] { kinetic::inverse_letterbox({0, 0, 10, 10}, bad_geometry); },
            "invalid inverse geometry rejected");
    require(kinetic::inverse_letterbox({0, 0, 0, 10}, geometry).empty(),
            "zero-sized detection discarded");
    rejects([&] { kinetic::letterbox_image(cv::Mat::zeros(640, 640, CV_8UC1), geometry); },
            "wrong image channel count rejected");
}
} // namespace

int main() {
    try {
        letterbox_tests();
        tensor_layout_tests();
        class_aware_nms_tests();
        invalid_tensor_tests();
        invalid_configuration_tests();
        std::cout << "Detector tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Detector test failure: " << error.what() << '\n';
        return 1;
    }
}
