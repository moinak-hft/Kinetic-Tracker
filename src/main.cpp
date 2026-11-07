#include "kinetic/detector.hpp"
#include "kinetic/scheduler.hpp"
#include "kinetic/tracker.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
static std::vector<std::string> split(const std::string &s) {
    std::vector<std::string> v;
    std::stringstream in(s);
    std::string x;
    while (std::getline(in, x, ','))
        v.push_back(x);
    return v;
}
static std::string escaped(const std::string &s) {
    std::string out;
    for (char c : s) {
        if (c == '\\' || c == '"')
            out += '\\';
        if (c == '\n')
            out += "\\n";
        else if (c != '\r')
            out += c;
    }
    return out;
}
static std::string canonical_path(const fs::path &path) {
    auto value = fs::weakly_canonical(fs::absolute(path)).generic_string();
#ifdef _WIN32
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return std::tolower(c); });
#endif
    return value;
}
static void guard_outputs(const std::vector<fs::path> &inputs, const fs::path &directory,
                          const std::string &video) {
    std::set<std::string> writes;
    for (const auto &name :
         {"mot.txt", "telemetry.csv", "frames.csv", "detections.csv", "summary.json"})
        writes.insert(canonical_path(directory / name));
    if (!video.empty() && !writes.insert(canonical_path(video)).second)
        throw std::runtime_error("output video collides with a result file");
    for (const auto &path : inputs)
        if (!path.empty() && writes.count(canonical_path(path)))
            throw std::runtime_error("output would overwrite input: " + path.string());
}
struct Options {
    std::string video, images, model, detections, frame_times,
        output = "runs/demo", schedule = "every", association = "sort", output_video,
        output_policy = "active";
    int frames = 0, interval = 4, max_gap = 8, min_gap = 1, min_hits = 2, input_size = 640,
        threads = 2, seed = 42, uniform_calls = 0;
    double fps = 25, risk_threshold = .18, novelty_threshold = .00008, max_age = .75,
           prediction_age = .4, target_rate = 1;
    float high = .35f, low = .08f, iou = .2f, score = .05f, nms = .45f;
    bool preview = false;
};
static Options options(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help") {
            std::cout
                << "Kinetic-Tracker 2: native C++ data-cache and live-video tracking\n"
                   "Inputs: --video file OR --images dir; --model yolo-v8.onnx for inference\n"
                   "Replay: --detections detections.csv --frames N --fps 25 [--images dir]\n"
                   "        [--frame-times frames.csv] preserves source sampling intervals\n"
                   "Output: --output directory [--output-video annotated.mp4] [--preview]\n"
                   "        --output-policy active|all (default active suppresses unmatched lost-track boxes)\n"
                   "Policy: --association sort|two_stage --schedule "
                   "every|fixed|random|age|covariance|risk|adaptive|uniform\n"
                   "        --interval 4 --max-gap 8 --min-gap 1 --risk-threshold .18 "
                   "--novelty-threshold .00008\n"
                   "        --target-rate .5 caps adaptive detector calls; --uniform-calls K "
                   "selects K equally spaced frames\n"
                   "Tracker: --high .35 --low .08 --iou .2 --min-hits 2 --max-age .75 "
                   "--prediction-age .4\n"
                   "Detector: --input-size 640 --score .05 --nms .45 --threads 2 --seed 42\n"
                   "Cached replay timings do not measure detector inference. No GPU is required.\n";
            std::exit(0);
        }
        if (a == "--preview") {
            o.preview = true;
            continue;
        }
        if (i + 1 == argc)
            throw std::runtime_error("missing value for " + a);
        std::string v = argv[++i];
        if (a == "--video")
            o.video = v;
        else if (a == "--images")
            o.images = v;
        else if (a == "--model")
            o.model = v;
        else if (a == "--detections")
            o.detections = v;
        else if (a == "--frame-times")
            o.frame_times = v;
        else if (a == "--output")
            o.output = v;
        else if (a == "--output-video")
            o.output_video = v;
        else if (a == "--schedule")
            o.schedule = v;
        else if (a == "--association")
            o.association = v;
        else if (a == "--output-policy")
            o.output_policy = v;
        else if (a == "--frames")
            o.frames = std::stoi(v);
        else if (a == "--fps")
            o.fps = std::stod(v);
        else if (a == "--interval")
            o.interval = std::stoi(v);
        else if (a == "--max-gap")
            o.max_gap = std::stoi(v);
        else if (a == "--min-gap")
            o.min_gap = std::stoi(v);
        else if (a == "--risk-threshold")
            o.risk_threshold = std::stod(v);
        else if (a == "--novelty-threshold")
            o.novelty_threshold = std::stod(v);
        else if (a == "--target-rate")
            o.target_rate = std::stod(v);
        else if (a == "--uniform-calls")
            o.uniform_calls = std::stoi(v);
        else if (a == "--min-hits")
            o.min_hits = std::stoi(v);
        else if (a == "--high")
            o.high = std::stof(v);
        else if (a == "--low")
            o.low = std::stof(v);
        else if (a == "--iou")
            o.iou = std::stof(v);
        else if (a == "--max-age")
            o.max_age = std::stod(v);
        else if (a == "--prediction-age")
            o.prediction_age = std::stod(v);
        else if (a == "--input-size")
            o.input_size = std::stoi(v);
        else if (a == "--score")
            o.score = std::stof(v);
        else if (a == "--nms")
            o.nms = std::stof(v);
        else if (a == "--threads")
            o.threads = std::stoi(v);
        else if (a == "--seed")
            o.seed = std::stoi(v);
        else
            throw std::runtime_error("unknown option: " + a);
    }
    const std::set<std::string> policies = {"every",      "fixed", "random",   "age",
                                            "covariance", "risk",  "adaptive", "uniform"};
    if (!policies.count(o.schedule))
        throw std::runtime_error("invalid schedule");
    if (o.association != "sort" && o.association != "two_stage")
        throw std::runtime_error("invalid association");
    if (o.output_policy != "all" && o.output_policy != "active")
        throw std::runtime_error("invalid output policy");
    if (o.frames < 0 || !std::isfinite(o.fps) || o.fps <= 0 || o.interval < 1 || o.min_gap < 1 ||
        o.max_gap < o.min_gap || o.input_size < 32 || o.threads < 1 || o.min_hits < 1)
        throw std::runtime_error("invalid positive configuration");
    if (!std::isfinite(o.risk_threshold) || o.risk_threshold < 0 ||
        !std::isfinite(o.novelty_threshold) || o.novelty_threshold < 0 || o.novelty_threshold > 1)
        throw std::runtime_error("invalid scheduler threshold");
    for (float x : {o.high, o.low, o.iou, o.score, o.nms})
        if (!std::isfinite(x) || x < 0 || x > 1)
            throw std::runtime_error("threshold must be in [0,1]");
    if (o.low > o.high || o.max_age <= 0 || o.prediction_age < 0 || !std::isfinite(o.max_age) ||
        !std::isfinite(o.prediction_age))
        throw std::runtime_error("invalid tracker configuration");
    if (!o.video.empty() && !o.images.empty())
        throw std::runtime_error("choose video or images");
    if (o.detections.empty() && (o.model.empty() || (o.video.empty() && o.images.empty())))
        throw std::runtime_error("inference needs --model and --video/--images");
    if (o.schedule == "adaptive" && o.video.empty() && o.images.empty())
        throw std::runtime_error(
            "adaptive new-arrival signal requires real input frames, not detection rows");
    if ((o.preview || !o.output_video.empty()) && o.video.empty() && o.images.empty())
        throw std::runtime_error("visualization needs input frames");
    return o;
}
struct Cache {
    std::map<int, std::vector<kinetic::Detection>> rows;
    std::map<int, double> times;
    int max_frame = 0;
};
static Cache read_cache(const std::string &path) {
    Cache c;
    if (path.empty())
        return c;
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("cannot read detection cache: " + path);
    std::string line;
    std::getline(f, line);
    if (line.find("frame,timestamp,x,y,w,h,score,class_id") != 0)
        throw std::runtime_error("invalid detection CSV header");
    while (std::getline(f, line)) {
        if (line.empty())
            continue;
        auto a = split(line);
        if (a.size() != 8)
            throw std::runtime_error("invalid detection row");
        int n = std::stoi(a[0]);
        double t = std::stod(a[1]);
        kinetic::Detection d{
            cv::Rect2f(std::stof(a[2]), std::stof(a[3]), std::stof(a[4]), std::stof(a[5])),
            std::stof(a[6]), std::stoi(a[7])};
        if (n < 1 || !std::isfinite(t) || d.box.width <= 0 || d.box.height <= 0 ||
            !std::isfinite(d.box.x) || !std::isfinite(d.box.y) || !std::isfinite(d.box.width) ||
            !std::isfinite(d.box.height) || !std::isfinite(d.score) || d.score < 0 || d.score > 1)
            throw std::runtime_error("invalid detection values");
        if (c.times.count(n) && std::abs(c.times[n] - t) > 1e-6)
            throw std::runtime_error("inconsistent cache timestamps");
        c.rows[n].push_back(d);
        c.times[n] = t;
        c.max_frame = std::max(c.max_frame, n);
    }
    return c;
}
static std::map<int, double> read_times(const std::string &path) {
    std::map<int, double> out;
    if (path.empty())
        return out;
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("cannot read frame-times file");
    std::string line;
    std::getline(f, line);
    auto h = split(line);
    int fi = -1, ti = -1;
    for (int i = 0; i < (int)h.size(); ++i) {
        if (h[i] == "frame")
            fi = i;
        if (h[i] == "timestamp" || h[i] == "timestamp_seconds")
            ti = i;
    }
    if (fi < 0 || ti < 0)
        throw std::runtime_error("frame-times needs frame,timestamp columns");
    while (std::getline(f, line)) {
        if (line.empty())
            continue;
        auto a = split(line);
        if ((int)a.size() <= std::max(fi, ti))
            throw std::runtime_error("invalid frame-time row");
        int n = std::stoi(a[fi]);
        double t = std::stod(a[ti]);
        if (n < 1 || !std::isfinite(t) || out.count(n))
            throw std::runtime_error("invalid/duplicate frame timestamp");
        out[n] = t;
    }
    return out;
}

static double scene_novelty(const cv::Mat &frame, cv::Mat &previous,
                            const std::vector<kinetic::TrackResult> &tracks) {
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, {3, 3}, 0);
    if (previous.empty()) {
        previous = gray;
        return 0;
    }
    cv::Mat diff, mask(frame.size(), CV_8U, cv::Scalar(255));
    cv::absdiff(previous, gray, diff);
    cv::threshold(diff, diff, 22, 255, cv::THRESH_BINARY);
    for (const auto &t : tracks) {
        if (t.state == "tentative")
            continue;
        const auto &b = t.box;
        int pad = std::max(3, (int)std::ceil(.25 * std::max(b.width, b.height)));
        cv::Rect r((int)std::floor(b.x) - pad, (int)std::floor(b.y) - pad,
                   (int)std::ceil(b.width) + 2 * pad, (int)std::ceil(b.height) + 2 * pad);
        r &= cv::Rect(0, 0, frame.cols, frame.rows);
        if (r.area() > 0)
            mask(r).setTo(0);
    }
    cv::bitwise_and(diff, mask, diff);
    previous = gray;
    return (double)cv::countNonZero(diff) / frame.total();
}
static double percentile(std::vector<double> v, double p) {
    if (v.empty())
        return 0;
    std::sort(v.begin(), v.end());
    return v[(size_t)std::floor(p * (v.size() - 1))];
}
int main(int argc, char **argv) {
    try {
        auto startup = Clock::now();
        Options o = options(argc, argv);
        cv::setNumThreads(o.threads);
        Cache cache = read_cache(o.detections);
        auto times = read_times(o.frame_times);
        cv::VideoCapture capture;
        std::vector<fs::path> images;
        if (!o.video.empty()) {
            capture.open(o.video);
            if (!capture.isOpened())
                throw std::runtime_error("cannot open video");
            double f = capture.get(cv::CAP_PROP_FPS);
            if (std::isfinite(f) && f > 0)
                o.fps = f;
        }
        if (!o.images.empty()) {
            for (const auto &p : fs::directory_iterator(o.images)) {
                auto e = p.path().extension().string();
                std::transform(e.begin(), e.end(), e.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                if (e == ".jpg" || e == ".jpeg" || e == ".png")
                    images.push_back(p.path());
            }
            std::sort(images.begin(), images.end());
            if (images.empty())
                throw std::runtime_error("image directory is empty");
            std::set<std::string> stems;
            for (const auto &path : images) {
                std::string stem = path.stem().string();
                std::transform(stem.begin(), stem.end(), stem.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                if (!stems.insert(stem).second)
                    throw std::runtime_error("duplicate image frame stem: " + stem);
            }
            if (!o.frames)
                o.frames = (int)images.size();
            else
                o.frames = std::min(o.frames, (int)images.size());
        }
        if (o.frames == 0 && capture.isOpened())
            o.frames = (int)capture.get(cv::CAP_PROP_FRAME_COUNT);
        if (o.frames == 0 && !times.empty())
            o.frames = times.rbegin()->first;
        if (o.frames == 0)
            o.frames = cache.max_frame;
        if (o.frames <= 0)
            throw std::runtime_error("frame count unknown; supply --frames");
        std::vector<fs::path> input_paths = images;
        for (const auto &path : {o.video, o.model, o.detections, o.frame_times})
            if (!path.empty())
                input_paths.emplace_back(path);
        guard_outputs(input_paths, o.output, o.output_video);
        double cache_origin = cache.times.empty() ? 0
                                                  : cache.times.begin()->second -
                                                        (cache.times.begin()->first - 1) / o.fps;
        if (o.frame_times.empty()) {
            for (const auto &row : cache.times)
                if (std::abs(row.second - (row.first - 1) / o.fps - cache_origin) > 1e-4)
                    throw std::runtime_error(
                        "irregular detection timestamps require complete --frame-times CSV");
        } else {
            double previous_time = -1;
            for (int n = 1; n <= o.frames; ++n) {
                if (!times.count(n))
                    throw std::runtime_error("frame-times CSV is incomplete");
                double value = times.at(n);
                if (value < 0 || (n > 1 && value <= previous_time))
                    throw std::runtime_error(
                        "timestamps must be nonnegative and strictly increasing");
                previous_time = value;
            }
        }
        kinetic::DetectorConfig detector_config;
        detector_config.input_size = o.input_size;
        detector_config.score_threshold = o.score;
        detector_config.nms_threshold = o.nms;
        kinetic::YoloDetector detector(o.model, detector_config);
        if (o.detections.empty()) {
            double prep = 0, infer = 0, post = 0;
            detector.run(cv::Mat(o.input_size, o.input_size, CV_8UC3, cv::Scalar(114, 114, 114)),
                         prep, infer, post);
        }
        kinetic::TrackerConfig config;
        config.association = o.association;
        config.high_threshold = o.high;
        config.low_threshold = o.low;
        config.iou_threshold = o.iou;
        config.min_hits = o.min_hits;
        config.max_age_seconds = o.max_age;
        config.prediction_output_seconds = o.prediction_age;
        kinetic::Tracker tracker(config);
        kinetic::SchedulingConfig schedule_config;
        schedule_config.mode = o.schedule;
        schedule_config.interval = o.interval;
        schedule_config.max_gap = o.max_gap;
        schedule_config.min_gap = o.min_gap;
        schedule_config.total_frames = o.frames;
        schedule_config.uniform_calls = o.uniform_calls;
        schedule_config.seed = o.seed;
        schedule_config.risk_threshold = o.risk_threshold;
        schedule_config.novelty_threshold = o.novelty_threshold;
        schedule_config.target_rate = o.target_rate;
        kinetic::Scheduler scheduler(schedule_config);
        fs::create_directories(o.output);
        std::ofstream mot(fs::path(o.output) / "mot.txt"),
            telemetry(fs::path(o.output) / "telemetry.csv"),
            frame_log(fs::path(o.output) / "frames.csv"),
            detlog(fs::path(o.output) / "detections.csv");
        if (!mot || !telemetry || !frame_log || !detlog)
            throw std::runtime_error("cannot create output files");
        telemetry << "frame,timestamp,id,x,y,w,h,score,class_id,state,observed\n";
        frame_log << "frame,timestamp,detector_ran,detections,tracks,uncertainty,ambiguity,novelty,"
                     "decode_ms,preprocess_ms,inference_ms,postprocess_ms,tracking_ms,schedule_ms,"
                     "output_ms,total_ms\n";
        detlog << "frame,timestamp,x,y,w,h,score,class_id\n";
        for (auto *stream : {&mot, &telemetry, &frame_log, &detlog})
            *stream << std::fixed << std::setprecision(8);
        cv::VideoWriter writer;
        cv::Mat previous;
        double init_ms = elapsed(startup), last_time = -1;
        auto wall = Clock::now();
        int count = 0, calls = 0;
        std::vector<double> totals, inferences;
        double decode_sum = 0, sched_sum = 0, track_sum = 0;
        for (int n = 1; n <= o.frames; ++n) {
            auto start = Clock::now(), t = start;
            cv::Mat frame;
            if (capture.isOpened()) {
                if (!capture.read(frame))
                    break;
            } else if (!images.empty()) {
                frame = cv::imread(images[n - 1].string());
                if (frame.empty())
                    throw std::runtime_error("unreadable image");
            }
            double decode_ms = elapsed(t);
            double timestamp = cache_origin + (n - 1) / o.fps;
            if (times.count(n))
                timestamp = times.at(n);
            else if (cache.times.count(n))
                timestamp = cache.times.at(n);
            if (last_time >= 0 && timestamp <= last_time)
                throw std::runtime_error("timestamps must be strictly increasing");
            last_time = timestamp;
            t = Clock::now();
            tracker.predict(timestamp);
            double tracking_ms = elapsed(t);
            double uncertainty = tracker.normalized_uncertainty(), ambiguity = tracker.ambiguity(),
                   novelty = 0;
            t = Clock::now();
            if (o.schedule == "adaptive")
                novelty = scene_novelty(frame, previous, tracker.tracks());
            bool run = scheduler.decide(n, uncertainty, ambiguity, novelty);
            double schedule_ms = elapsed(t), prep = 0, infer = 0, post = 0;
            std::vector<kinetic::Detection> dets;
            if (run) {
                ++calls;
                if (!o.detections.empty()) {
                    auto it = cache.rows.find(n);
                    if (it != cache.rows.end())
                        dets = it->second;
                } else {
                    dets = detector.run(frame, prep, infer, post);
                    inferences.push_back(infer);
                }
            }
            t = Clock::now();
            auto tracks = tracker.update(dets, run);
            tracking_ms += elapsed(t);
            t = Clock::now();
            int visible = 0;
            for (const auto &d : dets)
                detlog << n << ',' << timestamp << ',' << d.box.x << ',' << d.box.y << ','
                       << d.box.width << ',' << d.box.height << ',' << d.score << ',' << d.class_id
                       << '\n';
            for (const auto &tr : tracks) {
                const auto &b = tr.box;
                telemetry << n << ',' << timestamp << ',' << tr.id << ',' << b.x << ',' << b.y
                          << ',' << b.width << ',' << b.height << ',' << tr.confidence << ','
                          << tr.class_id << ',' << tr.state << ',' << tr.observed << '\n';
                if (tr.state == "tentative" || (o.output_policy == "active" && tr.state == "lost"))
                    continue;
                ++visible;
                mot << n << ',' << tr.id << ',' << b.x << ',' << b.y << ',' << b.width << ','
                    << b.height << ',' << tr.confidence << ",-1,-1,-1\n";
                if (!frame.empty() && (o.preview || !o.output_video.empty())) {
                    cv::Scalar color =
                        tr.observed ? cv::Scalar(80, 220, 70) : cv::Scalar(20, 180, 255);
                    cv::rectangle(frame, b, color, 2);
                    cv::putText(frame,
                                std::to_string(tr.id) + (tr.observed ? " observed" : " predicted"),
                                {(int)b.x, (int)b.y - 4}, cv::FONT_HERSHEY_SIMPLEX, .45, color, 1);
                }
            }
            if (!frame.empty() && (o.preview || !o.output_video.empty())) {
                std::string label = o.schedule + " | " + (run ? "DETECT" : "PREDICT") +
                                    " | calls " + std::to_string(calls) + "/" + std::to_string(n);
                cv::putText(frame, label, {12, 25}, cv::FONT_HERSHEY_SIMPLEX, .55, {30, 60, 240},
                            2);
                if (!o.output_video.empty()) {
                    if (!writer.isOpened()) {
                        fs::path p = o.output_video;
                        if (p.has_parent_path())
                            fs::create_directories(p.parent_path());
                        writer.open(o.output_video, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                                    o.fps, frame.size());
                        if (!writer.isOpened())
                            throw std::runtime_error("cannot create video");
                    }
                    writer.write(frame);
                }
                if (o.preview) {
                    cv::imshow("Kinetic-Tracker", frame);
                    if (cv::waitKey(1) == 27)
                        break;
                }
            }
            double output_ms = elapsed(t), total = elapsed(start);
            totals.push_back(total);
            decode_sum += decode_ms;
            sched_sum += schedule_ms;
            track_sum += tracking_ms;
            ++count;
            frame_log << n << ',' << timestamp << ',' << run << ',' << dets.size() << ',' << visible
                      << ',' << uncertainty << ',' << ambiguity << ',' << novelty << ','
                      << decode_ms << ',' << prep << ',' << infer << ',' << post << ','
                      << tracking_ms << ',' << schedule_ms << ',' << output_ms << ',' << total
                      << '\n';
        }
        mot.flush();
        telemetry.flush();
        detlog.flush();
        frame_log.flush();
        writer.release();
        if (o.preview)
            cv::destroyAllWindows();
        double wall_s = elapsed(wall) / 1000;
        std::ofstream summary(fs::path(o.output) / "summary.json");
        summary << std::setprecision(12) << "{\n  \"version\": \"2.0.0\",\n  \"mode\": \""
                << (o.detections.empty() ? "uncached_inference" : "cached_detection_replay")
                << "\",\n"
                << "  \"output_policy\": \"" << o.output_policy << "\",\n"
                << "  \"schedule\": \"" << o.schedule << "\", \"association\": \"" << o.association
                << "\",\n"
                << "  \"frames\": " << count << ", \"detector_calls\": " << calls
                << ", \"detector_rate\": " << (count ? (double)calls / count : 0) << ",\n"
                << "  \"wall_seconds\": " << wall_s
                << ", \"processing_fps\": " << (wall_s > 0 ? count / wall_s : 0) << ",\n"
                << "  \"median_frame_ms\": " << percentile(totals, .5)
                << ", \"p95_frame_ms\": " << percentile(totals, .95) << ",\n"
                << "  \"median_inference_ms\": " << percentile(inferences, .5)
                << ", \"p95_inference_ms\": " << percentile(inferences, .95) << ",\n"
                << "  \"decode_ms\": " << decode_sum << ", \"schedule_ms\": " << sched_sum
                << ", \"tracking_ms\": " << track_sum << ",\n"
                << "  \"initialization_ms\": " << init_ms << ", \"opencv\": \"" << CV_VERSION
                << "\", \"threads\": " << o.threads << ",\n"
                << "  \"risk_threshold\": " << o.risk_threshold
                << ", \"novelty_threshold\": " << o.novelty_threshold
                << ", \"interval\": " << o.interval << ", \"max_gap\": " << o.max_gap << ",\n"
                << "  \"target_rate\": " << o.target_rate
                << ", \"uniform_calls\": " << o.uniform_calls << ",\n"
                << "  \"high_threshold\": " << o.high << ", \"low_threshold\": " << o.low
                << ", \"iou_threshold\": " << o.iou << ", \"min_hits\": " << o.min_hits << ",\n"
                << "  \"max_age_seconds\": " << o.max_age
                << ", \"prediction_output_seconds\": " << o.prediction_age
                << ", \"seed\": " << o.seed << ",\n"
                << "  \"input_video\": \"" << escaped(o.video) << "\", \"input_images\": \""
                << escaped(o.images) << "\",\n"
                << "  \"model\": \"" << escaped(o.model) << "\", \"detection_cache\": \""
                << escaped(o.detections) << "\",\n"
                << "  \"timing_scope\": \"decode through output; initialization and one inference "
                   "warmup separate; frame CSV flush and video close included in wall time; cached "
                   "replay excludes detector work\"\n}\n";
        if (count == 0)
            throw std::runtime_error("no frames processed");
        std::cout << "frames=" << count << " detector_calls=" << calls
                  << " rate=" << (double)calls / count << " fps=" << count / wall_s
                  << " output=" << o.output << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
