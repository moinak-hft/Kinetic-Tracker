#include "kinetic/tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/video/tracking.hpp>
#include <stdexcept>
#include <utility>

namespace kinetic {
namespace {
bool valid_box(const cv::Rect2f &box) {
    return std::isfinite(box.x) && std::isfinite(box.y) && std::isfinite(box.width) &&
           std::isfinite(box.height) && box.width > 0.0f && box.height > 0.0f &&
           std::abs(box.x) <= 1e7f && std::abs(box.y) <= 1e7f && box.width <= 1e7f &&
           box.height <= 1e7f;
}

bool valid_detection(const Detection &detection) {
    return valid_box(detection.box) && std::isfinite(detection.score) && detection.score >= 0.0f &&
           detection.score <= 1.0f;
}

cv::Mat measurement(const cv::Rect2f &box) {
    return (cv::Mat_<double>(4, 1) << box.x + box.width * 0.5, box.y + box.height * 0.5, box.width,
            box.height);
}

double observation_noise(const cv::Rect2f &box) {
    return std::max(1.0, 0.03 * std::max(box.width, box.height));
}

// Conservative diagonal loading repairs roundoff without removing uncertainty.
// Only the new motion backend calls this; legacy filters retain exact arithmetic.
bool stabilize_covariance(cv::Mat &covariance) {
    if (!cv::checkRange(covariance))
        return false;
    covariance = (covariance + covariance.t()) * 0.5;
    cv::Mat eigenvalues;
    if (!cv::eigen(covariance, eigenvalues))
        return false;
    const double minimum = eigenvalues.at<double>(eigenvalues.rows - 1);
    if (!std::isfinite(minimum))
        return false;
    if (minimum < 1e-9)
        covariance += cv::Mat::eye(covariance.rows, covariance.cols, CV_64F) * (1e-9 - minimum);
    return cv::checkRange(covariance);
}

struct InternalTrack {
    int id;
    int class_id;
    int consecutive_hits = 1;
    bool confirmed = false;
    bool lost = false;
    bool observed = true;
    float score;
    double last_observed;
    cv::Rect2f box;
    cv::KalmanFilter filter;
    cv::Rect2f last_real_box;
    cv::Rect2f previous_real_box;
    double previous_real_time = 0;
    bool has_previous_real = false;
    int prediction_steps = 0;
    int real_observations = 1;
    double last_positive_predict_dt = 0;

    InternalTrack(int identity, const Detection &detection, double timestamp, int min_hits)
        : id(identity), class_id(detection.class_id), confirmed(min_hits <= 1),
          score(detection.score), last_observed(timestamp), box(detection.box),
          filter(8, 4, 0, CV_64F), last_real_box(detection.box), previous_real_box(detection.box),
          previous_real_time(timestamp) {
        cv::setIdentity(filter.transitionMatrix);
        filter.measurementMatrix = cv::Mat::zeros(4, 8, CV_64F);
        filter.statePost = cv::Mat::zeros(8, 1, CV_64F);
        filter.errorCovPost = cv::Mat::zeros(8, 8, CV_64F);
        const cv::Mat initial = measurement(box);
        const double scale = std::max(box.width, box.height);
        for (int dimension = 0; dimension < 4; ++dimension) {
            filter.measurementMatrix.at<double>(dimension, dimension) = 1.0;
            filter.statePost.at<double>(dimension) = initial.at<double>(dimension);
            filter.errorCovPost.at<double>(dimension, dimension) = std::pow(1.0 + 0.1 * scale, 2);
            filter.errorCovPost.at<double>(dimension + 4, dimension + 4) =
                std::pow(10.0 + 4.0 * scale, 2);
        }
        filter.statePre = filter.statePost.clone();
        filter.errorCovPre = filter.errorCovPost.clone();
    }

    bool reconstruct() {
        for (int index = 0; index < 8; ++index) {
            if (!std::isfinite(filter.statePost.at<double>(index)))
                return false;
        }
        const double width = filter.statePost.at<double>(2);
        const double height = filter.statePost.at<double>(3);
        if (width <= 0.0 || height <= 0.0)
            return false;
        box = cv::Rect2f(static_cast<float>(filter.statePost.at<double>(0) - width * 0.5),
                         static_cast<float>(filter.statePost.at<double>(1) - height * 0.5),
                         static_cast<float>(width), static_cast<float>(height));
        return valid_box(box) && cv::checkRange(filter.errorCovPost);
    }

    bool predict(double dt, bool motion = false) {
        observed = false;
        if (dt <= 0.0)
            return true;
        ++prediction_steps;
        last_positive_predict_dt = dt;
        cv::setIdentity(filter.transitionMatrix);
        filter.processNoiseCov = cv::Mat::zeros(8, 8, CV_64F);
        const double acceleration = 10.0 + 2.0 * std::max(box.width, box.height);
        const double variance = acceleration * acceleration;
        const double dt2 = dt * dt;
        const double dt3 = dt2 * dt;
        const double dt4 = dt2 * dt2;
        for (int dimension = 0; dimension < 4; ++dimension) {
            filter.transitionMatrix.at<double>(dimension, dimension + 4) = dt;
            filter.processNoiseCov.at<double>(dimension, dimension) = variance * dt4 * 0.25;
            filter.processNoiseCov.at<double>(dimension, dimension + 4) = variance * dt3 * 0.5;
            filter.processNoiseCov.at<double>(dimension + 4, dimension) = variance * dt3 * 0.5;
            filter.processNoiseCov.at<double>(dimension + 4, dimension + 4) = variance * dt2;
        }
        filter.predict();
        if (motion) {
            // A linear size velocity cannot make a physical box negative.
            for (int dimension = 2; dimension < 4; ++dimension) {
                if (filter.statePost.at<double>(dimension) < 0.5) {
                    filter.statePost.at<double>(dimension) = 0.5;
                    filter.statePost.at<double>(dimension + 4) = 0;
                    filter.statePre.at<double>(dimension) = 0.5;
                    filter.statePre.at<double>(dimension + 4) = 0;
                }
            }
            if (!stabilize_covariance(filter.errorCovPost))
                return false;
            filter.errorCovPre = filter.errorCovPost.clone();
        }
        return reconstruct();
    }

    void correct(const Detection &detection, double timestamp, int min_hits) {
        filter.measurementNoiseCov = cv::Mat::zeros(4, 4, CV_64F);
        const double scale = std::max(detection.box.width, detection.box.height);
        const double noise = std::max(1.0, 0.03 * scale);
        for (int dimension = 0; dimension < 4; ++dimension)
            filter.measurementNoiseCov.at<double>(dimension, dimension) = noise * noise;
        filter.correct(measurement(detection.box));
        reconstruct();
        if (timestamp - last_observed > 1e-9) {
            previous_real_box = last_real_box;
            previous_real_time = last_observed;
            has_previous_real = true;
            ++real_observations;
            last_real_box = detection.box;
        }
        prediction_steps = 0;
        score = detection.score;
        last_observed = timestamp;
        observed = true;
        lost = false;
        ++consecutive_hits;
        confirmed = confirmed || consecutive_hits >= min_hits;
    }

    double mahalanobis(const Detection &detection) const {
        cv::Mat covariance = filter.errorCovPost(cv::Rect(0, 0, 4, 4)).clone();
        const double noise = observation_noise(detection.box);
        covariance += cv::Mat::eye(4, 4, CV_64F) * (noise * noise);
        covariance = (covariance + covariance.t()) * 0.5;
        if (!cv::checkRange(covariance))
            return std::numeric_limits<double>::infinity();
        const cv::Mat residual = measurement(detection.box) - filter.statePost.rowRange(0, 4);
        cv::Mat solved;
        if (!cv::solve(covariance, residual, solved, cv::DECOMP_CHOLESKY))
            return std::numeric_limits<double>::infinity();
        const double distance = residual.dot(solved);
        return std::isfinite(distance) && distance >= -1e-8
                   ? std::max(0.0, distance)
                   : std::numeric_limits<double>::infinity();
    }

    double center_mahalanobis(const Detection& detection) const {
        cv::Mat covariance = filter.errorCovPost(cv::Rect(0, 0, 2, 2)).clone();
        const double noise = observation_noise(detection.box);
        covariance += cv::Mat::eye(2, 2, CV_64F) * (noise * noise);
        covariance = (covariance + covariance.t()) * 0.5;
        if (!cv::checkRange(covariance)) return std::numeric_limits<double>::infinity();
        const cv::Mat residual = measurement(detection.box).rowRange(0, 2) -
                                 filter.statePost.rowRange(0, 2);
        cv::Mat solved;
        if (!cv::solve(covariance, residual, solved, cv::DECOMP_CHOLESKY))
            return std::numeric_limits<double>::infinity();
        const double distance = residual.dot(solved);
        return std::isfinite(distance) && distance >= -1e-8
                   ? std::max(0.0, distance) : std::numeric_limits<double>::infinity();
    }

    double direction_penalty(const Detection &detection) const {
        if (!has_previous_real || last_observed <= previous_real_time)
            return 0;
        const auto center = [](const cv::Rect2f &b) {
            return cv::Point2d(b.x + b.width * 0.5, b.y + b.height * 0.5);
        };
        const auto previous_direction = center(last_real_box) - center(previous_real_box);
        const auto proposed_direction = center(detection.box) - center(last_real_box);
        const double first_length = cv::norm(previous_direction);
        const double second_length = cv::norm(proposed_direction);
        const double minimum_motion =
            std::max(1e-3, 0.05 * std::hypot(last_real_box.width, last_real_box.height));
        if (first_length < minimum_motion || second_length < minimum_motion)
            return 0;
        const double cosine = std::clamp(
            previous_direction.dot(proposed_direction) / (first_length * second_length), -1.0, 1.0);
        return (1.0 - cosine) * 0.5;
    }

    void correct_motion(const Detection &detection, double timestamp, const TrackerConfig &config) {
        const double gap = timestamp - last_observed;
        const cv::Mat endpoint_velocity =
            gap > 1e-9 ? (measurement(detection.box) - measurement(last_real_box)) / gap
                       : cv::Mat::zeros(4, 1, CV_64F);
        const double noise = observation_noise(detection.box);
        filter.measurementNoiseCov = cv::Mat::eye(4, 4, CV_64F) * (noise * noise);
        const cv::Mat prior_covariance = filter.errorCovPre.clone();
        filter.correct(measurement(detection.box));
        // Joseph form retains a symmetric PSD posterior in the motion backend.
        const cv::Mat remainder =
            cv::Mat::eye(8, 8, CV_64F) - filter.gain * filter.measurementMatrix;
        filter.errorCovPost = remainder * prior_covariance * remainder.t() +
                              filter.gain * filter.measurementNoiseCov * filter.gain.t();

        if (config.motion_recovery && prediction_steps > 1 && gap > 1e-9) {
            const double prior_interval =
                has_previous_real ? last_observed - previous_real_time : gap;
            const double blend =
                std::clamp(gap / (gap + std::max(1e-9, prior_interval)), 0.25, 0.8);
            const double old_noise = observation_noise(last_real_box);
            const double endpoint_variance = (old_noise * old_noise + noise * noise) / (gap * gap);
            for (int dimension = 0; dimension < 4; ++dimension) {
                const int velocity_index = dimension + 4;
                const double variance =
                    std::max(0.0, filter.errorCovPost.at<double>(velocity_index, velocity_index));
                const double limit =
                    std::max(std::hypot(detection.box.width, detection.box.height) / gap,
                             3.0 * std::sqrt(variance));
                const double innovation =
                    std::clamp(endpoint_velocity.at<double>(dimension) -
                                   filter.statePost.at<double>(velocity_index),
                               -limit, limit);
                filter.statePost.at<double>(velocity_index) += blend * innovation;
                // Endpoints are correlated with the positional correction. They
                // are NOT another independent observation: never shrink the
                // velocity covariance for this deterministic re-update.
                filter.errorCovPost.at<double>(velocity_index, velocity_index) =
                    std::max(variance, endpoint_variance) + blend * blend * innovation * innovation;
            }
        }
        if (!stabilize_covariance(filter.errorCovPost) || !reconstruct())
            throw std::runtime_error("motion filter produced an invalid posterior");
        previous_real_box = last_real_box;
        previous_real_time = last_observed;
        has_previous_real = gap > 1e-9;
        if (gap > 1e-9) ++real_observations;
        last_real_box = detection.box;
        score = detection.score;
        last_observed = timestamp;
        observed = true;
        lost = false;
        prediction_steps = 0;
        ++consecutive_hits;
        confirmed = confirmed || consecutive_hits >= config.min_hits;
    }
};
} // namespace

float iou(const cv::Rect2f &first, const cv::Rect2f &second) {
    if (!valid_box(first) || !valid_box(second))
        return 0.0f;
    const double left = std::max<double>(first.x, second.x);
    const double top = std::max<double>(first.y, second.y);
    const double right = std::min<double>(first.x + static_cast<double>(first.width),
                                          second.x + static_cast<double>(second.width));
    const double bottom = std::min<double>(first.y + static_cast<double>(first.height),
                                           second.y + static_cast<double>(second.height));
    const double intersection = std::max(0.0, right - left) * std::max(0.0, bottom - top);
    const double area = static_cast<double>(first.width) * first.height +
                        static_cast<double>(second.width) * second.height - intersection;
    return area > 0.0 ? static_cast<float>(intersection / area) : 0.0f;
}

std::vector<int> hungarian(const std::vector<std::vector<double>> &costs) {
    const std::size_t rows = costs.size();
    if (!rows)
        return {};
    const std::size_t columns = costs.front().size();
    double scale = 1.0;
    for (const auto &row : costs) {
        if (row.size() != columns)
            throw std::invalid_argument("assignment matrix must be rectangular");
        for (double value : row) {
            if (std::isnan(value))
                throw std::invalid_argument("assignment costs cannot contain NaN");
            if (std::isfinite(value))
                scale = std::max(scale, std::abs(value));
        }
    }
    std::vector<int> assignment(rows, -1);
    if (!columns)
        return assignment;

    // A dummy column per real row permits every track to remain unmatched.
    // Dummy rows permit unused detections. Penalty enforces maximum finite
    // cardinality before minimum cost; forbidden edges are never returned.
    const std::size_t size = rows + columns;
    const double unmatched = 2.0 * static_cast<double>(size) + 1.0;
    const double forbidden = 4.0 * (size + 1.0) * (size + 1.0);
    const auto value = [&](std::size_t row, std::size_t column) {
        if (row >= rows)
            return 0.0;
        if (column >= columns)
            return unmatched;
        return std::isfinite(costs[row][column]) ? costs[row][column] / scale : forbidden;
    };

    std::vector<double> row_potential(size + 1), column_potential(size + 1);
    std::vector<std::size_t> matched_row(size + 1), predecessor(size + 1);
    for (std::size_t row = 1; row <= size; ++row) {
        matched_row[0] = row;
        std::size_t column = 0;
        std::vector<double> minimum(size + 1, std::numeric_limits<double>::infinity());
        std::vector<bool> used(size + 1, false);
        do {
            used[column] = true;
            const std::size_t current_row = matched_row[column];
            double delta = std::numeric_limits<double>::infinity();
            std::size_t next_column = 0;
            for (std::size_t candidate = 1; candidate <= size; ++candidate) {
                if (used[candidate])
                    continue;
                const double reduced = value(current_row - 1, candidate - 1) -
                                       row_potential[current_row] - column_potential[candidate];
                if (reduced < minimum[candidate]) {
                    minimum[candidate] = reduced;
                    predecessor[candidate] = column;
                }
                if (minimum[candidate] < delta) {
                    delta = minimum[candidate];
                    next_column = candidate;
                }
            }
            for (std::size_t candidate = 0; candidate <= size; ++candidate) {
                if (used[candidate]) {
                    row_potential[matched_row[candidate]] += delta;
                    column_potential[candidate] -= delta;
                } else
                    minimum[candidate] -= delta;
            }
            column = next_column;
        } while (matched_row[column] != 0);
        do {
            const std::size_t previous = predecessor[column];
            matched_row[column] = matched_row[previous];
            column = previous;
        } while (column != 0);
    }
    for (std::size_t column = 1; column <= columns; ++column) {
        const std::size_t row = matched_row[column];
        if (row && row <= rows && std::isfinite(costs[row - 1][column - 1]))
            assignment[row - 1] = static_cast<int>(column - 1);
    }
    return assignment;
}

struct Tracker::Impl {
    TrackerConfig config;
    std::vector<InternalTrack> active;
    double timestamp = 0.0;
    bool has_timestamp = false;
    int next_id = 1;
    bool has_motion_update = false;
    double last_motion_update = 0;

    explicit Impl(TrackerConfig cfg) : config(std::move(cfg)) {
        if (config.association != "sort" && config.association != "two_stage" &&
            config.association != "motion" && config.association != "recovery")
            throw std::invalid_argument("association must be sort, two_stage, motion, or recovery");
        if (!std::isfinite(config.high_threshold) || !std::isfinite(config.low_threshold) ||
            config.low_threshold < 0 || config.high_threshold < config.low_threshold ||
            config.high_threshold > 1 || !std::isfinite(config.iou_threshold) ||
            config.iou_threshold <= 0 || config.iou_threshold > 1 || config.min_hits < 1 ||
            !std::isfinite(config.max_age_seconds) || config.max_age_seconds < 0 ||
            !std::isfinite(config.prediction_output_seconds) ||
            config.prediction_output_seconds < 0)
            throw std::invalid_argument("invalid tracker configuration");
        if ((config.association == "motion" || config.association == "recovery") &&
            (!std::isfinite(config.motion_gate) || config.motion_gate <= 0 ||
             !std::isfinite(config.motion_mahalanobis_weight) ||
             config.motion_mahalanobis_weight < 0 || config.motion_mahalanobis_weight > 1 ||
             !std::isfinite(config.motion_direction_weight) || config.motion_direction_weight < 0 ||
             config.motion_direction_weight > 1 ||
             !std::isfinite(config.motion_max_center_distance) ||
             config.motion_max_center_distance <= 0 || !std::isfinite(config.motion_cost_limit) ||
             config.motion_cost_limit <= 0 || config.motion_cost_limit > 2 ||
             !std::isfinite(config.motion_ambiguity_margin) || config.motion_ambiguity_margin < 0 ||
             config.motion_ambiguity_margin > 1))
            throw std::invalid_argument("invalid motion association configuration");
        if (config.association == "recovery" &&
            (!std::isfinite(config.recovery_gate) || config.recovery_gate <= 0))
            throw std::invalid_argument("invalid third-pass recovery gate");
    }

    void associate(const std::vector<Detection> &detections,
                   const std::vector<std::size_t> &track_indices,
                   const std::vector<std::size_t> &detection_indices,
                   std::vector<bool> &matched_tracks, std::vector<bool> &matched_detections) {
        if (track_indices.empty() || detection_indices.empty())
            return;
        std::vector<std::vector<double>> costs(
            track_indices.size(),
            std::vector<double>(detection_indices.size(), std::numeric_limits<double>::infinity()));
        std::vector<std::vector<bool>> nonoverlap_recovery(
            track_indices.size(), std::vector<bool>(detection_indices.size(), false));
        for (std::size_t row = 0; row < track_indices.size(); ++row) {
            const auto &track = active[track_indices[row]];
            for (std::size_t column = 0; column < detection_indices.size(); ++column) {
                const auto &detection = detections[detection_indices[column]];
                const float overlap = iou(track.box, detection.box);
                if (track.class_id != detection.class_id)
                    continue;
                if (config.association != "motion" && overlap >= config.iou_threshold)
                    costs[row][column] = 1.0 - overlap;
                if (config.association == "motion") {
                    const double distance = track.mahalanobis(detection);
                    const double dx = track.box.x + track.box.width * 0.5 - detection.box.x -
                                      detection.box.width * 0.5;
                    const double dy = track.box.y + track.box.height * 0.5 - detection.box.y -
                                      detection.box.height * 0.5;
                    const double scale = std::max(
                        1.0, 0.5 * (std::hypot(track.box.width, track.box.height) +
                                    std::hypot(detection.box.width, detection.box.height)));
                    const bool overlap_fallback =
                        config.motion_overlap_fallback && overlap >= config.iou_threshold;
                    if ((!overlap_fallback && distance > config.motion_gate) ||
                        std::hypot(dx, dy) / scale > config.motion_max_center_distance)
                        continue;
                    const double weight = config.motion_mahalanobis_weight;
                    // Keep the strict attempt's multiply/divide arithmetic
                    // unchanged. Hybrid recovery bounds a heuristic distance;
                    // the chi-square-shaped threshold is not calibrated coverage.
                    const double normalized_distance = distance / config.motion_gate;
                    const double motion_cost = config.motion_overlap_fallback
                        ? weight * (std::isfinite(normalized_distance)
                                        ? std::clamp(normalized_distance, 0.0, 1.0) : 1.0)
                        : weight * distance / config.motion_gate;
                    const double cost =
                        (1.0 - weight) * (1.0 - overlap) + motion_cost +
                        config.motion_direction_weight * track.direction_penalty(detection);
                    if (!std::isfinite(cost) || (!overlap_fallback && cost > config.motion_cost_limit))
                        continue;
                    costs[row][column] = cost;
                    nonoverlap_recovery[row][column] = overlap < config.iou_threshold;
                }
            }
        }
        if (config.association == "motion" && config.motion_ambiguity_margin > 0) {
            const auto candidates = costs;
            for (std::size_t row = 0; row < candidates.size(); ++row) {
                for (std::size_t column = 0; column < candidates[row].size(); ++column) {
                    if (!nonoverlap_recovery[row][column])
                        continue;
                    const double cost = candidates[row][column];
                    bool ambiguous = false;
                    for (std::size_t other = 0; other < candidates[row].size(); ++other)
                        if (other != column &&
                            candidates[row][other] <= cost + config.motion_ambiguity_margin)
                            ambiguous = true;
                    for (std::size_t other = 0; other < candidates.size(); ++other)
                        if (other != row &&
                            candidates[other][column] <= cost + config.motion_ambiguity_margin)
                            ambiguous = true;
                    if (ambiguous)
                        costs[row][column] = std::numeric_limits<double>::infinity();
                }
            }
        }
        const auto assignment = hungarian(costs);
        for (std::size_t row = 0; row < assignment.size(); ++row) {
            if (assignment[row] < 0)
                continue;
            const auto track_index = track_indices[row];
            const auto detection_index = detection_indices[assignment[row]];
            if (config.association == "motion")
                active[track_index].correct_motion(detections[detection_index], timestamp, config);
            else
                active[track_index].correct(detections[detection_index], timestamp,
                                            config.min_hits);
            matched_tracks[track_index] = true;
            matched_detections[detection_index] = true;
        }
    }

    void recover(const std::vector<Detection>& detections,
                 const std::vector<std::size_t>& track_indices,
                 const std::vector<std::size_t>& detection_indices,
                 std::vector<bool>& matched_tracks, std::vector<bool>& matched_detections) {
        if (track_indices.empty() || detection_indices.empty()) return;
        std::vector<std::vector<double>> costs(track_indices.size(),
            std::vector<double>(detection_indices.size(), std::numeric_limits<double>::infinity()));
        for (std::size_t row = 0; row < track_indices.size(); ++row) {
            const auto& track = active[track_indices[row]];
            const double gap = timestamp - track.last_observed;
            // A birth has no observed direction. A normal adjacent observation
            // is not an occlusion. Neither may acquire a weak recovery edge.
            if (!track.confirmed || track.real_observations < 2 || !track.has_previous_real ||
                track.last_positive_predict_dt <= 0 ||
                gap + 1e-9 < 1.5 * track.last_positive_predict_dt)
                continue;
            for (std::size_t column = 0; column < detection_indices.size(); ++column) {
                const auto& detection = detections[detection_indices[column]];
                if (track.class_id != detection.class_id) continue;
                const double width_ratio = detection.box.width / track.last_real_box.width;
                const double height_ratio = detection.box.height / track.last_real_box.height;
                if (width_ratio < .5 || width_ratio > 2 || height_ratio < .5 || height_ratio > 2)
                    continue;
                const double distance = track.center_mahalanobis(detection);
                if (!std::isfinite(distance) || distance > config.recovery_gate) continue;
                const double dx = track.box.x + track.box.width * .5 - detection.box.x - detection.box.width * .5;
                const double dy = track.box.y + track.box.height * .5 - detection.box.y - detection.box.height * .5;
                const double scale = std::max(1.0, .5 * (std::hypot(track.box.width, track.box.height) +
                                                        std::hypot(detection.box.width, detection.box.height)));
                if (std::hypot(dx, dy) / scale > config.motion_max_center_distance) continue;
                const double weight = config.motion_mahalanobis_weight;
                const double cost = (1 - weight) * (1 - iou(track.box, detection.box)) +
                                    weight * distance / config.recovery_gate +
                                    config.motion_direction_weight * track.direction_penalty(detection);
                if (std::isfinite(cost) && cost <= config.motion_cost_limit) costs[row][column] = cost;
            }
        }
        const auto candidates = costs;
        for (std::size_t row = 0; row < costs.size(); ++row) {
            for (std::size_t column = 0; column < costs[row].size(); ++column) {
                if (!std::isfinite(candidates[row][column])) continue;
                bool ambiguous = false;
                for (std::size_t other = 0; other < costs[row].size(); ++other)
                    if (other != column && candidates[row][other] <= candidates[row][column] + config.motion_ambiguity_margin)
                        ambiguous = true;
                for (std::size_t other = 0; other < costs.size(); ++other)
                    if (other != row && candidates[other][column] <= candidates[row][column] + config.motion_ambiguity_margin)
                        ambiguous = true;
                if (ambiguous) costs[row][column] = std::numeric_limits<double>::infinity();
            }
        }
        const auto assignment = hungarian(costs);
        for (std::size_t row = 0; row < assignment.size(); ++row) {
            if (assignment[row] < 0) continue;
            const auto track_index = track_indices[row];
            const auto detection_index = detection_indices[assignment[row]];
            active[track_index].correct_motion(detections[detection_index], timestamp, config);
            matched_tracks[track_index] = true;
            matched_detections[detection_index] = true;
        }
    }
};

Tracker::Tracker(TrackerConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}
Tracker::~Tracker() = default;
Tracker::Tracker(Tracker &&) noexcept = default;
Tracker &Tracker::operator=(Tracker &&) noexcept = default;

void Tracker::predict(double timestamp_seconds) {
    if (!std::isfinite(timestamp_seconds) || timestamp_seconds < 0 ||
        (impl_->has_timestamp && timestamp_seconds < impl_->timestamp))
        throw std::invalid_argument("timestamps must be finite, nonnegative, and monotonic");
    const double dt = impl_->has_timestamp ? timestamp_seconds - impl_->timestamp : 0.0;
    impl_->timestamp = timestamp_seconds;
    impl_->has_timestamp = true;
    auto &active = impl_->active;
    active.erase(std::remove_if(active.begin(), active.end(),
                                [&](InternalTrack &track) {
                                    return timestamp_seconds - track.last_observed >
                                               impl_->config.max_age_seconds ||
                                           !track.predict(dt,
                                                          impl_->config.association == "motion");
                                }),
                 active.end());
}

std::vector<TrackResult> Tracker::update(const std::vector<Detection> &detections,
                                         bool detector_ran) {
    if ((impl_->config.association == "motion" || impl_->config.association == "recovery") && detector_ran) {
        if (impl_->has_motion_update && impl_->timestamp <= impl_->last_motion_update)
            throw std::invalid_argument(
                "motion detector observations require strictly increasing timestamps");
        impl_->has_motion_update = true;
        impl_->last_motion_update = impl_->timestamp;
    }
    for (auto &track : impl_->active)
        track.observed = false;
    if (!detector_ran)
        return tracks();
    std::vector<std::size_t> high, low, track_indices;
    for (std::size_t index = 0; index < detections.size(); ++index) {
        if (!valid_detection(detections[index]))
            continue;
        if (detections[index].score >= impl_->config.high_threshold)
            high.push_back(index);
        else if (detections[index].score >= impl_->config.low_threshold)
            low.push_back(index);
    }
    for (std::size_t index = 0; index < impl_->active.size(); ++index)
        track_indices.push_back(index);
    std::vector<bool> matched_tracks(impl_->active.size(), false),
        matched_detections(detections.size(), false);
    impl_->associate(detections, track_indices, high, matched_tracks, matched_detections);
    if (impl_->config.association == "two_stage" || impl_->config.association == "motion" ||
        impl_->config.association == "recovery") {
        track_indices.clear();
        for (std::size_t index = 0; index < impl_->active.size(); ++index)
            if (!matched_tracks[index] && impl_->active[index].confirmed)
                track_indices.push_back(index);
        impl_->associate(detections, track_indices, low, matched_tracks, matched_detections);
    }
    if (impl_->config.association == "recovery") {
        track_indices.clear();
        std::vector<std::size_t> unmatched_high;
        for (std::size_t index = 0; index < impl_->active.size(); ++index)
            if (!matched_tracks[index] && impl_->active[index].confirmed) track_indices.push_back(index);
        for (const auto index : high)
            if (!matched_detections[index]) unmatched_high.push_back(index);
        impl_->recover(detections, track_indices, unmatched_high, matched_tracks, matched_detections);
    }
    for (std::size_t index = 0; index < impl_->active.size(); ++index) {
        if (!matched_tracks[index]) {
            impl_->active[index].lost = impl_->active[index].confirmed;
            impl_->active[index].consecutive_hits = 0;
        }
    }
    for (const auto index : high) {
        if (!matched_detections[index])
            impl_->active.emplace_back(impl_->next_id++, detections[index], impl_->timestamp,
                                       impl_->config.min_hits);
    }
    return tracks();
}

std::vector<TrackResult> Tracker::tracks() const {
    std::vector<TrackResult> results;
    for (const auto &track : impl_->active) {
        const double age = impl_->timestamp - track.last_observed;
        if (age > impl_->config.prediction_output_seconds && !track.observed)
            continue;
        results.push_back({track.id, track.box, track.score, track.class_id,
                           !track.confirmed ? "tentative"
                           : track.lost     ? "lost"
                                            : "confirmed",
                           track.observed});
    }
    return results;
}

double Tracker::normalized_uncertainty(bool active_only) const {
    if (impl_->active.empty())
        return 1.0;
    double uncertainty = 0.0;
    bool eligible = false;
    for (const auto &track : impl_->active) {
        if (active_only &&
            (!track.confirmed || track.lost ||
             impl_->timestamp - track.last_observed > impl_->config.prediction_output_seconds))
            continue;
        eligible = true;
        const double variance = std::max(0.0, track.filter.errorCovPost.at<double>(0, 0) +
                                                  track.filter.errorCovPost.at<double>(1, 1));
        const double diagonal = std::hypot(track.box.width, track.box.height);
        uncertainty = std::max(uncertainty, std::sqrt(variance) / std::max(1.0, diagonal));
    }
    return eligible ? std::clamp(uncertainty, 0.0, 1.0) : 1.0;
}

double Tracker::ambiguity(bool active_only) const {
    double overlap = 0.0;
    const auto eligible = [&](const InternalTrack &track) {
        return !active_only ||
               (track.confirmed && !track.lost &&
                impl_->timestamp - track.last_observed <= impl_->config.prediction_output_seconds);
    };
    for (std::size_t first = 0; first < impl_->active.size(); ++first)
        for (std::size_t second = first + 1; second < impl_->active.size(); ++second)
            if (eligible(impl_->active[first]) && eligible(impl_->active[second]) &&
                impl_->active[first].class_id == impl_->active[second].class_id)
                overlap = std::max<double>(
                    overlap, iou(impl_->active[first].box, impl_->active[second].box));
    return overlap;
}

} // namespace kinetic
