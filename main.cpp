#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <fstream>

using namespace std;
using namespace cv;

struct Detection {
    Rect box;
    int class_id;
    float confidence;
};

struct Track {
    int id;
    KalmanFilter kf;
    Rect box;
    int time_since_update;
};

float calculateIoU(const Rect& box1, const Rect& box2) {
    float x1 = max(box1.x, box2.x);
    float y1 = max(box1.y, box2.y);
    float x2 = min(box1.x + box1.width, box2.x + box2.width);
    float y2 = min(box1.y + box1.height, box2.y + box2.height);

    float width = max(0.0f, x2 - x1);
    float height = max(0.0f, y2 - y1);
    float intersectionArea = width * height;

    float area1 = box1.width * box1.height;
    float area2 = box2.width * box2.height;
    float unionArea = area1 + area2 - intersectionArea;

    if (unionArea == 0.0f) return 0.0f;
    return intersectionArea / unionArea;
}

void solveHungarian(const vector<vector<float>>& costMatrix, vector<int>& assignment) {
    int n = costMatrix.size();
    int m = costMatrix[0].size();
    
    vector<float> u(n + 1), v(m + 1);
    vector<int> p(m + 1), way(m + 1);
    
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        vector<float> minv(m + 1, numeric_limits<float>::infinity());
        vector<bool> used(m + 1, false);
        
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = 0;
            float delta = numeric_limits<float>::infinity();
            
            for (int j = 1; j <= m; ++j) {
                if (!used[j]) {
                    float cur = costMatrix[i0 - 1][j - 1] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            
            for (int j = 0; j <= m; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }
    
    assignment.assign(n, -1);
    for (int j = 1; j <= m; ++j) {
        if (p[j] != 0) {
            assignment[p[j] - 1] = j - 1;
        }
    }
}

KalmanFilter initKalmanFilter(const Rect& box) {
    KalmanFilter kf(7, 4, 0);
    
    kf.transitionMatrix = (Mat_<float>(7, 7) << 
        1, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 1);

    kf.measurementMatrix = (Mat_<float>(4, 7) << 
        1, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0);

    kf.statePre.at<float>(0) = box.x + box.width / 2.0f;
    kf.statePre.at<float>(1) = box.y + box.height / 2.0f;
    kf.statePre.at<float>(2) = box.area();
    kf.statePre.at<float>(3) = (float)box.width / box.height;
    kf.statePre.at<float>(4) = 0;
    kf.statePre.at<float>(5) = 0;
    kf.statePre.at<float>(6) = 0;

    kf.statePost = kf.statePre.clone();

    setIdentity(kf.processNoiseCov, Scalar::all(1e-2));
    setIdentity(kf.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(kf.errorCovPost, Scalar::all(1.0));

    return kf;
}

int main() {
    try {
        cv::dnn::Net net = cv::dnn::readNetFromONNX("../best.onnx");
        cv::VideoCapture cap("../test_video.mp4");
        
        if (!cap.isOpened()) {
            return -1;
        }

        ofstream logFile("flight_telemetry.csv");
        logFile << "Frame,Track_ID,X_Center,Y_Center,Width,Height\n";

        cv::Mat frame;
        vector<Track> active_tracks;
        int next_track_id = 0;
        int frame_count = 0;
        cv::TickMeter tm;

        while (true) {
            cap >> frame;
            if (frame.empty()) break;
            frame_count++;

            tm.reset();
            tm.start();

            cv::resize(frame, frame, cv::Size(1024, 768));

            cv::Mat blob;
            cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
            net.setInput(blob);

            vector<cv::Mat> outputs;
            net.forward(outputs, net.getUnconnectedOutLayersNames());

            int rows = outputs[0].size[1];
            int cols = outputs[0].size[2];
            cv::Mat out(rows, cols, CV_32F, outputs[0].ptr<float>());
            cv::Mat preds = out.t();

            float x_factor = frame.cols / 640.0f;
            float y_factor = frame.rows / 640.0f;

            vector<int> class_ids;
            vector<float> confidences;
            vector<cv::Rect> boxes;

            for (int i = 0; i < preds.rows; ++i) {
                float* rowPtr = preds.ptr<float>(i);
                float maxScore = 0;
                int classId = -1;
                
                for (int c = 4; c < rows; ++c) {
                    if (rowPtr[c] > maxScore) {
                        maxScore = rowPtr[c];
                        classId = c - 4;
                    }
                }

                if (maxScore > 0.25) {
                    float cx = rowPtr[0] * x_factor;
                    float cy = rowPtr[1] * y_factor;
                    float w = rowPtr[2] * x_factor;
                    float h = rowPtr[3] * y_factor;

                    int left = int(cx - 0.5 * w);
                    int top = int(cy - 0.5 * h);

                    boxes.push_back(cv::Rect(left, top, int(w), int(h)));
                    confidences.push_back(maxScore);
                    class_ids.push_back(classId);
                }
            }

            vector<int> indices;
            cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.4, indices);

            vector<cv::Rect> current_detections;
            for (int idx : indices) {
                current_detections.push_back(boxes[idx]);
            }

            for (auto& track : active_tracks) {
                cv::Mat prediction = track.kf.predict();
                track.box.x = prediction.at<float>(0) - track.box.width / 2.0f;
                track.box.y = prediction.at<float>(1) - track.box.height / 2.0f;
                track.time_since_update++;
            }

            int trkNum = active_tracks.size();
            int detNum = current_detections.size();
            int dim = max({1, trkNum, detNum});
            vector<vector<float>> costMatrix(dim, vector<float>(dim, 1.0f));

            if (trkNum > 0 && detNum > 0) {
                for (int t = 0; t < trkNum; t++) {
                    for (int d = 0; d < detNum; d++) {
                        float iou = calculateIoU(active_tracks[t].box, current_detections[d]);
                        costMatrix[t][d] = 1.0f - iou;
                    }
                }
            }

            vector<int> assignment;
            if (trkNum > 0 && detNum > 0) {
                solveHungarian(costMatrix, assignment);
            }

            vector<bool> matched_detections(detNum, false);
            for (int t = 0; t < trkNum; t++) {
                if (assignment.size() > t && assignment[t] >= 0 && assignment[t] < detNum) {
                    int d = assignment[t];
                    if (costMatrix[t][d] < 0.7f) {
                        cv::Mat measurement = (cv::Mat_<float>(4, 1) << 
                            current_detections[d].x + current_detections[d].width / 2.0f,
                            current_detections[d].y + current_detections[d].height / 2.0f,
                            current_detections[d].area(),
                            (float)current_detections[d].width / current_detections[d].height);
                        
                        active_tracks[t].kf.correct(measurement);
                        active_tracks[t].box = current_detections[d];
                        active_tracks[t].time_since_update = 0;
                        matched_detections[d] = true;
                    }
                }
            }

            for (int d = 0; d < detNum; d++) {
                if (!matched_detections[d]) {
                    Track new_track;
                    new_track.id = next_track_id++;
                    new_track.kf = initKalmanFilter(current_detections[d]);
                    new_track.box = current_detections[d];
                    new_track.time_since_update = 0;
                    active_tracks.push_back(new_track);
                }
            }

            active_tracks.erase(remove_if(active_tracks.begin(), active_tracks.end(),
                [](const Track& t) { return t.time_since_update > 15; }), active_tracks.end());

            tm.stop();
            string profile_label = cv::format("Latency: %.1f ms | FPS: %.1f", tm.getTimeMilli(), tm.getFPS());
            cv::putText(frame, profile_label, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);

            for (const auto& track : active_tracks) {
                if (track.time_since_update == 0) {
                    cv::rectangle(frame, track.box, cv::Scalar(0, 255, 0), 2);
                    string label = cv::format("ID: %d", track.id);
                    cv::putText(frame, label, cv::Point(track.box.x, track.box.y - 5), 
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

                    logFile << frame_count << "," 
                            << track.id << "," 
                            << track.box.x + track.box.width / 2 << "," 
                            << track.box.y + track.box.height / 2 << "," 
                            << track.box.width << "," 
                            << track.box.height << "\n";
                }
            }

            cv::imshow("AeroTrack-CPP", frame);
            
            if (cv::waitKey(30) == 27) break;
        }
        
        logFile.close();
        cap.release();
        cv::destroyAllWindows();
        
    } catch (const std::exception& e) {
        cout << "\n[FATAL C++ EXCEPTION] " << e.what() << endl;
    } catch (...) {
        cout << "\n[FATAL UNKNOWN EXCEPTION] Crashed in memory." << endl;
    }
    return 0;
}
