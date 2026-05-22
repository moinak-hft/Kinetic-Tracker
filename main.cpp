#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

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

    setIdentity(kf.processNoiseCov, Scalar::all(1e-2));
    setIdentity(kf.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(kf.errorCovPost, Scalar::all(1.0));

    return kf;
}

int main() {
    try {
        cout << "[DEBUG] Program started. Booting engine..." << endl;
        
        cv::VideoCapture cap("../test_video.mp4");
        cout << "[DEBUG] VideoCapture object created." << endl;
        
        if (!cap.isOpened()) {
            cout << "[ERROR] Failed to open ../test_video.mp4" << endl;
            return -1;
        }

        cv::Mat frame;
        vector<Track> active_tracks;
        int next_track_id = 0;
        
        cout << "[DEBUG] Entering video loop." << endl;

        while (true) {
            cap >> frame;
            
            if (frame.empty()) {
                cout << "[DEBUG] End of stream." << endl;
                break;
            }
            
            cv::imshow("Alpha Tracker - MVP", frame);
            
            if (cv::waitKey(30) == 27) { 
                break;
            }
        }
        
        cap.release();
        cv::destroyAllWindows();
        cout << "[DEBUG] Clean exit." << endl;
        
    } catch (const std::exception& e) {
        cout << "\n[FATAL C++ EXCEPTION] " << e.what() << endl;
    } catch (...) {
        cout << "\n[FATAL UNKNOWN EXCEPTION] Crashed in memory." << endl;
    }
    return 0;
}