#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

using namespace std;

struct Rect {
    float x, y, width, height;
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

int main() {
    vector<Rect> tracks = {
        {100, 100, 50, 50},
        {300, 300, 40, 40}
    };

    vector<Rect> detections = {
        {105, 102, 50, 50},
        {500, 500, 60, 60},
        {310, 295, 40, 40}
    };

    int numTracks = tracks.size();
    int numDetections = detections.size();
    vector<vector<float>> costMatrix(numTracks, vector<float>(numDetections));

    for (int i = 0; i < numTracks; ++i) {
        for (int j = 0; j < numDetections; ++j) {
            float iou = calculateIoU(tracks[i], detections[j]);
            costMatrix[i][j] = 1.0f - iou;
        }
    }

    vector<int> assignment;
    solveHungarian(costMatrix, assignment);

    for (int i = 0; i < numTracks; ++i) {
        cout << "Track " << i << " assigned to Detection " << assignment[i] << "\n";
    }

    return 0;
}
