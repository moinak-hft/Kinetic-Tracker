***

# Kinetic-Tracker

A low-latency, real-time multi-object tracking engine implemented in bare-metal C++. 

This tracking engine is designed for execution contexts where performance and deterministic state maintenance are critical. It utilizes YOLOv8 for rapid inference and couples the detection pipeline with a custom, highly optimized implementation of the SORT (Simple Online and Realtime Tracking) algorithm.

## Core Architecture

* **Bare-Metal C++:** Engineered for maximum execution speed, minimal overhead, and strict memory management.
* **State Estimation:** Utilizes Kalman Filtering for predictive bounding box modeling and velocity estimation across frames.
* **Optimal Assignment:** Implements the Kuhn-Munkres (Hungarian) algorithm to solve the bipartite matching problem in `O(N^3)` time complexity, ensuring accurate ID assignment between predicted states and new YOLOv8 detections.
* **Low-Latency Pipeline:** Structurally optimized to minimize bottlenecks between the inference engine and the tracking logic.

## Project Structure

* `main.cpp` - The primary entry point containing the video processing loop and inference-tracking integration.
* `test_matching.cpp` - Isolated testing environment specifically for validating the bipartite matching algorithms and Kalman filter state matrices.
* `test_video.mp4` - Standardized input video for benchmarking and verification.
* `CMakeLists.txt` - CMake configuration for cross-platform builds.
* `LICENSE` - MIT License documentation.

## Prerequisites

* C++17 standard or higher
* CMake 3.10+
* OpenCV 4.x
* ONNX Runtime or TensorRT 

## Build Instructions

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Execution

```bash
./Kinetic-Tracker ../test_video.mp4
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
