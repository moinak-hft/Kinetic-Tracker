# Kinetic-Tracker: Bare-Metal Edge Inference System

**Kinetic-Tracker** (powered by the AlphaTracker engine) is a real-time, C++ based Multi-Object Tracking (MOT) architecture designed for agricultural pest detection. It bypasses the latency constraints of standard Python environments by executing the neural network forward passes and tracking physics entirely on the edge using OpenCV DNN and ONNX graph serialization.

## Systems Architecture

Successfully tracking objects in high-noise environments requires more than just a neural network. This engine implements a custom **Simple Online and Realtime Tracking (SORT)** pipeline to resolve track fragmentation and hardware occlusion.

* **Detection (YOLOv8):** Ingests raw video frames and outputs bounding box coordinates via an ONNX-serialized YOLOv8 model (`best.onnx`).
* **Prediction (Kalman Filters):** Actively predicts the velocity and trajectory matrices of bounding boxes across temporal frames.
* **Assignment (Hungarian Algorithm):** Computes the globally optimal Intersection over Union (IoU) assignment matrix in $O(N^3)$ time complexity, guaranteeing entity ID persistence even when detection confidence fluctuates.

## Repository Structure

* `main.cpp`: The core C++ inference and tracking engine. Handles the video ingestion, OpenCV DNN forward passes, and custom MOT mathematics.
* `test_matching.cpp`: Unit testing for the Hungarian algorithm and matrix allocation.
* `CMakeLists.txt`: Build configuration for MSYS2/MinGW environments.
* `visualize_telemetry.py`: Python visualization script that applies high-pass filters to the raw C++ telemetry to graph continuous spatial vectors.
* `AlphaTracker_Analysis.ipynb`: The primary data analytics Jupyter Notebook containing the architectural breakdown and trajectory graphs.
* `best.onnx` / `yolov8n.onnx`: The compiled neural network weights.

## Performance & Telemetry

The C++ pipeline is profiled using `cv::TickMeter` to strictly measure raw inference latency. The engine continuously streams spatial coordinates (X, Y, Width, Height) to a local CSV for post-execution analytics.

* **CPU Fallback Latency:** ~99.4 ms
* **Frames Per Second:** ~10.1 FPS
* **Fragmentation Resolution:** The Kalman filter successfully bridges 3-frame occlusion gaps, maintaining unbroken ID locks without re-assignment.

## Build and Execution

**1. Compile the C++ Engine**
Ensure you have MSYS2 and MinGW installed with the OpenCV libraries.
```bash
mkdir build
cd build
cmake ..
mingw32-make
```

**2. Run the Inference**
Execute the compiled binary to process the video and generate the `flight_telemetry.csv` file.
```bash
./AlphaTracker.exe
```

**3. Generate the Analytics**
Execute the Jupyter Notebook or run the Python script to render the flight path trajectories.
```bash
pip install pandas matplotlib seaborn
python visualize_telemetry.py
```

**4. Execute Your Own Custom Version**
To run the Kinetic-Tracker architecture on your own footage or custom-trained neural networks, replace the target assets in the root directory:

* **Custom Video Tracking:** Place your `.mp4` file in the root directory and rename it to `test_video.mp4`, or manually update the `cv::VideoCapture` target path in `main.cpp`.
* **Custom YOLOv8 Weights:** If you are using a custom-trained model, export it to ONNX format and place it in the root directory as `best.onnx`.

```bash
yolo export model=your_custom_model.pt format=onnx
mingw32-make
./AlphaTracker.exe
```
