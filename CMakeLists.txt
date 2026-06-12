cmake_minimum_required(VERSION 3.10)

project(AlphaTracker)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(AlphaTracker main.cpp)

target_link_libraries(AlphaTracker ${OpenCV_LIBS})