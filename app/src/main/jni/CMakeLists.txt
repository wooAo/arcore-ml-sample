project(yolov5ncnn)

set (CMAKE_CXX_STANDARD 11)

cmake_minimum_required(VERSION 3.10)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/eigen-3.3.9)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.6.0-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc features2d highgui)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnnvulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(yolov5ncnn SHARED
        yolov5ncnn_jni.cpp
        ndkcamera.cpp
        yolov8.cpp
        BYTETracker.cpp
        kalmanFilter.cpp
        lapjv.cpp
        STrack.cpp
        utils.cpp
        )

target_link_libraries(
        yolov5ncnn
        ncnn
        jnigraphics
        ${OpenCV_LIBS}
        camera2ndk
        mediandk
)