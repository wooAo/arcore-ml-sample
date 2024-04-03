//
// Created by victor wu on 2023/8/20.
//


#ifndef CAMERADEMO_YOLOV8_H
#define CAMERADEMO_YOLOV8_H
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

struct Object
{
    cv::Rect_<float> rect;
    std::string label;
    std::string sku_name;
    std::string sub_category;
    int box_type = 0; // 0 for unit, 1 for sku
    float prob;
};
#endif //CAMERADEMO_YOLOV8_H

