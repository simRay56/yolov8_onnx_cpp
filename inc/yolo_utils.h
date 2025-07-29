#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>

struct Detection {
    cv::Rect box;
    float score;
    int class_id;
};

// Preprocess image: resize, BGR2RGB, normalize, HWC->CHW
void preprocess(const cv::Mat& image, std::vector<float>& input_tensor_values, int input_width, int input_height);

// Postprocess YOLOv8 output: parse boxes, scores, class ids, and NMS
void postprocess(const float* output_data, const cv::Mat& image, std::vector<Detection>& detections, float conf_threshold, float iou_threshold);

// Draw detections on image
void draw_detections(cv::Mat& image, const std::vector<Detection>& detections, const std::vector<std::string>& class_names);
