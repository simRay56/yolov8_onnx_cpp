#pragma once
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

enum class YoloModelType { DETECT, SEGMENT, POSE };

struct Detection {
    cv::Rect box;
    float score;
    int class_id;
    cv::Mat mask;
    std::vector<cv::Point> keypoints;
    std::vector<float> keypoint_scores;
};

class YOLOv8 {
public:
    YOLOv8(const std::string& model_path, YoloModelType type);
    
    std::vector<Detection> inference(const cv::Mat& image, 
                                   float conf_threshold = 0.5f,
                                   float iou_threshold = 0.5f);
    
    static void visualize(cv::Mat& image, 
                        const std::vector<Detection>& detections,
                        const std::vector<std::string>& class_names,
                        YoloModelType type);

private:
    void preprocess(const cv::Mat& image, std::vector<float>& input_tensor_values);
    std::vector<Detection> postprocessDetect(const float* output_data, 
                                           const cv::Mat& image,
                                           float conf_threshold,
                                           float iou_threshold);
    std::vector<Detection> postprocessSegment(const float* output_data,
                                            const float* mask_proto,  
                                            const cv::Mat& image,
                                            float conf_threshold,
                                            float iou_threshold);
    std::vector<Detection> postprocessPose(const float* output_data,
                                         const cv::Mat& image,
                                         float conf_threshold,
                                         float iou_threshold);

    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    Ort::SessionOptions session_options;
    YoloModelType model_type;
    Ort::AllocatorWithDefaultOptions allocator;
    
    std::vector<int64_t> input_shape{1, 3, 640, 640};
    const int num_classes = 80;
    
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<std::string> input_name_strs;  
    std::vector<std::string> output_name_strs; 
};