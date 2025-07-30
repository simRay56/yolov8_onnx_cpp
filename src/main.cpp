
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include "yolo_utils.h"
#include <windows.h>

// COCO class names
const std::vector<std::string> COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

std::string getExePath() {
    char exe_path[MAX_PATH];
    GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
    return std::string(exe_path).substr(0, std::string(exe_path).find_last_of("\\/"));
}

int main(int argc, char* argv[]) {
    std::cout << "Starting YOLOv8 ONNXRuntime inference demo..." << std::endl;
    std::string exe_dir = getExePath();

    try {
        // Parse command line arguments
        YoloModelType model_type = YoloModelType::DETECT;

        int arg_offset = 1;
        if (argc > 1) {
            std::string type_arg = argv[1];
            if (type_arg == "detect") {
                model_type = YoloModelType::DETECT;
            } else if (type_arg == "segment") {
                model_type = YoloModelType::SEGMENT;
            } else if (type_arg == "pose") {
                model_type = YoloModelType::POSE;
            } else {
                std::cerr << "Invalid task type: " << type_arg << "\nSupported types: detect, segment, pose\n";
                std::cerr << "Usage example:\n  " << argv[0] << " detect\n";
                return 1;
            }
        }
        // model_path will be determined based on the model type
        std::string model_path;
        if (model_type == YoloModelType::DETECT) {
            model_path = exe_dir + "\\modules\\yolov8n.onnx";
        } else if (model_type == YoloModelType::SEGMENT) {
            model_path = exe_dir + "\\modules\\yolov8s-seg.onnx";
        } else if (model_type == YoloModelType::POSE) {
            model_path = exe_dir + "\\modules\\yolov8n-pose.onnx";
        }
        std::string image_path = exe_dir + "\\data\\bus.jpg";

        float conf_threshold = 0.25f;
        float iou_threshold = 0.45f;

        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Error: Could not load image: " << image_path << std::endl;
            return 1;
        }
        std::cout << "Image loaded: " << image.rows << "x" << image.cols << std::endl;

        YOLOv8 yolo(model_path, model_type);
        std::cout << "Model loaded successfully." << std::endl;

        auto detections = yolo.inference(image, conf_threshold, iou_threshold);
        std::cout << "Detected " << detections.size() << " objects." << std::endl;

        YOLOv8::visualize(image, detections, COCO_CLASSES, model_type);

        std::string result_dir = exe_dir + "\\..\\..\\..\\output";
        std::filesystem::create_directories(result_dir);
        std::string result_path = result_dir + "\\result.jpg";
        cv::imwrite(result_path, image);
        std::cout << "Result saved as " << result_path << std::endl;

        std::cout << "Demo finished successfully." << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}