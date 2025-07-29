
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include "yolo_utils.h"
#include <windows.h>

int main() {
    std::cout << "Starting YOLOv8 ONNXRuntime inference demo..." << std::endl;

    char exe_path[MAX_PATH];
    GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
    std::string exe_dir = std::string(exe_path);
    exe_dir = exe_dir.substr(0, exe_dir.find_last_of("\\/"));

    // 1. Load image
    std::string image_path = exe_dir + "\\data\\bus.jpg";
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "[Error] Could not load bus.jpg. Path: " << image_path << std::endl;
        return 1;
    }
    std::cout << "Image loaded: " << image.rows << "x" << image.cols << std::endl;

    // 2. Initialize ONNXRuntime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolov8");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // 3. Load YOLOv8 ONNX model
    std::wstring model_path_str = std::wstring(exe_dir.begin(), exe_dir.end()) + L"\\modules\\yolov8n.onnx";
    Ort::Session session(env, model_path_str.c_str(), session_options);
    std::cout << "ONNX model loaded." << std::endl;

    // 4. Preprocess image (resize to 640x640, BGR to RGB, normalize to 0~1, HWC->CHW)
    int input_width = 640;
    int input_height = 640;
    std::vector<float> input_tensor_values(input_width * input_height * 3);
    preprocess(image, input_tensor_values, input_width, input_height);


    // 5. Create input tensor
    std::vector<int64_t> input_shape = {1, 3, input_height, input_width};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    // Get input/output names (new API, returns Ort::AllocatedStringPtr)
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
    auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    input_names.push_back(input_name_ptr.get());
    output_names.push_back(output_name_ptr.get());

    // 6. Inference
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);

    // 7. Parse output
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "Output shape: ";
    for (auto s : output_shape) std::cout << s << " ";
    std::cout << std::endl;

    // Postprocess: parse boxes, scores, class ids, and NMS
    std::vector<Detection> detections;
    float conf_threshold = 0.25f;
    float iou_threshold = 0.45f;
    postprocess(output_data, image, detections, conf_threshold, iou_threshold);

    // COCO80 class names
    std::vector<std::string> class_names = {
        "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
        "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
        "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
        "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
        "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
        "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
        "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
        "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
    };
    draw_detections(image, detections, class_names);

    // 8. Save result
    std::string result_path = exe_dir + "\\result.jpg";
    cv::imwrite(result_path, image);
    std::cout << "Result saved as " << result_path << std::endl;

    std::cout << "Demo finished." << std::endl;
    return 0;
}