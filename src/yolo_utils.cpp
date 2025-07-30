#include "yolo_utils.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <filesystem>

YOLOv8::YOLOv8(const std::string& model_path, YoloModelType type)
    : model_type(type),
      env(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8")) {
    
    // Check if model file exists
    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("Model file not found: " + model_path);
    }

    // Configure session options
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Create inference session
    session = std::make_unique<Ort::Session>(env, std::wstring(model_path.begin(), model_path.end()).c_str(), session_options);

    size_t num_inputs = session->GetInputCount();
    for (size_t i = 0; i < num_inputs; ++i) {
        auto input_name_ptr = session->GetInputNameAllocated(i, allocator);
        input_name_strs.push_back(input_name_ptr.get());
    }

    size_t num_outputs = session->GetOutputCount();
    for (size_t i = 0; i < num_outputs; ++i) {
        auto output_name_ptr = session->GetOutputNameAllocated(i, allocator);
        std::string out_name = output_name_ptr.get();
        if (out_name.empty()) {
            out_name = "output" + std::to_string(i);
            std::cerr << "Warning: Empty output name detected, using default: " << out_name << std::endl;
        }
        output_name_strs.push_back(out_name);
    }

    input_names.clear();
    output_names.clear();

    for (const auto& name : input_name_strs) {
        input_names.push_back(name.c_str());
    }
    for (const auto& name : output_name_strs) {
        output_names.push_back(name.c_str());
    }
    
    // Print debug info
    std::cout << "Input name: " << input_name_strs[0] << std::endl;
    std::cout << "Output name: " << output_name_strs[0] << std::endl;
    std::cout << "Output size: " << output_names.size() << std::endl;
}

void YOLOv8::preprocess(const cv::Mat& image, std::vector<float>& input_tensor_values) {
    // Resize image to model input size
    cv::Mat resized_img;
    cv::resize(image, resized_img, cv::Size(static_cast<int>(input_shape[3]), static_cast<int>(input_shape[2])), 0, 0, cv::INTER_LINEAR);
    
    // Convert BGR to RGB and normalize to 0-1
    cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);
    resized_img.convertTo(resized_img, CV_32F, 1.0 / 255.0);
    
    // Convert HWC to CHW format
    std::vector<cv::Mat> chw;
    for (int i = 0; i < 3; ++i)
        chw.push_back(cv::Mat(static_cast<int>(input_shape[2]), static_cast<int>(input_shape[3]), CV_32F, 
            input_tensor_values.data() + i * static_cast<int>(input_shape[2]) * static_cast<int>(input_shape[3])));
    cv::split(resized_img, chw);
}

std::vector<Detection> YOLOv8::inference(const cv::Mat& image, float conf_threshold, float iou_threshold) {
    std::vector<float> input_tensor_values(static_cast<int>(input_shape[1]) * static_cast<int>(input_shape[2]) * static_cast<int>(input_shape[3]));
    preprocess(image, input_tensor_values);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, 
        input_tensor_values.data(), input_tensor_values.size(), 
        input_shape.data(), input_shape.size());
    
    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = session->Run(Ort::RunOptions{nullptr}, 
            input_names.data(), &input_tensor, 1, output_names.data(), output_names.size());
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return {};
    }

    // Process based on model type
    switch (model_type) {
        case YoloModelType::DETECT:
        case YoloModelType::POSE: {
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            return model_type == YoloModelType::DETECT ? 
                postprocessDetect(output_data, image, conf_threshold, iou_threshold) :
                postprocessPose(output_data, image, conf_threshold, iou_threshold);
        }
        case YoloModelType::SEGMENT: {
            if (output_tensors.size() < 2) {
                throw std::runtime_error("Segment model requires 2 output tensors");
            }
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            float* mask_proto = output_tensors[1].GetTensorMutableData<float>();
            return postprocessSegment(output_data, mask_proto, image, conf_threshold, iou_threshold);
        }
        default:
            throw std::runtime_error("Unsupported model type");
    }
}

std::vector<Detection> YOLOv8::postprocessDetect(
    const float* output_data, const cv::Mat& image, float conf_threshold, float iou_threshold) {
    const int num_boxes = 8400;
    float x_factor = image.cols / static_cast<float>(input_shape[3]);
    float y_factor = image.rows / static_cast<float>(input_shape[2]);
    
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<Detection> detections;

    for (int i = 0; i < num_boxes; ++i) {
        float max_score = -1;
        int class_id = -1;

        for (int c = 0; c < static_cast<int>(num_classes); ++c) {
            float score = output_data[(c + 4) * num_boxes + i];
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }

        if (max_score >= conf_threshold) {
            float cx = output_data[0 * num_boxes + i];
            float cy = output_data[1 * num_boxes + i];
            float w = output_data[2 * num_boxes + i];
            float h = output_data[3 * num_boxes + i];

            int left = static_cast<int>((cx - w / 2) * x_factor);
            int top = static_cast<int>((cy - h / 2) * y_factor);
            int width = static_cast<int>(w * x_factor);
            int height = static_cast<int>(h * y_factor);

            boxes.push_back(cv::Rect(left, top, width, height));
            scores.push_back(max_score);
            class_ids.push_back(class_id);
        }
    }

    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, iou_threshold, nms_indices);
    
    for (int idx : nms_indices) {
        Detection det;
        det.box = boxes[idx];
        det.score = scores[idx];
        det.class_id = class_ids[idx];
        detections.push_back(det);
    }
    
    return detections;
}

std::vector<Detection> YOLOv8::postprocessSegment(const float* output_data, const float* mask_proto, const cv::Mat& image, float conf_threshold, float iou_threshold) {
    const int num_boxes = 8400;
    const int num_classes = 80; 
    const int num_protos = 32;
    const int proto_h = 160;
    const int proto_w = 160;

    float x_factor = image.cols / static_cast<float>(640); 
    float y_factor = image.rows / static_cast<float>(640);

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> mask_coeffs;

    for (int i = 0; i < num_boxes; ++i) {
            float max_score = -1;
            int class_id = -1;

            for (int c = 0; c < num_classes; ++c) {
                float score = output_data[(c + 4) * num_boxes + i];
                if (score > max_score) {
                    max_score = score;
                    class_id = c;
                }
            }

            if (max_score >= conf_threshold) {
                float cx = output_data[0 * num_boxes + i];
                float cy = output_data[1 * num_boxes + i];
                float w = output_data[2 * num_boxes + i];
                float h = output_data[3 * num_boxes + i];

                int left = static_cast<int>((cx - w / 2) * x_factor);
                int top = static_cast<int>((cy - h / 2) * y_factor);
                int width = static_cast<int>(w * x_factor);
                int height = static_cast<int>(h * y_factor);

                boxes.push_back(cv::Rect(left, top, width, height));
                scores.push_back(max_score);
                class_ids.push_back(class_id);

                std::vector<float> coeffs(num_protos);
                for (int p = 0; p < num_protos; ++p) {
                    coeffs[p] = output_data[(num_classes + 4 + p) * num_boxes + i];
                }
                mask_coeffs.push_back(coeffs);
            }
        }

        std::vector<int> nms_indices;
        cv::dnn::NMSBoxes(boxes, scores, conf_threshold, iou_threshold, nms_indices);

        cv::Mat mat_proto(num_protos, proto_h * proto_w, CV_32F, const_cast<float*>(mask_proto));

        std::vector<Detection> detections;
        for (int idx : nms_indices) {
            Detection det;
            det.box = boxes[idx];
            det.score = scores[idx];
            det.class_id = class_ids[idx];

            cv::Mat coeff_mat = cv::Mat(1, num_protos, CV_32F, mask_coeffs[idx].data()).clone();

            cv::Mat mask_flat;
            cv::gemm(coeff_mat, mat_proto, 1.0, cv::Mat(), 0.0, mask_flat);
            cv::Mat mask = mask_flat.reshape(1, proto_h);
            cv::resize(mask, mask, image.size(), 0, 0, cv::INTER_LINEAR);
            cv::threshold(mask, mask, 0.5, 1.0, cv::THRESH_BINARY); 

            det.mask = mask.clone();
            detections.push_back(det);
        }
    
    return detections;
}

std::vector<Detection> YOLOv8::postprocessPose(
    const float* output_data, const cv::Mat& image, float conf_threshold, float iou_threshold) {
    const int num_boxes = 8400;
    const int num_keypoints = 17; 
    float x_factor = image.cols / static_cast<float>(input_shape[3]);
    float y_factor = image.rows / static_cast<float>(input_shape[2]);

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<std::vector<cv::Point>> keypoints_list;
    std::vector<std::vector<float>> keypoint_scores_list;
    std::vector<Detection> pose_detections;
    
    // Pre-allocate memory for vectors to prevent reallocations
    boxes.reserve(num_boxes);
    scores.reserve(num_boxes);
    keypoints_list.reserve(num_boxes);
    keypoint_scores_list.reserve(num_boxes);
    pose_detections.reserve(num_boxes);

    try {
        for (int i = 0; i < num_boxes; ++i) {
            float max_score = -1;
            int class_id = 0; 

            float score = output_data[4 * num_boxes + i]; 
            if (score > max_score && score >= conf_threshold) {
                max_score = score;

                float cx = output_data[0 * num_boxes + i];
                float cy = output_data[1 * num_boxes + i];
                float w = output_data[2 * num_boxes + i];
                float h = output_data[3 * num_boxes + i];

                int left = static_cast<int>((cx - w / 2) * x_factor);
                int top = static_cast<int>((cy - h / 2) * y_factor);
                int width = static_cast<int>(w * x_factor);
                int height = static_cast<int>(h * y_factor);

                std::vector<cv::Point> keypoints;
                std::vector<float> keypoint_scores;
                keypoints.reserve(num_keypoints);
                keypoint_scores.reserve(num_keypoints);
                
                for (int k = 0; k < num_keypoints; ++k) {
                    float kp_x = output_data[(5 + k * 3) * num_boxes + i] * x_factor;
                    float kp_y = output_data[(6 + k * 3) * num_boxes + i] * y_factor;
                    float kp_score = output_data[(7 + k * 3) * num_boxes + i];
                    keypoints.emplace_back(static_cast<int>(kp_x), static_cast<int>(kp_y));
                    keypoint_scores.push_back(kp_score);
                }

                boxes.push_back(cv::Rect(left, top, width, height));
                scores.push_back(max_score);
                keypoints_list.push_back(std::move(keypoints));
                keypoint_scores_list.push_back(std::move(keypoint_scores));
            }
        }

        std::vector<int> nms_indices;
        cv::dnn::NMSBoxes(boxes, scores, conf_threshold, iou_threshold, nms_indices);

        for (int idx : nms_indices) {
            Detection det;
            det.box = boxes[idx];
            det.score = scores[idx];
            det.keypoints = std::move(keypoints_list[idx]);
            det.keypoint_scores = std::move(keypoint_scores_list[idx]);
            pose_detections.push_back(det);
        }

        return pose_detections;
    } catch (const std::exception& e) {
        std::cerr << "Error in postprocessPose: " << e.what() << std::endl;
        return {};
    }
}

void YOLOv8::visualize(cv::Mat& image, const std::vector<Detection>& detections,
                       const std::vector<std::string>& class_names, YoloModelType type) {
    for (const auto& det : detections) {
        cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);
        std::string label = class_names[det.class_id] + ": " + cv::format("%.2f", det.score);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int top = std::max(det.box.y, label_size.height);
        cv::rectangle(image, 
            cv::Point(det.box.x, top - label_size.height),
            cv::Point(det.box.x + label_size.width, top + baseLine),
            cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(image, label, cv::Point(det.box.x, top),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

        switch (type) {
            case YoloModelType::SEGMENT:
                if (!det.mask.empty()) {
                    cv::Mat mask_u8;
                    det.mask.convertTo(mask_u8, CV_8U, 255);
                    cv::Mat color_mask = cv::Mat::zeros(image.size(), image.type());
                    color_mask.setTo(cv::Scalar(0, 128, 0), mask_u8);
                    cv::addWeighted(image, 0.9, color_mask, 0.2, 0, image);
                }
                break;
            case YoloModelType::POSE:
                if (!det.keypoints.empty()) {
                    const std::vector<std::pair<int,int>> skeleton = {
                        {15,13}, {13,11}, {16,14}, {14,12}, {11,12}, 
                        {5,11}, {6,12}, {5,6}, {5,7}, {6,8}, {7,9}, {8,10}, {1,2}, 
                        {0,1}, {0,2}, {1,3}, {2,4}, {3,5}, {4,6}
                    };

                    for (size_t i = 0; i < det.keypoints.size(); ++i) {
                        if (det.keypoint_scores[i] > 0.f) {
                            cv::circle(image, det.keypoints[i], 3, cv::Scalar(0, 0, 255), -1);
                        }
                    }

                    for (const auto& conn : skeleton) {
                        if (conn.first < det.keypoints.size() && conn.second < det.keypoints.size() &&
                            det.keypoint_scores[conn.first] > 0.f && det.keypoint_scores[conn.second] > 0.f) {
                            cv::line(image, det.keypoints[conn.first], det.keypoints[conn.second], cv::Scalar(255, 0, 0), 2);
                        }
                    }
                }
                break;
        }
    }
}