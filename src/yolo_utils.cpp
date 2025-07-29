#include "yolo_utils.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <algorithm>

void preprocess(const cv::Mat& image, std::vector<float>& input_tensor_values, int input_width, int input_height) {
    cv::Mat resized_img;
    cv::resize(image, resized_img, cv::Size(input_width, input_height), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);
    resized_img.convertTo(resized_img, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> chw;
    for (int i = 0; i < 3; ++i)
        chw.push_back(cv::Mat(input_height, input_width, CV_32F, input_tensor_values.data() + i * input_width * input_height));
    cv::split(resized_img, chw);
}

void postprocess(const float* output_data, const cv::Mat& image, std::vector<Detection>& detections, float conf_threshold, float iou_threshold) {
    const int num_classes = 80;
    const int num_boxes = 8400;
    float x_factor = image.cols / 640.0f;
    float y_factor = image.rows / 640.0f;
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    for (int i = 0; i < num_boxes; ++i) {
        float max_class_score = -1;
        int class_id = -1;
        float box_score = -1;

        for (int c = 0; c < num_classes; ++c) {
            float score = output_data[(c + 4) * num_boxes + i];
            if (score > max_class_score) {
                max_class_score = score;
                class_id = c;
                box_score = score;
            }
        }

        if (box_score >= conf_threshold) {
            float cx = output_data[0 * num_boxes + i];
            float cy = output_data[1 * num_boxes + i];
            float w  = output_data[2 * num_boxes + i];
            float h  = output_data[3 * num_boxes + i];

            int left = static_cast<int>((cx - w / 2) * x_factor);
            int top  = static_cast<int>((cy - h / 2) * y_factor);
            int width = static_cast<int>(w * x_factor);
            int height = static_cast<int>(h * y_factor);
            boxes.push_back(cv::Rect(left, top, width, height));
            scores.push_back(box_score);
            class_ids.push_back(class_id);
        }
    }
    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, iou_threshold, nms_indices);
    for (size_t i = 0; i < nms_indices.size(); ++i) {
        int idx = nms_indices[i];
        detections.push_back({boxes[idx], scores[idx], class_ids[idx]});
    }
}

void draw_detections(cv::Mat& image, const std::vector<Detection>& detections, const std::vector<std::string>& class_names) {
    for (const auto& det : detections) {
        cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);
        std::string label = class_names[det.class_id] + ": " + cv::format("%.2f", det.score);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int top = std::max(det.box.y, label_size.height);
        cv::rectangle(image, cv::Point(det.box.x, top - label_size.height),
            cv::Point(det.box.x + label_size.width, top + baseLine), cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(image, label, cv::Point(det.box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
    }
}
