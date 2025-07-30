# YOLOv8 ONNXRuntime C++ Demo

## 项目简介
这是一个使用ONNXRuntime在C++中运行YOLOv8模型的演示项目，支持检测(detect)、分割(segment)和姿态估计(pose)三种任务类型。

## 系统要求
- Windows/Linux
- CMake 3.10+
- OpenCV 4.5+
- ONNXRuntime 1.8+

## 功能特点
- 支持 YOLOv8 ONNX 格式模型推理
- 使用 OpenCV 进行图像处理和可视化
- 支持 COCO 80 类目标检测
- 完整的预处理和后处理实现
- 简洁的 CMake 构建系统

## 环境依赖
- Windows 10/11
- Visual Studio 2019/2022
- CMake 3.15+
- OpenCV 4.x
- ONNXRuntime 1.14.0+ (https://onnxruntime.ai/)

## 目录结构
```
onnx_yolo_c++/
├── .git/                # Git版本控制目录
├── .vscode/             # VSCode配置目录
│   └── settings.json    # 编辑器配置
├── src/                 # 源代码目录
│   ├── main.cpp         # 主程序入口
│   └── yolo_utils.cpp   # 工具函数实现
├── inc/                 # 头文件目录
│   └── yolo_utils.h     # 工具函数声明
├── modules/             # 模型文件目录
│   ├── yolov8n.onnx     # YOLOv8n检测模型
│   ├── yolov8m-seg.onnx # YOLOv8m分割模型
│   └── yolov8n-pose.onnx # YOLOv8n姿态估计模型
├── data/                # 测试数据目录
│   └── bus.jpg          # 测试图片
├── build/               # 构建输出目录
│   ├── bin/             # 可执行文件目录
│   │   └── Release/     # Release版本输出
│   │       ├── data/    # 运行时数据目录
│   │       ├── modules/ # 运行时模型目录
│   │       ├── onnxruntime.dll # ONNX运行时库
│   │       └── yolo_inference.exe # 主程序
│   └── CMake缓存和生成文件
└── CMakeLists.txt       # CMake 构建配置
```

## 安装步骤

1. **安装 OpenCV**
   - 下载 OpenCV Windows 版本：https://opencv.org/releases/
   - 解压到合适的目录（例如：`D:/opencv4.9.0/opencv/`）
   - 在 CMakeLists.txt 中设置 OpenCV_DIR 路径：
     ```cmake
     set(OpenCV_DIR "D:/opencv4.9.0/opencv/build")
     ```
   - 将 `<OpenCV安装路径>/build/x64/vc16/bin` 添加到系统环境变量 PATH

2. **安装 ONNXRuntime**
   - 下载 ONNXRuntime Windows 预编译包：https://github.com/microsoft/onnxruntime/releases
   - 解压到合适的目录（例如：`D:/onnxruntime/`）
   - 在 CMakeLists.txt 中设置 ONNXRUNTIME_DIR 路径：
     ```cmake
     set(ONNXRUNTIME_DIR "D:/onnxruntime/onnxruntime-win-x64-gpu-1.16.0")
     ```
   - 将 `<ONNXRuntime安装路径>/lib` 添加到系统环境变量 PATH

3. **获取 YOLOv8 模型**
   - 从 Ultralytics 官方转换 YOLOv8 模型到 ONNX 格式
   - 或使用预转换的模型（如 yolov8n.onnx）
   - 将模型文件放置在 `modules` 目录下

4. **编译项目**
   ```bash
   # 创建并进入构建目录
   mkdir build
   cd build

   # 生成 Visual Studio 解决方案
   cmake ..

   # 编译 Release 版本
   cmake --build . --config Release
   ```

## 使用说明

支持以下任务类型参数：
- `detect`: 目标检测（默认）
- `segment`: 实例分割
- `pose`: 关键点检测

### 示例命令
```bash
# 默认检测任务
./yolo_inference

# 指定分割任务
./yolo_inference segment

# 使用姿态估计
./yolo_inference pose
```

支持以下可选参数：
- `--model`: 指定ONNX模型文件名（默认：yolov8n.onnx）
- `--image`: 指定输入图片文件名（默认：bus.jpg）
- `--conf`: 设置置信度阈值（默认：0.25）
- `--iou`: 设置IOU阈值（默认：0.45）

1. **准备运行环境**
   - 确保所有依赖库的 DLL 在系统 PATH 中
   - 检查 `build/bin/Release` 目录结构完整：
     ```
     bin/Release/
     ├── yolo_inference.exe    # 主程序
     ├── onnxruntime.dll      # ONNXRuntime 运行时
     ├── modules/             # 模型目录
     │   ├── yolov8n.onnx     # 目标检测模型
     │   ├── yolov8m-seg.onnx # 实例分割模型
     │   └── yolov8n-pose.onnx # 姿态估计模型
     └── data/                # 数据目录
         └── bus.jpg          # 测试图片
     ```

2. **修改源码并运行程序**
   - 打开 `src/main.cpp` 文件
   - 修改以下硬编码路径变量：
     * 模型路径：`std::string model_path = "modules/yolov8n.onnx"`
     * 图片路径：`std::string image_path = "data/bus.jpg"`
   - 保存文件后重新编译项目：
   ```bash
   cd build
   cmake --build . --config Release
   ```

3. **查看结果**
   - 程序会在执行目录生成 `result.jpg`
   - 检测结果包含：
     * 彩色边界框
     * 类别标签
     * 置信度分数（保留两位小数）

## 实现细节

### 预处理流程
1. 图像缩放到 640x640
2. BGR 转 RGB
3. 归一化到 0-1 范围
4. HWC 转 CHW 格式

### 后处理流程
1. 解析模型输出 (shape: [1, 84, 8400])
2. 置信度阈值过滤
3. 非极大值抑制 (NMS)
4. 绘制检测结果

## 常见问题

1. **找不到 DLL**
   ```
   无法启动此程序，因为计算机中丢失 opencv_world4xx.dll/onnxruntime.dll
   ```
   解决方案：
   - 确保 OpenCV 的 `build/x64/vc16/bin` 目录在系统 PATH 中
   - 确保 ONNXRuntime 的 `lib` 目录在系统 PATH 中
   - 或直接将所需 DLL 复制到可执行文件目录

2. **编译错误**
   ```
   CMake Error: The following variables are used in this project, but they are set to NOTFOUND
   ```
   解决方案：
   - 检查 CMakeLists.txt 中的 OpenCV_DIR 路径是否正确
   - 检查 CMakeLists.txt 中的 ONNXRUNTIME_DIR 路径是否正确
   - 确保使用正斜杠(/)或双反斜杠(\\\\)作为路径分隔符

3. **运行时找不到文件**
   ```
   [Error] Could not load bus.jpg/yolov8n.onnx
   ```
   解决方案：
   - 检查 data 和 modules 目录是否正确复制到执行文件目录
   - 检查文件权限是否正确
   - 使用绝对路径进行测试

## 参考资料
- ONNXRuntime C++ API: https://onnxruntime.ai/docs/api/c/
- YOLOv8 官方仓库: https://github.com/ultralytics/ultralytics
- OpenCV 文档: https://docs.opencv.org/

## 许可证
MIT License