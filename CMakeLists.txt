cmake_minimum_required(VERSION 3.10)
project(yolo_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 项目目录结构
set(INC_DIR ${CMAKE_SOURCE_DIR}/inc)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(MODULES_DIR ${CMAKE_SOURCE_DIR}/modules)
set(DATA_DIR ${CMAKE_SOURCE_DIR}/data)

# 添加头文件目录
include_directories(${INC_DIR})

# Find OpenCV
set(OpenCV_DIR "D:/opencv4.9.0/opencv/build")
find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV first.")
endif()

# === ONNXRuntime 配置 ===
set(ONNXRUNTIME_DIR "D:/onnxruntime/onnxruntime-win-x64-gpu-1.16.0/onnxruntime-win-x64-gpu-1.16.0")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# 收集源文件
file(GLOB SRC_FILES 
    ${SRC_DIR}/*.cpp
)

# 创建可执行文件
add_executable(yolo_inference ${SRC_FILES})

# 链接库
target_link_libraries(yolo_inference PRIVATE ${OpenCV_LIBS} onnxruntime)

# 复制依赖文件到输出目录
add_custom_command(TARGET yolo_inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:yolo_inference>
)

# 复制模型和测试数据到输出目录
add_custom_command(TARGET yolo_inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${MODULES_DIR}"
        "$<TARGET_FILE_DIR:yolo_inference>/modules"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${DATA_DIR}"
        "$<TARGET_FILE_DIR:yolo_inference>/data"
)