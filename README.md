# Emotion Inference for ESP32-S3

基于 TFLite Micro 的表情识别推理框架，专为 ESP32-S3-EYE 开发板设计。

## 功能特性

- 支持三种表情识别：哭 (cry)、笑 (happy)、生气 (angry)
- 输入分辨率：96x96 RGB
- 使用 FreeRTOS 任务队列进行数据流管理
- 支持 INT8 量化模型，优化内存和性能
- PSRAM 优化内存分配

## 目录结构

```
emotion_inference/
├── include/
│   ├── emotion_types.h           # 类型定义
│   ├── emotion_inference_task.h  # 任务 API
│   ├── emotion_preprocess.h      # 预处理 API
│   └── emotion_postprocess.h     # 后处理 API
├── src/
│   ├── emotion_inference_task.c  # 任务实现
│   ├── emotion_preprocess.c      # 预处理实现
│   └── emotion_postprocess.c     # 后处理实现
├── CMakeLists.txt
└── README.md
```

## 快速开始

### 1. 集成到 ESP-IDF 项目

将 `emotion_inference` 目录复制到您的 ESP-IDF 项目的 `components/` 目录下：

```bash
cp -r emotion_inference <your_project>/components/
```

### 2. 添加模型文件

将训练好的 `.tflite` 模型文件添加到项目中，并在 CMakeLists.txt 中引用：

```cmake
idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES emotion_inference
)

# 嵌入模型到固件
idf_build_set_property(COMPILE_OPTIONS "-DTFLITE_MODEL=\"${CMAKE_CURRENT_SOURCE_DIR}/emotion_model.tflite\"" APPEND)
```

### 3. 使用示例

```c
#include "emotion_inference_task.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

// 全局队列
QueueHandle_t image_queue;
QueueHandle_t result_queue;

// 推理完成回调
void on_inference_complete(const EmotionResult_t* result, void* user_data) {
    printf("Detected: %s (%.2f%%)\n", 
           EmotionInferenceTask_GetEmotionName(result->emotion),
           result->confidence * 100);
}

void app_main(void) {
    // 创建队列
    image_queue = xQueueCreate(IMAGE_QUEUE_SIZE, sizeof(ImageFrame_t));
    result_queue = xQueueCreate(RESULT_QUEUE_SIZE, sizeof(EmotionResult_t));
    
    // 初始化推理任务
    EmotionInferenceTask_Config_t config = {
        .model_path = TFLITE_MODEL,
        .input_width = 96,
        .input_height = 96,
        .input_channels = 3,
        .confidence_threshold = 0.5f,
        .image_queue = image_queue,
        .result_queue = result_queue,
        .on_complete = on_inference_complete,
        .user_data = NULL
    };
    
    EmotionErrorCode_t err = EmotionInferenceTask_Init(&config);
    if (err != EMOTION_OK) {
        printf("Failed to initialize emotion inference: %d\n", err);
        return;
    }
    
    printf("Emotion inference task started\n");
    
    // 主循环 - 从摄像头获取图像并推送到队列
    while (1) {
        // 这里应该从摄像头获取图像
        // ImageFrame_t frame = capture_camera_frame();
        // xQueueSend(image_queue, &frame, portMAX_DELAY);
        
        // 处理结果
        EmotionResult_t result;
        if (xQueueReceive(result_queue, &result, pdMS_TO_TICKS(100)) == pdPASS) {
            printf("Emotion: %s, Confidence: %.2f%%\n",
                   EmotionInferenceTask_GetEmotionName(result.emotion),
                   result.confidence * 100);
        }
        
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
```

## API 参考

### emotion_types.h

```c
// 错误码
typedef enum {
    EMOTION_OK = 0,
    EMOTION_ERR_INVALID_PARAM = -1,
    EMOTION_ERR_MEMORY = -2,
    EMOTION_ERR_MODEL = -3,
    EMOTION_ERR_PREPROCESS = -4,
    EMOTION_ERR_INFERENCE = -5,
    EMOTION_ERR_POSTPROCESS = -6
} EmotionErrorCode_t;

// 表情类型
typedef enum {
    EMOTION_CRY = 0,
    EMOTION_HAPPY = 1,
    EMOTION_ANGRY = 2,
    EMOTION_UNKNOWN = -1
} EmotionType_t;

// 图像帧结构
typedef struct {
    uint8_t* data;
    size_t width;
    size_t height;
    size_t channels;
    size_t size;
    uint64_t timestamp;
} ImageFrame_t;

// 推理结果
typedef struct {
    EmotionType_t emotion;
    float confidence;
    float probabilities[3];
    uint64_t timestamp;
} EmotionResult_t;
```

### emotion_inference_task.h

```c
// 初始化推理任务
EmotionErrorCode_t EmotionInferenceTask_Init(const EmotionInferenceTask_Config_t* config);

// 停止推理任务
EmotionErrorCode_t EmotionInferenceTask_Stop(void);

// 获取表情名称
const char* EmotionInferenceTask_GetEmotionName(EmotionType_t emotion);

// 设置置信度阈值
void EmotionInferenceTask_SetConfidenceThreshold(float threshold);

// 获取置信度阈值
float EmotionInferenceTask_GetConfidenceThreshold(void);
```

### emotion_preprocess.h

```c
// 初始化默认配置
void EmotionPreprocess_InitDefault(PreprocessConfig_t* config);

// RGB565 转 RGB888
void EmotionPreprocess_RGB565ToRGB888(const uint16_t* rgb565,
                                       uint8_t* rgb888,
                                       size_t width,
                                       size_t height);

// RGB888 转灰度
void EmotionPreprocess_RGB888ToGray(const uint8_t* rgb888,
                                     uint8_t* gray,
                                     size_t width,
                                     size_t height);

// 缩放 RGB 图像
void EmotionPreprocess_ScaleRGB888(const uint8_t* src,
                                    size_t src_width,
                                    size_t src_height,
                                    uint8_t* dst,
                                    size_t dst_width,
                                    size_t dst_height);

// 归一化到 [0, 1]
void EmotionPreprocess_NormalizeFloat(const uint8_t* src,
                                       float* dst,
                                       size_t size);

// 完整预处理流程
int EmotionPreprocess_FullPipeline(const uint16_t* rgb565_data,
                                    size_t src_width,
                                    size_t src_height,
                                    uint8_t* output_tensor,
                                    const PreprocessConfig_t* config);
```

### emotion_postprocess.h

```c
// Softmax 函数
int EmotionPostprocess_Softmax(float* logits, size_t size);

// Softmax 分类
EmotionErrorCode_t EmotionPostprocess_SoftmaxClassify(const float* logits,
                                                       size_t size,
                                                       EmotionPostprocess_Result_t* result_out);

// 获取最大概率类别
int EmotionPostprocess_GetMaxClassId(const float* probabilities, size_t size);

// 获取最大概率
float EmotionPostprocess_GetMaxProbability(const float* probabilities, size_t size);

// 检查置信度
bool EmotionPostprocess_CheckConfidence(float confidence, float threshold);

// 计算熵
float EmotionPostprocess_CalculateEntropy(const float* probabilities, size_t size);
```

## 训练模型

### 环境要求

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

### 数据准备

准备训练数据，目录结构如下：

```
data/
├── cry/
│   ├── img1.jpg
│   └── ...
├── happy/
│   ├── img1.jpg
│   └── ...
└── angry/
    ├── img1.jpg
    └── ...
```

### 训练模型

```bash
cd train_emotion_model
python train_emotion.py --data_dir ../data --train --convert
```

### 验证模型

```bash
python train_emotion.py --verify ../models/emotion_model_int8.tflite
```

## 性能优化

### 内存优化

- 使用 PSRAM 分配大缓冲区
- 复用缓冲区减少内存分配
- 使用 INT8 量化模型减少内存占用

### 性能优化

- 使用 ESP-DSP 库加速矩阵运算
- 使用 SIMD 指令优化图像处理
- 使用双缓冲机制实现流水线处理

## 故障排除

### 内存不足

如果遇到内存不足的问题，可以尝试：

1. 减小输入图像尺寸
2. 使用更小的模型
3. 增加 PSRAM 使用

### 推理速度慢

如果推理速度太慢，可以尝试：

1. 使用 INT8 量化模型
2. 降低输入帧率
3. 使用更简单的模型架构

## 许可证

MIT License
</content>
<task_progress>
- [x] 分析硬件规格（ESP32-S3-EYE, OV2640, 8MB PSRAM）
- [x] 设计数据结构（ImageFrame_t, EmotionResult_t）
- [x] 设计任务队列架构
- [x] 设计核心 API 接口
- [x] 确认模型规格（96x96，3 类表情）
- [x] 创建 emotion_types.h 类型定义
- [x] 创建 emotion_inference_task.h 头文件
- [x] 创建 emotion_preprocess.h 预处理头文件
- [x] 创建 emotion_inference_task.c 实现
- [x] 创建 emotion_postprocess.h 后处理头文件
- [x] 创建 emotion_postprocess.c 后处理实现
- [x] 创建 emotion_preprocess.c 预处理实现
- [x] 创建 CMakeLists.txt 构建配置
- [x] 创建 Python 训练脚本
- [x] 创建 README.md 使用文档
</task_progress>
