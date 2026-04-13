/**
 * @file emotion_types.h
 * @brief 表情识别类型定义
 * @for ESP32-S3 + TFLite Micro
 */

#ifndef EMOTION_TYPES_H
#define EMOTION_TYPES_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * 表情类别枚举
 *===========================================================================*/
typedef enum {
    EMOTION_CRY = 0,      // 哭
    EMOTION_HAPPY,        // 笑
    EMOTION_ANGRY,        // 生气
    EMOTION_COUNT,        // 表情类别数量
    EMOTION_UNKNOWN = -1  // 未知/低置信度
} EmotionType_t;

/*============================================================================
 * 图像帧数据结构（输入队列）
 *===========================================================================*/
typedef struct {
    uint8_t* image_data;      // 图像数据指针 (RGB565 或 RGB888)
    size_t width;             // 原始图像宽度
    size_t height;            // 原始图像高度
    uint32_t timestamp;       // 时间戳 (ms)
    uint8_t buffer_id;        // 缓冲池 ID (用于内存管理)
    uint8_t reserved[3];      // 对齐填充
} ImageFrame_t;

/*============================================================================
 * 推理结果数据结构（输出队列）
 *===========================================================================*/
typedef struct {
    EmotionType_t emotion;        // 识别出的表情
    float confidence;             // 置信度 (0.0 - 1.0)
    float probabilities[EMOTION_COUNT]; // 各类别概率
    uint32_t inference_time_ms;   // 推理耗时 (ms)
    uint8_t buffer_id;            // 关联的输入帧 ID
    uint8_t reserved[3];          // 对齐填充
} EmotionResult_t;

/*============================================================================
 * 配置常量
 *===========================================================================*/
#define EMOTION_INPUT_WIDTH       96      // 模型输入宽度
#define EMOTION_INPUT_HEIGHT      96      // 模型输入高度
#define EMOTION_INPUT_CHANNELS    3       // RGB 三通道
#define EMOTION_INPUT_SIZE        (EMOTION_INPUT_WIDTH * EMOTION_INPUT_HEIGHT * EMOTION_INPUT_CHANNELS)  // 884736 bytes (RGB888)
#define EMOTION_OUTPUT_SIZE       EMOTION_COUNT  // 3 个输出

#define EMOTION_CONFIDENCE_THRESH 0.5f    // 置信度阈值

/*============================================================================
 * 队列配置
 *===========================================================================*/
#define IMAGE_QUEUE_SIZE          3       // 图像输入队列深度
#define RESULT_QUEUE_SIZE         3       // 结果输出队列深度
#define IMAGE_BUFFER_POOL_COUNT   3       // 图像缓冲池数量

/*============================================================================
 * 任务配置
 *===========================================================================*/
#define INFERENCE_TASK_STACK_SIZE 8192    // 任务栈大小 (bytes)
#define INFERENCE_TASK_PRIORITY   5       // 任务优先级
#define INFERENCE_TASK_CORE_ID    1       // 运行核心 (APP_CPU)

#ifdef __cplusplus
}
#endif

#endif /* EMOTION_TYPES_H */