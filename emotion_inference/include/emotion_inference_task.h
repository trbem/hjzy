/**
 * @file emotion_inference_task.h
 * @brief 表情识别推理任务 API
 * @for ESP32-S3 + TFLite Micro
 * 
 * 功能说明:
 * 1. 从输入队列接收原始图像数据
 * 2. 执行 TFLite 模型推理
 * 3. 将表情结果发送到输出队列
 * 
 * 使用流程:
 * 1. EmotionInferenceTask_Init() - 初始化任务和模型
 * 2. EmotionInferenceTask_Start() - 启动推理任务
 * 3. EmotionInferenceTask_Stop() - 停止推理任务
 * 4. EmotionInferenceTask_Deinit() - 释放资源
 */

#ifndef EMOTION_INFERENCE_TASK_H
#define EMOTION_INFERENCE_TASK_H

#include <stdint.h>
#include <stdbool.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "tensorflow/lite/micro/tflite_bridge.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 前置声明 */
typedef struct EmotionInferenceContext EmotionInferenceContext_t;

/*============================================================================
 * 回调函数类型定义
 *===========================================================================*/

/**
 * @brief 推理完成回调函数
 * @param result 推理结果指针
 * @param user_data 用户自定义数据
 */
typedef void (*EmotionInferenceCallback_t)(const EmotionResult_t* result, void* user_data);

/**
 * @brief 错误回调函数
 * @param error_code 错误码
 * @param message 错误信息
 * @param user_data 用户自定义数据
 */
typedef void (*EmotionInferenceErrorCallback_t)(int error_code, const char* message, void* user_data);

/*============================================================================
 * 错误码定义
 *===========================================================================*/
typedef enum {
    EMOTION_OK = 0,                    // 成功
    EMOTION_ERR_INVALID_PARAM = -1,    // 无效参数
    EMOTION_ERR_MODEL_LOAD = -2,       // 模型加载失败
    EMOTION_ERR_MODEL_INVALID = -3,    // 模型无效
    EMOTION_ERR_INFERENCE = -4,        // 推理失败
    EMOTION_ERR_QUEUE_SEND = -5,       // 队列发送失败
    EMOTION_ERR_QUEUE_RECV = -6,       // 队列接收失败
    EMOTION_ERR_MEMORY = -7,           // 内存分配失败
    EMOTION_ERR_TASK_CREATE = -8,      // 任务创建失败
    EMOTION_ERR_NOT_INITIALIZED = -9,  // 未初始化
    EMOTION_ERR_PREPROCESS = -10,      // 预处理失败
    EMOTION_ERR_NOT_SUPPORTED = -11    // 不支持的操作
} EmotionErrorCode_t;

/*============================================================================
 * 初始化配置结构
 *===========================================================================*/
typedef struct {
    const void* model_data;           // TFLite 模型数据指针
    size_t model_size;                // 模型大小 (bytes)
    QueueHandle_t input_queue;        // 输入队列句柄
    QueueHandle_t output_queue;       // 输出队列句柄
    EmotionInferenceCallback_t on_inference_complete;  // 推理完成回调
    EmotionInferenceErrorCallback_t on_error;          // 错误回调
    void* user_data;                  // 用户自定义数据
    int32_t input_width;              // 模型输入宽度 (默认 96)
    int32_t input_height;             // 模型输入高度 (默认 96)
    int32_t input_channels;           // 输入通道数 (默认 3)
} EmotionInferenceConfig_t;

/*============================================================================
 * API 函数声明
 *===========================================================================*/

/**
 * @brief 初始化表情识别推理任务
 * @param config 初始化配置
 * @param ctx_out 输出：上下文指针
 * @return 错误码
 * 
 * 示例:
 *   EmotionInferenceConfig_t config = {0};
 *   config.model_data = g_emotion_model;
 *   config.model_size = sizeof(g_emotion_model);
 *   config.input_queue = image_queue;
 *   config.output_queue = result_queue;
 *   
 *   EmotionInferenceContext_t* ctx;
 *   EmotionInferenceTask_Init(&config, &ctx);
 */
EmotionErrorCode_t EmotionInferenceTask_Init(const EmotionInferenceConfig_t* config,
                                              EmotionInferenceContext_t** ctx_out);

/**
 * @brief 启动推理任务
 * @param ctx 上下文指针
 * @return 错误码
 */
EmotionErrorCode_t EmotionInferenceTask_Start(EmotionInferenceContext_t* ctx);

/**
 * @brief 停止推理任务
 * @param ctx 上下文指针
 * @return 错误码
 */
EmotionErrorCode_t EmotionInferenceTask_Stop(EmotionInferenceContext_t* ctx);

/**
 * @brief 释放推理任务资源
 * @param ctx 上下文指针
 * @return 错误码
 */
EmotionErrorCode_t EmotionInferenceTask_Deinit(EmotionInferenceContext_t* ctx);

/**
 * @brief 获取推理状态
 * @param ctx 上下文指针
 * @return true 正在运行，false 未运行
 */
bool EmotionInferenceTask_IsRunning(const EmotionInferenceContext_t* ctx);

/**
 * @brief 获取最后一次错误信息
 * @param ctx 上下文指针
 * @return 错误信息字符串
 */
const char* EmotionInferenceTask_GetLastError(EmotionInferenceContext_t* ctx);

/**
 * @brief 获取表情名称字符串
 * @param emotion 表情类型
 * @return 表情名称
 */
const char* EmotionInferenceTask_GetEmotionName(EmotionType_t emotion);

/**
 * @brief 执行单次推理（阻塞模式，用于测试）
 * @param ctx 上下文指针
 * @param image_data 输入图像数据 (RGB888, 96x96)
 * @param result_out 输出：推理结果
 * @return 错误码
 */
EmotionErrorCode_t EmotionInferenceTask_InferSync(EmotionInferenceContext_t* ctx,
                                                   const uint8_t* image_data,
                                                   EmotionResult_t* result_out);

/*============================================================================
 * 宏定义（向后兼容）
 *===========================================================================*/
#define EMOTION_TASK_DEFAULT_INPUT_WIDTH    96
#define EMOTION_TASK_DEFAULT_INPUT_HEIGHT   96
#define EMOTION_TASK_DEFAULT_INPUT_CHANNELS 3

#ifdef __cplusplus
}
#endif

#endif /* EMOTION_INFERENCE_TASK_H */