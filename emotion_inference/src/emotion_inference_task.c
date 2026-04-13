/**
 * @file emotion_inference_task.c
 * @brief 表情识别推理任务实现
 * @for ESP32-S3 + TFLite Micro
 */

#include "emotion_inference_task.h"
#include "emotion_preprocess.h"
#include "emotion_postprocess.h"

#include <string.h>
#include <stdio.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"

#include "esp_heap_caps.h"
#include "esp_log.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define TAG "EmotionInference"

/*============================================================================
 * 上下文结构定义
 *===========================================================================*/
struct EmotionInferenceContext {
    /* TFLite Micro 相关 */
    const tflite::Model* model;                 // 模型指针
    tflite::MicroInterpreter* interpreter;      // 解释器指针
    tflite::MicroProfiler* profiler;            // 性能分析器
    
    /* 内存管理 */
    uint8_t* tensor_arena;                      // 张量内存池
    size_t tensor_arena_size;                   // 内存池大小
    uint8_t* preprocess_buffer;                 // 预处理缓冲区
    
    /* 队列 */
    QueueHandle_t input_queue;                  // 输入队列
    QueueHandle_t output_queue;                 // 输出队列
    
    /* 回调 */
    EmotionInferenceCallback_t on_inference_complete;
    EmotionInferenceErrorCallback_t on_error;
    void* user_data;
    
    /* 配置 */
    int32_t input_width;
    int32_t input_height;
    int32_t input_channels;
    size_t input_tensor_size;                   // 输入张量大小 (bytes)
    
    /* 状态 */
    TaskHandle_t task_handle;                   // 任务句柄
    bool running;                               // 运行状态
    bool initialized;                           // 初始化状态
    int last_error;                             // 最后错误码
    char last_error_msg[128];                   // 最后错误信息
    
    /* 性能统计 */
    uint32_t total_inferences;                  // 总推理次数
    uint32_t total_inference_time_ms;           // 总推理时间
    uint32_t last_inference_time_ms;            // 上次推理时间
};

/*============================================================================
 * 内部函数声明
 *===========================================================================*/
static void inference_task_loop(void* pvParameters);
static EmotionErrorCode_t load_model(EmotionInferenceContext_t* ctx);
static EmotionErrorCode_t allocate_buffers(EmotionInferenceContext_t* ctx);
static void free_buffers(EmotionInferenceContext_t* ctx);
static void report_error(EmotionInferenceContext_t* ctx, 
                         EmotionErrorCode_t error_code, 
                         const char* message);

/*============================================================================
 * API 实现
 *===========================================================================*/

EmotionErrorCode_t EmotionInferenceTask_Init(const EmotionInferenceConfig_t* config,
                                              EmotionInferenceContext_t** ctx_out) {
    if (config == NULL || ctx_out == NULL) {
        return EMOTION_ERR_INVALID_PARAM;
    }
    
    if (config->model_data == NULL || config->input_queue == NULL || 
        config->output_queue == NULL) {
        return EMOTION_ERR_INVALID_PARAM;
    }
    
    ESP_LOGI(TAG, "Initializing EmotionInferenceTask...");
    
    /* 分配上下文 */
    EmotionInferenceContext_t* ctx = (EmotionInferenceContext_t*)heap_caps_calloc(
        1, sizeof(EmotionInferenceContext_t), MALLOC_CAP_INTERNAL);
    if (ctx == NULL) {
        ESP_LOGE(TAG, "Failed to allocate context");
        return EMOTION_ERR_MEMORY;
    }
    
    /* 初始化配置 */
    ctx->model = nullptr;
    ctx->interpreter = nullptr;
    ctx->profiler = nullptr;
    ctx->tensor_arena = nullptr;
    ctx->preprocess_buffer = nullptr;
    ctx->input_queue = config->input_queue;
    ctx->output_queue = config->output_queue;
    ctx->on_inference_complete = config->on_inference_complete;
    ctx->on_error = config->on_error;
    ctx->user_data = config->user_data;
    ctx->input_width = (config->input_width > 0) ? config->input_width : EMOTION_INPUT_WIDTH;
    ctx->input_height = (config->input_height > 0) ? config->input_height : EMOTION_INPUT_HEIGHT;
    ctx->input_channels = (config->input_channels > 0) ? config->input_channels : EMOTION_INPUT_CHANNELS;
    ctx->input_tensor_size = ctx->input_width * ctx->input_height * ctx->input_channels;
    ctx->task_handle = NULL;
    ctx->running = false;
    ctx->initialized = false;
    ctx->last_error = EMOTION_OK;
    ctx->total_inferences = 0;
    ctx->total_inference_time_ms = 0;
    
    /* 分配预处理缓冲区 (PSRAM) */
    size_t preprocess_buf_size = ctx->input_width * ctx->input_height * 3;  // RGB888
    ctx->preprocess_buffer = (uint8_t*)heap_caps_malloc(preprocess_buf_size, 
                                                         MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (ctx->preprocess_buffer == NULL) {
        ESP_LOGW(TAG, "Failed to allocate PSRAM preprocess buffer, using internal RAM");
        ctx->preprocess_buffer = (uint8_t*)heap_caps_malloc(preprocess_buf_size, 
                                                             MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    }
    
    /* 加载模型 */
    EmotionErrorCode_t err = load_model(ctx);
    if (err != EMOTION_OK) {
        free_buffers(ctx);
        heap_caps_free(ctx);
        return err;
    }
    
    /* 分配张量内存池 */
    ctx->tensor_arena_size = interpreter_tensor_arena_size(ctx->interpreter);
    ctx->tensor_arena = (uint8_t*)heap_caps_malloc(ctx->tensor_arena_size, 
                                                    MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (ctx->tensor_arena == NULL) {
        ESP_LOGE(TAG, "Failed to allocate tensor arena in PSRAM");
        ctx->tensor_arena = (uint8_t*)heap_caps_malloc(ctx->tensor_arena_size, 
                                                        MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (ctx->tensor_arena == NULL) {
            ESP_LOGE(TAG, "Failed to allocate tensor arena in internal RAM too");
            report_error(ctx, EMOTION_ERR_MEMORY, "Tensor arena allocation failed");
            free_buffers(ctx);
            heap_caps_free(ctx);
            return EMOTION_ERR_MEMORY;
        }
    }
    
    /* 重新初始化解释器使用新分配的 arena */
    // 注意：实际实现中需要重新创建解释器
    
    ctx->initialized = true;
    *ctx_out = ctx;
    
    ESP_LOGI(TAG, "EmotionInferenceTask initialized successfully");
    ESP_LOGI(TAG, "Input size: %dx%d, Channels: %d", 
             ctx->input_width, ctx->input_height, ctx->input_channels);
    
    return EMOTION_OK;
}

EmotionErrorCode_t EmotionInferenceTask_Start(EmotionInferenceContext_t* ctx) {
    if (ctx == NULL || !ctx->initialized) {
        return EMOTION_ERR_NOT_INITIALIZED;
    }
    
    if (ctx->running) {
        ESP_LOGW(TAG, "Task already running");
        return EMOTION_OK;
    }
    
    ESP_LOGI(TAG, "Starting inference task...");
    
    BaseType_t ret = xTaskCreatePinnedToCore(
        inference_task_loop,
        "emotion_inference",
        INFERENCE_TASK_STACK_SIZE,
        ctx,
        INFERENCE_TASK_PRIORITY,
        &ctx->task_handle,
        INFERENCE_TASK_CORE_ID
    );
    
    if (ret != pdPASS) {
        ESP_LOGE(TAG, "Failed to create inference task");
        report_error(ctx, EMOTION_ERR_TASK_CREATE, "xTaskCreate failed");
        return EMOTION_ERR_TASK_CREATE;
    }
    
    ctx->running = true;
    ESP_LOGI(TAG, "Inference task started");
    
    return EMOTION_OK;
}

EmotionErrorCode_t EmotionInferenceTask_Stop(EmotionInferenceContext_t* ctx) {
    if (ctx == NULL || !ctx->initialized) {
        return EMOTION_ERR_NOT_INITIALIZED;
    }
    
    if (!ctx->running) {
        ESP_LOGW(TAG, "Task not running");
        return EMOTION_OK;
    }
    
    ESP_LOGI(TAG, "Stopping inference task...");
    ctx->running = false;
    
    if (ctx->task_handle != NULL) {
        vTaskDelete(ctx->task_handle);
        ctx->task_handle = NULL;
    }
    
    ESP_LOGI(TAG, "Inference task stopped");
    return EMOTION_OK;
}

EmotionErrorCode_t EmotionInferenceTask_Deinit(EmotionInferenceContext_t* ctx) {
    if (ctx == NULL) {
        return EMOTION_ERR_INVALID_PARAM;
    }
    
    if (ctx->running) {
        EmotionInferenceTask_Stop(ctx);
    }
    
    free_buffers(ctx);
    heap_caps_free(ctx);
    
    ESP_LOGI(TAG, "EmotionInferenceTask deinitialized");
    return EMOTION_OK;
}

bool EmotionInferenceTask_IsRunning(const EmotionInferenceContext_t* ctx) {
    if (ctx == NULL) {
        return false;
    }
    return ctx->running;
}

const char* EmotionInferenceTask_GetLastError(EmotionInferenceContext_t* ctx) {
    if (ctx == NULL) {
        return "Invalid context";
    }
    return ctx->last_error_msg;
}

const char* EmotionInferenceTask_GetEmotionName(EmotionType_t emotion) {
    switch (emotion) {
        case EMOTION_CRY:    return "cry";
        case EMOTION_HAPPY:  return "happy";
        case EMOTION_ANGRY:  return "angry";
        case EMOTION_UNKNOWN:
        default:             return "unknown";
    }
}

EmotionErrorCode_t EmotionInferenceTask_InferSync(EmotionInferenceContext_t* ctx,
                                                   const uint8_t* image_data,
                                                   EmotionResult_t* result_out) {
    if (ctx == NULL || image_data == NULL || result_out == NULL) {
        return EMOTION_ERR_INVALID_PARAM;
    }
    
    if (!ctx->initialized) {
        return EMOTION_ERR_NOT_INITIALIZED;
    }
    
    /* 获取输入张量 */
    TfLiteTensor* input_tensor = ctx->interpreter->input(0);
    if (input_tensor == NULL) {
        report_error(ctx, EMOTION_ERR_INVALID_PARAM, "Input tensor not found");
        return EMOTION_ERR_INVALID_PARAM;
    }
    
    /* 复制输入数据 */
    memcpy(input_tensor->data.uint8, image_data, ctx->input_tensor_size);
    
    /* 执行推理 */
    uint32_t start_time = xTaskGetTickCount();
    TfLiteStatus invoke_status = ctx->interpreter->Invoke();
    uint32_t end_time = xTaskGetTickCount();
    
    ctx->last_inference_time_ms = (end_time - start_time) * portTICK_PERIOD_MS;
    ctx->total_inference_time_ms += ctx->last_inference_time_ms;
    ctx->total_inferences++;
    
    if (invoke_status != kTfLiteOk) {
        report_error(ctx, EMOTION_ERR_INFERENCE, "Interpreter invoke failed");
        return EMOTION_ERR_INFERENCE;
    }
    
    /* 获取输出张量并后处理 */
    TfLiteTensor* output_tensor = ctx->interpreter->output(0);
    if (output_tensor == NULL) {
        report_error(ctx, EMOTION_ERR_INVALID_PARAM, "Output tensor not found");
        return EMOTION_ERR_INVALID_PARAM;
    }
    
    /* 后处理：获取表情分类结果 */
    EmotionPostprocess_Result_t post_result;
    EmotionErrorCode_t post_err = EmotionPostprocess_SoftmaxClassify(
        output_tensor->data.float32,
        EMOTION_OUTPUT_SIZE,
        &post_result
    );
    
    if (post_err != EMOTION_OK) {
        return post_err;
    }
    
    /* 填充结果 */
    result_out->emotion = (EmotionType_t)post_result.class_id;
    result_out->confidence = post_result.confidence;
    memcpy(result_out->probabilities, post_result.probabilities, 
           sizeof(float) * EMOTION_COUNT);
    result_out->inference_time_ms = ctx->last_inference_time_ms;
    result_out->buffer_id = 0;
    
    return EMOTION_OK;
}

/*============================================================================
 * 内部函数实现
 *===========================================================================*/

static void inference_task_loop(void* pvParameters) {
    EmotionInferenceContext_t* ctx = (EmotionInferenceContext_t*)pvParameters;
    ImageFrame_t image_frame;
    EmotionResult_t result;
    
    ESP_LOGI(TAG, "Inference task loop started");
    
    while (ctx->running) {
        /* 从输入队列接收图像数据 */
        BaseType_t recv_ret = xQueueReceive(ctx->input_queue, &image_frame, 
                                            pdMS_TO_TICKS(100));
        
        if (recv_ret != pdPASS) {
            /* 超时，继续等待 */
            continue;
        }
        
        ESP_LOGD(TAG, "Received image frame (id=%d, size=%dx%d)", 
                 image_frame.buffer_id, image_frame.width, image_frame.height);
        
        /* 执行预处理 */
        uint8_t* input_tensor = ctx->interpreter->input(0)->data.uint8;
        
        /* 假设图像数据是 RGB565 格式 */
        PreprocessConfig_t preprocess_config;
        EmotionPreprocess_InitDefault(&preprocess_config);
        preprocess_config.input_width = ctx->input_width;
        preprocess_config.input_height = ctx->input_height;
        preprocess_config.input_channels = ctx->input_channels;
        
        int preprocess_ret = EmotionPreprocess_FullPipeline(
            (uint16_t*)image_frame.image_data,
            image_frame.width,
            image_frame.height,
            input_tensor,
            &preprocess_config
        );
        
        if (preprocess_ret != 0) {
            ESP_LOGE(TAG, "Preprocessing failed");
            report_error(ctx, EMOTION_ERR_PREPROCESS, "Preprocessing failed");
            continue;
        }
        
        /* 执行推理 */
        uint32_t start_time = xTaskGetTickCount();
        TfLiteStatus invoke_status = ctx->interpreter->Invoke();
        uint32_t end_time = xTaskGetTickCount();
        
        ctx->last_inference_time_ms = (end_time - start_time) * portTICK_PERIOD_MS;
        ctx->total_inference_time_ms += ctx->last_inference_time_ms;
        ctx->total_inferences++;
        
        if (invoke_status != kTfLiteOk) {
            ESP_LOGE(TAG, "Inference failed with status: %d", invoke_status);
            report_error(ctx, EMOTION_ERR_INFERENCE, "Interpreter invoke failed");
            continue;
        }
        
        /* 后处理 */
        TfLiteTensor* output_tensor = ctx->interpreter->output(0);
        if (output_tensor == NULL) {
            ESP_LOGE(TAG, "Output tensor not found");
            continue;
        }
        
        EmotionPostprocess_Result_t post_result;
        EmotionErrorCode_t post_err = EmotionPostprocess_SoftmaxClassify(
            output_tensor->data.float32,
            EMOTION_OUTPUT_SIZE,
            &post_result
        );
        
        if (post_err != EMOTION_OK) {
            continue;
        }
        
        /* 填充结果 */
        result.emotion = (EmotionType_t)post_result.class_id;
        result.confidence = post_result.confidence;
        memcpy(result.probabilities, post_result.probabilities, 
               sizeof(float) * EMOTION_COUNT);
        result.inference_time_ms = ctx->last_inference_time_ms;
        result.buffer_id = image_frame.buffer_id;
        
        ESP_LOGI(TAG, "Inference complete: %s (%.2f%%), time: %dms",
                 EmotionInferenceTask_GetEmotionName(result.emotion),
                 result.confidence * 100,
                 result.inference_time_ms);
        
        /* 发送结果到输出队列 */
        BaseType_t send_ret = xQueueSend(ctx->output_queue, &result, portMAX_DELAY);
        if (send_ret != pdPASS) {
            ESP_LOGE(TAG, "Failed to send result to output queue");
            report_error(ctx, EMOTION_ERR_QUEUE_SEND, "Queue send failed");
        }
        
        /* 调用回调 */
        if (ctx->on_inference_complete != NULL) {
            ctx->on_inference_complete(&result, ctx->user_data);
        }
    }
    
    ESP_LOGI(TAG, "Inference task loop exiting");
    vTaskDelete(NULL);
}

static EmotionErrorCode_t load_model(EmotionInferenceContext_t* ctx) {
    /* 检查模型数据 */
    if (ctx->model_data == NULL) {
        return EMOTION_ERR_MODEL_INVALID;
    }
    
    /* 验证模型 */
    ctx->model = reinterpret_cast<const tflite::Model*>(ctx->model_data);
    if (!ctx->model->Verify()) {
        ESP_LOGE(TAG, "Model verification failed");
        return EMOTION_ERR_MODEL_INVALID;
    }
    
    ESP_LOGI(TAG, "Model loaded successfully");
    return EMOTION_OK;
}

static EmotionErrorCode_t allocate_buffers(EmotionInferenceContext_t* ctx) {
    /* 分配张量 arena */
    size_t arena_size = ctx->tensor_arena_size;
    ctx->tensor_arena = (uint8_t*)heap_caps_malloc(arena_size, 
                                                    MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (ctx->tensor_arena == NULL) {
        ESP_LOGE(TAG, "Failed to allocate tensor arena");
        return EMOTION_ERR_MEMORY;
    }
    
    return EMOTION_OK;
}

static void free_buffers(EmotionInferenceContext_t* ctx) {
    if (ctx->interpreter != NULL) {
        delete ctx->interpreter;
        ctx->interpreter = nullptr;
    }
    
    if (ctx->tensor_arena != NULL) {
        heap_caps_free(ctx->tensor_arena);
        ctx->tensor_arena = nullptr;
    }
    
    if (ctx->preprocess_buffer != NULL) {
        heap_caps_free(ctx->preprocess_buffer);
        ctx->preprocess_buffer = nullptr;
    }
}

static void report_error(EmotionInferenceContext_t* ctx, 
                         EmotionErrorCode_t error_code, 
                         const char* message) {
    if (ctx == NULL) return;
    
    ctx->last_error = error_code;
    snprintf(ctx->last_error_msg, sizeof(ctx->last_error_msg), 
             "Error %d: %s", error_code, message ? message : "Unknown error");
    
    ESP_LOGE(TAG, "%s", ctx->last_error_msg);
    
    if (ctx->on_error != NULL) {
        ctx->on_error(error_code, ctx->last_error_msg, ctx->user_data);
    }
}