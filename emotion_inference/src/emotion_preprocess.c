/**
 * @file emotion_preprocess.c
 * @brief 表情识别图像预处理实现
 * @for ESP32-S3 + TFLite Micro
 */

#include "emotion_preprocess.h"
#include <string.h>
#include <math.h>

/*============================================================================
 * 内部函数
 *===========================================================================*/
static float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

static int clamp(int val, int min_val, int max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

/*============================================================================
 * API 实现
 *===========================================================================*/

void EmotionPreprocess_InitDefault(PreprocessConfig_t* config) {
    if (config == NULL) return;
    
    config->input_width = EMOTION_INPUT_WIDTH;
    config->input_height = EMOTION_INPUT_HEIGHT;
    config->input_channels = EMOTION_INPUT_CHANNELS;
    config->mean = 0.0f;
    config->std = 1.0f;
    config->normalize = true;
    config->quantize = false;
}

void EmotionPreprocess_RGB565ToRGB888(const uint16_t* rgb565,
                                       uint8_t* rgb888,
                                       size_t width,
                                       size_t height) {
    if (rgb565 == NULL || rgb888 == NULL) return;
    
    size_t total_pixels = width * height;
    for (size_t i = 0; i < total_pixels; i++) {
        uint16_t pixel = rgb565[i];
        
        /* 提取 RGB565 分量 */
        uint8_t r5 = (pixel >> 11) & 0x1F;
        uint8_t g6 = (pixel >> 5) & 0x3F;
        uint8_t b5 = pixel & 0x1F;
        
        /* 扩展到 8 位 */
        rgb888[i * 3 + 0] = (r5 << 3) | (r5 >> 2);  // R
        rgb888[i * 3 + 1] = (g6 << 2) | (g6 >> 4);  // G
        rgb888[i * 3 + 2] = (b5 << 3) | (b5 >> 2);  // B
    }
}

void EmotionPreprocess_RGB888ToGray(const uint8_t* rgb888,
                                     uint8_t* gray,
                                     size_t width,
                                     size_t height) {
    if (rgb888 == NULL || gray == NULL) return;
    
    size_t total_pixels = width * height;
    for (size_t i = 0; i < total_pixels; i++) {
        uint8_t r = rgb888[i * 3 + 0];
        uint8_t g = rgb888[i * 3 + 1];
        uint8_t b = rgb888[i * 3 + 2];
        
        /* 使用标准亮度系数 */
        gray[i] = (uint8_t)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

void EmotionPreprocess_ScaleRGB888(const uint8_t* src,
                                    size_t src_width,
                                    size_t src_height,
                                    uint8_t* dst,
                                    size_t dst_width,
                                    size_t dst_height) {
    if (src == NULL || dst == NULL) return;
    if (src_width == 0 || src_height == 0) return;
    
    float x_ratio = (float)src_width / dst_width;
    float y_ratio = (float)src_height / dst_height;
    
    for (size_t dy = 0; dy < dst_height; dy++) {
        for (size_t dx = 0; dx < dst_width; dx++) {
            float src_x = dx * x_ratio;
            float src_y = dy * y_ratio;
            
            int x0 = (int)src_x;
            int y0 = (int)src_y;
            int x1 = clamp(x0 + 1, 0, src_width - 1);
            int y1 = clamp(y0 + 1, 0, src_height - 1);
            
            float fx = src_x - x0;
            float fy = src_y - y0;
            
            /* 双线性插值 */
            for (int c = 0; c < 3; c++) {
                uint8_t p00 = src[(y0 * src_width + x0) * 3 + c];
                uint8_t p10 = src[(y0 * src_width + x1) * 3 + c];
                uint8_t p01 = src[(y1 * src_width + x0) * 3 + c];
                uint8_t p11 = src[(y1 * src_width + x1) * 3 + c];
                
                float val = lerp(lerp(p00, p10, fx), lerp(p01, p11, fx), fy);
                dst[(dy * dst_width + dx) * 3 + c] = (uint8_t)val;
            }
        }
    }
}

void EmotionPreprocess_ScaleGray(const uint8_t* src,
                                  size_t src_width,
                                  size_t src_height,
                                  uint8_t* dst,
                                  size_t dst_width,
                                  size_t dst_height) {
    if (src == NULL || dst == NULL) return;
    if (src_width == 0 || src_height == 0) return;
    
    float x_ratio = (float)src_width / dst_width;
    float y_ratio = (float)src_height / dst_height;
    
    for (size_t dy = 0; dy < dst_height; dy++) {
        for (size_t dx = 0; dx < dst_width; dx++) {
            float src_x = dx * x_ratio;
            float src_y = dy * y_ratio;
            
            int x0 = (int)src_x;
            int y0 = (int)src_y;
            int x1 = clamp(x0 + 1, 0, src_width - 1);
            int y1 = clamp(y0 + 1, 0, src_height - 1);
            
            float fx = src_x - x0;
            float fy = src_y - y0;
            
            /* 双线性插值 */
            uint8_t p00 = src[y0 * src_width + x0];
            uint8_t p10 = src[y0 * src_width + x1];
            uint8_t p01 = src[y1 * src_width + x0];
            uint8_t p11 = src[y1 * src_width + x1];
            
            float val = lerp(lerp(p00, p10, fx), lerp(p01, p11, fx), fy);
            dst[dy * dst_width + dx] = (uint8_t)val;
        }
    }
}

void EmotionPreprocess_NormalizeFloat(const uint8_t* src,
                                       float* dst,
                                       size_t size) {
    if (src == NULL || dst == NULL) return;
    
    for (size_t i = 0; i < size; i++) {
        dst[i] = src[i] / 255.0f;
    }
}

void EmotionPreprocess_NormalizeMeanStd(const uint8_t* src,
                                         float* dst,
                                         size_t size,
                                         float mean,
                                         float std) {
    if (src == NULL || dst == NULL) return;
    if (std == 0.0f) std = 1.0f;
    
    for (size_t i = 0; i < size; i++) {
        dst[i] = (src[i] - mean) / std;
    }
}

int EmotionPreprocess_FullPipeline(const uint16_t* rgb565_data,
                                    size_t src_width,
                                    size_t src_height,
                                    uint8_t* output_tensor,
                                    const PreprocessConfig_t* config) {
    if (rgb565_data == NULL || output_tensor == NULL) return -1;
    if (config == NULL) return -1;
    
    size_t total_pixels = src_width * src_height;
    size_t output_size = config->input_width * config->input_height * config->input_channels;
    
    /* 临时缓冲区 */
    uint8_t* rgb888_buffer = (uint8_t*)malloc(total_pixels * 3);
    if (rgb888_buffer == NULL) return -1;
    
    uint8_t* scaled_buffer = (uint8_t*)malloc(config->input_width * config->input_height * 3);
    if (scaled_buffer == NULL) {
        free(rgb888_buffer);
        return -1;
    }
    
    /* 步骤 1: RGB565 -> RGB888 */
    EmotionPreprocess_RGB565ToRGB888(rgb565_data, rgb888_buffer, src_width, src_height);
    
    /* 步骤 2: 缩放 */
    EmotionPreprocess_ScaleRGB888(rgb888_buffer, src_width, src_height,
                                   scaled_buffer, 
                                   config->input_width, config->input_height);
    
    /* 步骤 3: 复制到输出并归一化 */
    if (config->normalize) {
        float* float_buffer = (float*)malloc(output_size * sizeof(float));
        if (float_buffer == NULL) {
            free(rgb888_buffer);
            free(scaled_buffer);
            return -1;
        }
        
        EmotionPreprocess_NormalizeFloat(scaled_buffer, float_buffer, output_size);
        
        /* 转换为 uint8 (用于 INT8 模型) 或 float (用于 FP32 模型) */
        if (config->quantize) {
            for (size_t i = 0; i < output_size; i++) {
                output_tensor[i] = (uint8_t)(float_buffer[i] * 255.0f);
            }
        } else {
            /* 对于 FP32 模型，输出应该是 float */
            memcpy(output_tensor, float_buffer, output_size * sizeof(float));
        }
        
        free(float_buffer);
    } else {
        memcpy(output_tensor, scaled_buffer, output_size);
    }
    
    free(rgb888_buffer);
    free(scaled_buffer);
    
    return 0;
}

int EmotionPreprocess_FullPipeline_PSRAM(const uint16_t* rgb565_data,
                                          size_t src_width,
                                          size_t src_height,
                                          uint8_t* output_tensor,
                                          const PreprocessConfig_t* config) {
    if (rgb565_data == NULL || output_tensor == NULL) return -1;
    if (config == NULL) return -1;
    
    /* 使用 PSRAM 分配大缓冲区 */
    size_t total_pixels = src_width * src_height;
    size_t output_size = config->input_width * config->input_height * config->input_channels;
    
    uint8_t* rgb888_buffer = (uint8_t*)heap_caps_malloc(total_pixels * 3, 
                                                         MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (rgb888_buffer == NULL) {
        rgb888_buffer = (uint8_t*)malloc(total_pixels * 3);
        if (rgb888_buffer == NULL) return -1;
    }
    
    uint8_t* scaled_buffer = (uint8_t*)heap_caps_malloc(config->input_width * 
                                                         config->input_height * 3,
                                                         MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (scaled_buffer == NULL) {
        scaled_buffer = (uint8_t*)malloc(config->input_width * config->input_height * 3);
        if (scaled_buffer == NULL) {
            if (rgb888_buffer != NULL && rgb888_buffer != (uint8_t*)heap_caps_malloc(1, MALLOC_CAP_SPIRAM)) {
                free(rgb888_buffer);
            }
            return -1;
        }
    }
    
    /* 步骤 1: RGB565 -> RGB888 */
    EmotionPreprocess_RGB565ToRGB888(rgb565_data, rgb888_buffer, src_width, src_height);
    
    /* 步骤 2: 缩放 */
    EmotionPreprocess_ScaleRGB888(rgb888_buffer, src_width, src_height,
                                   scaled_buffer,
                                   config->input_width, config->input_height);
    
    /* 步骤 3: 复制到输出并归一化 */
    if (config->normalize) {
        float* float_buffer = (float*)heap_caps_malloc(output_size * sizeof(float),
                                                        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (float_buffer == NULL) {
            float_buffer = (float*)malloc(output_size * sizeof(float));
            if (float_buffer == NULL) {
                if (scaled_buffer != (uint8_t*)heap_caps_malloc(1, MALLOC_CAP_SPIRAM)) {
                    free(scaled_buffer);
                }
                if (rgb888_buffer != (uint8_t*)heap_caps_malloc(1, MALLOC_CAP_SPIRAM)) {
                    free(rgb888_buffer);
                }
                return -1;
            }
        }
        
        EmotionPreprocess_NormalizeFloat(scaled_buffer, float_buffer, output_size);
        
        if (config->quantize) {
            for (size_t i = 0; i < output_size; i++) {
                output_tensor[i] = (uint8_t)(float_buffer[i] * 255.0f);
            }
        } else {
            memcpy(output_tensor, float_buffer, output_size * sizeof(float));
        }
        
        if (float_buffer != (float*)heap_caps_malloc(1, MALLOC_CAP_SPIRAM)) {
            free(float_buffer);
        }
    } else {
        memcpy(output_tensor, scaled_buffer, output_size);
    }
    
    if (scaled_buffer != (uint8_t*)heap_caps_malloc(1, MALLOC_CAP_SPIRAM)) {
        free(scaled_buffer);
    }
    if (rgb888_buffer != (uint8_t*)heap_caps_malloc(1, MALLOC_CAP_SPIRAM)) {
        free(rgb888_buffer);
    }
    
    return 0;
}