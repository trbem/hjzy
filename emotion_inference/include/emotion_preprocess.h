/**
 * @file emotion_preprocess.h
 * @brief 表情识别图像预处理
 * @for ESP32-S3 + TFLite Micro
 * 
 * 功能说明:
 * 1. RGB565 转 RGB888
 * 2. 图像缩放（任意分辨率 -> 96x96）
 * 3. 归一化（0-255 -> 0.0-1.0）
 * 4. 颜色空间转换
 */

#ifndef EMOTION_PREPROCESS_H
#define EMOTION_PREPROCESS_H

#include <stdint.h>
#include <stddef.h>
#include "emotion_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * 预处理配置
 *===========================================================================*/
typedef struct {
    int32_t input_width;          // 目标输入宽度
    int32_t input_height;         // 目标输入高度
    int32_t input_channels;       // 输入通道数 (1=灰度，3=RGB)
    float mean;                   // 归一化均值
    float std;                    // 归一化标准差
    bool normalize;               // 是否归一化到 [0,1]
    bool quantize;                // 是否量化到 [0,255] (INT8 模型)
} PreprocessConfig_t;

/*============================================================================
 * API 函数声明
 *===========================================================================*/

/**
 * @brief 初始化预处理配置（默认配置）
 * @param config 输出：配置结构
 */
void EmotionPreprocess_InitDefault(PreprocessConfig_t* config);

/**
 * @brief RGB565 转 RGB888
 * @param rgb565 输入：RGB565 数据
 * @param rgb888 输出：RGB888 数据
 * @param width 图像宽度
 * @param height 图像高度
 */
void EmotionPreprocess_RGB565ToRGB888(const uint16_t* rgb565,
                                       uint8_t* rgb888,
                                       size_t width,
                                       size_t height);

/**
 * @brief RGB888 转灰度
 * @param rgb888 输入：RGB888 数据
 * @param gray 输出：灰度数据
 * @param width 图像宽度
 * @param height 图像高度
 */
void EmotionPreprocess_RGB888ToGray(const uint8_t* rgb888,
                                     uint8_t* gray,
                                     size_t width,
                                     size_t height);

/**
 * @brief 双线性插值缩放（RGB888）
 * @param src 输入图像数据
 * @param src_width 输入宽度
 * @param src_height 输入高度
 * @param dst 输出图像数据
 * @param dst_width 输出宽度
 * @param dst_height 输出高度
 */
void EmotionPreprocess_ScaleRGB888(const uint8_t* src,
                                    size_t src_width,
                                    size_t src_height,
                                    uint8_t* dst,
                                    size_t dst_width,
                                    size_t dst_height);

/**
 * @brief 双线性插值缩放（灰度）
 * @param src 输入图像数据
 * @param src_width 输入宽度
 * @param src_height 输入高度
 * @param dst 输出图像数据
 * @param dst_width 输出宽度
 * @param dst_height 输出高度
 */
void EmotionPreprocess_ScaleGray(const uint8_t* src,
                                  size_t src_width,
                                  size_t src_height,
                                  uint8_t* dst,
                                  size_t dst_width,
                                  size_t dst_height);

/**
 * @brief 归一化到 [0.0, 1.0]
 * @param src 输入数据 (0-255)
 * @param dst 输出数据 (float)
 * @param size 数据大小
 */
void EmotionPreprocess_NormalizeFloat(const uint8_t* src,
                                       float* dst,
                                       size_t size);

/**
 * @brief 归一化到均值标准差
 * @param src 输入数据
 * @param dst 输出数据
 * @param size 数据大小
 * @param mean 均值
 * @param std 标准差
 */
void EmotionPreprocess_NormalizeMeanStd(const uint8_t* src,
                                         float* dst,
                                         size_t size,
                                         float mean,
                                         float std);

/**
 * @brief 完整预处理流程（RGB565 -> 模型输入）
 * @param rgb565_data 输入：RGB565 图像数据
 * @param src_width 原始宽度
 * @param src_height 原始高度
 * @param output_tensor 输出：模型输入张量
 * @param config 预处理配置
 * @return 0 成功，-1 失败
 */
int EmotionPreprocess_FullPipeline(const uint16_t* rgb565_data,
                                    size_t src_width,
                                    size_t src_height,
                                    uint8_t* output_tensor,
                                    const PreprocessConfig_t* config);

/**
 * @brief 使用 PSRAM 优化的预处理（大图像推荐）
 * @param rgb565_data 输入：RGB565 图像数据
 * @param src_width 原始宽度
 * @param src_height 原始高度
 * @param output_tensor 输出：模型输入张量
 * @param config 预处理配置
 * @return 0 成功，-1 失败
 */
int EmotionPreprocess_FullPipeline_PSRAM(const uint16_t* rgb565_data,
                                          size_t src_width,
                                          size_t src_height,
                                          uint8_t* output_tensor,
                                          const PreprocessConfig_t* config);

#ifdef __cplusplus
}
#endif

#endif /* EMOTION_PREPROCESS_H */