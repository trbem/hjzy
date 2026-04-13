/**
 * @file emotion_postprocess.h
 * @brief 表情识别后处理
 * @for ESP32-S3 + TFLite Micro
 * 
 * 功能说明:
 * 1. Softmax 归一化
 * 2. 分类结果提取
 * 3. 置信度计算
 */

#ifndef EMOTION_POSTPROCESS_H
#define EMOTION_POSTPROCESS_H

#include <stdint.h>
#include "emotion_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * 后处理结果结构
 *===========================================================================*/
typedef struct {
    int class_id;             // 预测类别 ID
    float confidence;         // 置信度 (0.0-1.0)
    float probabilities[EMOTION_COUNT]; // 各类别概率
} EmotionPostprocess_Result_t;

/*============================================================================
 * API 函数声明
 *===========================================================================*/

/**
 * @brief Softmax 函数（原地计算）
 * @param logits 输入/输出：logits 数组
 * @param size 数组大小
 * @return 0 成功，-1 失败
 */
int EmotionPostprocess_Softmax(float* logits, size_t size);

/**
 * @brief 从 logits 获取分类结果
 * @param logits 输入：logits 数组
 * @param size 数组大小
 * @param result_out 输出：结果结构
 * @return 错误码
 */
EmotionErrorCode_t EmotionPostprocess_SoftmaxClassify(const float* logits,
                                                       size_t size,
                                                       EmotionPostprocess_Result_t* result_out);

/**
 * @brief 获取最高概率的类别
 * @param probabilities 概率数组
 * @param size 数组大小
 * @return 最高概率的类别 ID
 */
int EmotionPostprocess_GetMaxClassId(const float* probabilities, size_t size);

/**
 * @brief 获取最高概率值
 * @param probabilities 概率数组
 * @param size 数组大小
 * @return 最高概率值
 */
float EmotionPostprocess_GetMaxProbability(const float* probabilities, size_t size);

/**
 * @brief 检查置信度是否超过阈值
 * @param confidence 置信度
 * @param threshold 阈值
 * @return true 超过阈值，false 未超过
 */
bool EmotionPostprocess_CheckConfidence(float confidence, float threshold);

/**
 * @brief 计算熵值（不确定性度量）
 * @param probabilities 概率数组
 * @param size 数组大小
 * @return 熵值
 */
float EmotionPostprocess_CalculateEntropy(const float* probabilities, size_t size);

#ifdef __cplusplus
}
#endif

#endif /* EMOTION_POSTPROCESS_H */