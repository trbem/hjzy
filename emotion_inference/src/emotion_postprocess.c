/**
 * @file emotion_postprocess.c
 * @brief 表情识别后处理实现
 * @for ESP32-S3 + TFLite Micro
 */

#include "emotion_postprocess.h"
#include <string.h>
#include <math.h>
#include <float.h>

#define LOG2_E 1.44269504088896340736f  // log2(e)

/*============================================================================
 * 内部函数
 *===========================================================================*/
static float fast_exp(float x) {
    /* 使用泰勒级数近似 exp(x)，适合嵌入式环境 */
    const float ONE_OVER_6 = 1.0f / 6.0f;
    const float ONE_OVER_24 = 1.0f / 24.0f;
    const float ONE_OVER_120 = 1.0f / 120.0f;
    
    /* 限制 x 的范围避免溢出 */
    if (x > 10.0f) x = 10.0f;
    if (x < -10.0f) x = -10.0f;
    
    /* 泰勒级数：exp(x) ≈ 1 + x + x²/2! + x³/3! + x⁴/4! */
    return 1.0f + x + (x * x) * 0.5f + (x * x * x) * ONE_OVER_6 + 
           (x * x * x * x) * ONE_OVER_24 + (x * x * x * x * x) * ONE_OVER_120;
}

static float fast_log2(float x) {
    /* 使用自然对数近似 */
    if (x <= 0.0f) return 0.0f;
    
    /* 使用多项式近似 log(x) */
    float ln_x;
    const float LN_2 = 0.69314718056f;
    
    /* 简化近似 */
    int exponent = 0;
    while (x > 2.0f) { x *= 0.5f; exponent++; }
    while (x < 0.5f) { x *= 2.0f; exponent--; }
    
    /* 对 [0.5, 2] 范围内的 x 进行多项式近似 */
    float t = x - 1.0f;
    ln_x = t * (1.0f - t * 0.5f + t * t * 0.333f - t * t * t * 0.25f);
    
    return (exponent * LN_2 + ln_x) * LOG2_E;
}

/*============================================================================
 * API 实现
 *===========================================================================*/

int EmotionPostprocess_Softmax(float* logits, size_t size) {
    if (logits == NULL || size == 0) {
        return -1;
    }
    
    /* 找到最大值，用于数值稳定性 */
    float max_val = logits[0];
    for (size_t i = 1; i < size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }
    
    /* 计算 exp(x - max) 并求和 */
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        logits[i] = fast_exp(logits[i] - max_val);
        sum += logits[i];
    }
    
    /* 归一化 */
    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (size_t i = 0; i < size; i++) {
            logits[i] *= inv_sum;
        }
    }
    
    return 0;
}

EmotionErrorCode_t EmotionPostprocess_SoftmaxClassify(const float* logits,
                                                       size_t size,
                                                       EmotionPostprocess_Result_t* result_out) {
    if (logits == NULL || result_out == NULL || size == 0) {
        return EMOTION_ERR_INVALID_PARAM;
    }
    
    /* 复制 logits */
    float* temp_logits = (float*)malloc(size * sizeof(float));
    if (temp_logits == NULL) {
        return EMOTION_ERR_MEMORY;
    }
    memcpy(temp_logits, logits, size * sizeof(float));
    
    /* 执行 Softmax */
    if (EmotionPostprocess_Softmax(temp_logits, size) != 0) {
        free(temp_logits);
        return EMOTION_ERR_PREPROCESS;
    }
    
    /* 找到最大概率的类别 */
    int max_class_id = 0;
    float max_prob = temp_logits[0];
    
    for (size_t i = 1; i < size; i++) {
        if (temp_logits[i] > max_prob) {
            max_prob = temp_logits[i];
            max_class_id = (int)i;
        }
    }
    
    /* 填充结果 */
    result_out->class_id = max_class_id;
    result_out->confidence = max_prob;
    memcpy(result_out->probabilities, temp_logits, size * sizeof(float));
    
    free(temp_logits);
    return EMOTION_OK;
}

int EmotionPostprocess_GetMaxClassId(const float* probabilities, size_t size) {
    if (probabilities == NULL || size == 0) {
        return -1;
    }
    
    int max_class_id = 0;
    float max_prob = probabilities[0];
    
    for (size_t i = 1; i < size; i++) {
        if (probabilities[i] > max_prob) {
            max_prob = probabilities[i];
            max_class_id = (int)i;
        }
    }
    
    return max_class_id;
}

float EmotionPostprocess_GetMaxProbability(const float* probabilities, size_t size) {
    if (probabilities == NULL || size == 0) {
        return 0.0f;
    }
    
    float max_prob = probabilities[0];
    for (size_t i = 1; i < size; i++) {
        if (probabilities[i] > max_prob) {
            max_prob = probabilities[i];
        }
    }
    
    return max_prob;
}

bool EmotionPostprocess_CheckConfidence(float confidence, float threshold) {
    return confidence >= threshold;
}

float EmotionPostprocess_CalculateEntropy(const float* probabilities, size_t size) {
    if (probabilities == NULL || size == 0) {
        return 0.0f;
    }
    
    float entropy = 0.0f;
    for (size_t i = 0; i < size; i++) {
        if (probabilities[i] > 0.0f) {
            entropy -= probabilities[i] * fast_log2(probabilities[i]);
        }
    }
    
    return entropy;
}