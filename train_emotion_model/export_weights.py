"""
export_weights.py
将 PyTorch 模型权重导出为 C 语言头文件
用于 ESP32-S3 嵌入式部署
"""

import os
import torch
import numpy as np

# 配置
MODEL_PATH = "models/best_model.pth"
OUTPUT_HEADER = "models/emotion_model_weights.h"
NUM_CLASSES = 3


class SimpleCNN(torch.nn.Module):
    """与训练时相同的模型结构"""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        
        self.features = torch.nn.Sequential(
            # 卷积块 1
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Dropout(0.25),
            
            # 卷积块 2
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Dropout(0.25),
            
            # 卷积块 3
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Dropout(0.25),
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 12 * 12, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def quantize_weights(weights, scale=127.0):
    """将浮点权重量化为 INT8"""
    quantized = np.round(weights * scale).clip(-128, 127).astype(np.int8)
    return quantized


def array_to_c_init(arr, var_name, max_lines=100):
    """将 numpy 数组转换为 C 语言初始化代码"""
    lines = []
    flat = arr.flatten()
    
    # 每行最多显示 max_lines 个元素
    chunk_size = max_lines
    for i in range(0, len(flat), chunk_size):
        chunk = flat[i:i+chunk_size]
        line = "    " + ", ".join(f"{int(x):4d}" for x in chunk)
        if i + chunk_size < len(flat):
            line += ","
        lines.append(line)
    
    return "\n".join(lines)


def export_weights():
    """导出模型权重"""
    print(f"Loading model from {MODEL_PATH}...")
    
    # 加载模型
    model = SimpleCNN()
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 获取所有权重
    weights = {}
    
    for name, param in model.named_parameters():
        weights[name] = param.detach().numpy()
        print(f"  {name}: {param.shape}")
    
    for name, buf in model.named_buffers():
        weights[name] = buf.detach().numpy()
        print(f"  {name}: {buf.shape}")
    
    # 生成 C 头文件
    print(f"\nGenerating C header: {OUTPUT_HEADER}")
    
    with open(OUTPUT_HEADER, 'w', encoding='utf-8') as f:
        f.write("""/*
 * emotion_model_weights.h
 * 表情识别模型权重文件 (INT8 量化)
 * 自动生成 - 不要手动编辑
 * 
 * 模型结构：SimpleCNN
 * 输入：96x96 RGB 图像
 * 输出：3 类表情概率 [cry, happy, angry]
 */

#ifndef EMOTION_MODEL_WEIGHTS_H
#define EMOTION_MODEL_WEIGHTS_H

#include <stdint.h>

// 模型配置
#define EMOTION_INPUT_WIDTH    96
#define EMOTION_INPUT_HEIGHT   96
#define EMOTION_NUM_CLASSES    3

// 类别名称
static const char* EMOTION_CLASS_NAMES[] = {
    "cry", "happy", "angry"
};

// 量化参数
#define WEIGHT_SCALE           127.0f
#define ACTIVATION_SCALE       127.0f

""")
        
        # 导出每个权重层
        layer_idx = 0
        for name, weight in weights.items():
            # 替换非法字符
            var_name = name.replace('.', '_').replace('weight', 'w').replace('bias', 'b')
            var_name = var_name.replace('features_', 'f').replace('classifier_', 'c')
            
            # 量化权重
            if 'weight' in name:
                quantized = quantize_weights(weight)
            else:
                quantized = weight.astype(np.float32)
            
            size = quantized.size
            dtype = "int8_t" if 'weight' in name else "float32_t"
            
            f.write(f"// Layer {layer_idx}: {name}\n")
            f.write(f"static const {dtype} {var_name}[{size}] = {{\n")
            
            if 'weight' in name:
                # 量化权重
                flat = quantized.flatten()
                for i in range(0, len(flat), 20):
                    chunk = flat[i:i+20]
                    line = "    " + ", ".join(f"{int(x):4d}" for x in chunk)
                    if i + 20 < len(flat):
                        line += ","
                    f.write(line + "\n")
            else:
                # 偏置和 BN 参数保持浮点
                flat = quantized.flatten()
                for i in range(0, len(flat), 10):
                    chunk = flat[i:i+10]
                    line = "    " + ", ".join(f"{x:12.6f}" for x in chunk)
                    if i + 10 < len(flat):
                        line += ","
                    f.write(line + "\n")
            
            f.write("};\n\n")
            layer_idx += 1
        
        f.write("""
#endif // EMOTION_MODEL_WEIGHTS_H
""")
    
    file_size = os.path.getsize(OUTPUT_HEADER)
    print(f"Header file created: {OUTPUT_HEADER} ({file_size / 1024:.2f} KB)")
    return True


if __name__ == '__main__':
    export_weights()