"""
convert_to_tflite.py
将 PyTorch 模型转换为 TFLite 格式
使用 ONNX 作为中间格式
"""

import os
import numpy as np
from pathlib import Path

# 配置
IMG_HEIGHT = 96
IMG_WIDTH = 96
NUM_CLASSES = 3
MODEL_PATH = "models/best_model.pth"
OUTPUT_DIR = "models"

CLASS_NAMES = ['cry', 'happy', 'angry']


def convert_pytorch_to_onnx(onnx_path):
    """将 PyTorch 模型转换为 ONNX"""
    import io
    import sys
    # 修复 Windows 控制台编码问题
    old_stdout = sys.stdout
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch not found!")
        return False
    finally:
        sys.stdout = old_stdout
    
    # 定义简单的 CNN 模型结构
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=NUM_CLASSES):
            super().__init__()
            
            self.features = nn.Sequential(
                # 卷积块 1
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.25),
                
                # 卷积块 2
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.25),
                
                # 卷积块 3
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.25),
            )
            
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 12 * 12, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
    
    # 加载模型
    print(f"Loading PyTorch model from {MODEL_PATH}...")
    model = SimpleCNN()
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 导出为 ONNX
    print(f"Exporting to ONNX: {onnx_path}")
    dummy_input = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH)
    
    # 使用旧版 API 避免编码问题
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        input_names=['input'],
        output_names=['output'],
        dynamo=False
    )
    
    print(f"ONNX model saved to {onnx_path}")
    print(f"Model size: {os.path.getsize(onnx_path) / 1024:.2f} KB")
    return True


def convert_onnx_to_tflite(onnx_path, tflite_path):
    """将 ONNX 模型转换为 TFLite"""
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow not found, skipping TFLite conversion")
        print("Install with: pip install tensorflow")
        return False
    
    try:
        print(f"Converting ONNX to TFLite: {tflite_path}")
        converter = tf.lite.TFLiteConverter.from_onnx(onnx_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model saved to {tflite_path}")
        print(f"Model size: {os.path.getsize(tflite_path) / 1024:.2f} KB")
        return True
    except Exception as e:
        print(f"TFLite conversion failed: {e}")
        return False


def verify_tflite_model(tflite_path):
    """验证 TFLite 模型"""
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow not found, skipping verification")
        return False
    
    print(f"Verifying TFLite model: {tflite_path}")
    
    # 加载 TFLite 模型
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # 获取输入输出详情
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    
    # 测试推理
    test_input = np.random.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"Output: {output}")
    
    return True


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    onnx_path = os.path.join(OUTPUT_DIR, "emotion_model.onnx")
    tflite_path = os.path.join(OUTPUT_DIR, "emotion_model.tflite")
    
    # 步骤 1: PyTorch -> ONNX
    if not convert_pytorch_to_onnx(onnx_path):
        print("Failed to convert to ONNX")
        return
    
    # 步骤 2: ONNX -> TFLite
    if convert_onnx_to_tflite(onnx_path, tflite_path):
        # 步骤 3: 验证模型
        verify_tflite_model(tflite_path)
    
    print("\nConversion complete!")
    print(f"ONNX model: {onnx_path}")
    print(f"TFLite model: {tflite_path}")


if __name__ == '__main__':
    main()