"""
emotion_server.py
表情识别推理服务器 - Flask + PyTorch

功能：
- HTTP API 提供表情识别服务
- 支持图片上传推理
- 支持摄像头实时流推理
- 提供网页前端界面
"""

import os
import sys
import io
import base64
import json
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from threading import Thread, Lock
from queue import Queue

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image

from flask import Flask, render_template, request, jsonify, Response, stream_with_context

# 配置
IMG_HEIGHT = 96
IMG_WIDTH = 96
NUM_CLASSES = 3
MODEL_PATH = "models/best_model.pth"
HOST = "0.0.0.0"
PORT = 5000

# 表情类别
CLASS_NAMES = ['cry', 'happy', 'angry']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

# 表情对应的 emoji 和话语
EMOTION_CONFIG = {
    'happy': {
        'emoji': '😊',
        'message': '你看起来很开心！分享一下你的快乐吧！',
        'color': '#4CAF50'
    },
    'cry': {
        'emoji': '😢',
        'message': '一切都会好起来的，加油！',
        'color': '#2196F3'
    },
    'angry': {
        'emoji': '😠',
        'message': '深呼吸，冷静一下，别生气。',
        'color': '#F44336'
    },
    'unknown': {
        'emoji': '😐',
        'message': '正在识别你的表情...',
        'color': '#9E9E9E'
    }
}

# 全局变量
app = Flask(__name__, template_folder='emotion_web')
model = None
device = None
model_lock = Lock()

# 历史记录（内存存储，最多 10 条）
history_queue = Queue(maxsize=10)
history_lock = Lock()


def create_simple_cnn(num_classes=NUM_CLASSES):
    """创建简单的 CNN 模型（与训练时一致）"""
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
    
    return SimpleCNN()


def load_model(model_path):
    """加载 PyTorch 模型"""
    global model, device
    
    print(f"Loading model from {model_path}...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    model = create_simple_cnn()
    
    # 加载权重
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded. Previous accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    else:
        print(f"Warning: Model file not found: {model_path}")
        print("Using random weights for demo purposes.")
    
    model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_pil):
    """预处理图像"""
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image_pil).unsqueeze(0).to(device)
    return img_tensor


def predict_emotion(image_tensor):
    """推理单张图像"""
    global model, device
    
    with model_lock:
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            confidences = probs.cpu().numpy()[0]
    
    # 获取预测结果
    pred_idx = np.argmax(confidences)
    pred_class = IDX_TO_CLASS[pred_idx]
    pred_conf = confidences[pred_idx]
    
    # 获取表情配置
    emotion_info = EMOTION_CONFIG.get(pred_class, EMOTION_CONFIG['unknown'])
    
    return {
        'emotion': pred_class,
        'confidence': float(pred_conf),
        'probabilities': {
            CLASS_NAMES[i]: float(confidences[i]) 
            for i in range(NUM_CLASSES)
        },
        'emoji': emotion_info['emoji'],
        'message': emotion_info['message'],
        'color': emotion_info['color']
    }


def predict_from_base64(base64_string):
    """从 base64 编码的图片进行推理"""
    try:
        # 解码 base64
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # 预处理
        img_tensor = preprocess_image(image_pil)
        
        # 推理
        result = predict_emotion(img_tensor)
        result['timestamp'] = datetime.now().isoformat()
        
        # 添加到历史记录
        add_to_history(result)
        
        return result
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def add_to_history(result):
    """添加记录到历史"""
    with history_lock:
        if history_queue.full():
            history_queue.get()
        history_queue.put(result.copy())


def get_history():
    """获取历史记录"""
    with history_lock:
        return list(history_queue.queue)


def gen_camera_stream():
    """生成摄像头视频流"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    last_predict_time = 0
    predict_interval = 0.5  # 每 0.5 秒推理一次
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 定时推理
        current_time = time.time()
        if current_time - last_predict_time >= predict_interval:
            # 转换格式
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # 预处理
            img_tensor = preprocess_image(frame_pil)
            
            # 推理
            result = predict_emotion(img_tensor)
            last_predict_time = current_time
            
            # 添加到历史记录
            add_to_history(result)
        else:
            # 使用上次结果
            with history_lock:
                result = history_queue.queue[-1] if not history_queue.empty() else None
        
        # 在帧上绘制结果
        if result:
            # 绘制表情图标和话语
            cv2.putText(frame, f"{result['emoji']} {result['emotion'].upper()}", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, result['message'], 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {result['confidence']*100:.1f}%", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 绘制边框
            color = tuple(int(c) for c in bytes(result['color']))
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 130), color, -1)
        
        # 编码为 JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        
        # 生成 MJPEG 流
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()


# Flask 路由

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """图片上传推理接口"""
    try:
        # 检查请求类型
        if 'file' in request.files:
            # 文件上传
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # 读取图片
            image_pil = Image.open(file.stream).convert('RGB')
            img_tensor = preprocess_image(image_pil)
            result = predict_emotion(img_tensor)
            result['timestamp'] = datetime.now().isoformat()
            add_to_history(result)
            
        elif 'image' in request.json:
            # Base64 图片数据
            result = predict_from_base64(request.json['image'])
            if result is None:
                return jsonify({'error': 'Failed to process image'}), 400
        else:
            return jsonify({'error': 'No image data provided'}), 400
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/stream')
def stream():
    """摄像头实时流接口"""
    return Response(gen_camera_stream(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/history', methods=['GET'])
def get_history_api():
    """获取历史记录 API"""
    return jsonify(get_history())


@app.route('/clear_history', methods=['POST'])
def clear_history():
    """清空历史记录"""
    with history_lock:
        while not history_queue.empty():
            history_queue.get()
    return jsonify({'status': 'ok'})


def main():
    """主函数"""
    global model
    
    # 获取模型路径（支持相对路径）
    script_dir = Path(__file__).parent
    model_full_path = script_dir / MODEL_PATH
    
    # 加载模型
    load_model(str(model_full_path))
    
    # 启动 Flask 服务器
    print(f"\n{'='*50}")
    print(f"Emotion Recognition Server")
    print(f"{'='*50}")
    print(f"Server URL: http://{HOST}:{PORT}")
    print(f"Open in browser to start using!")
    print(f"{'='*50}\n")
    
    app.run(host=HOST, port=PORT, debug=False, threaded=True)


if __name__ == '__main__':
    main()