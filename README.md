# ESP32-S3 表情识别系统

基于 ESP32-S3 的实时表情识别系统，支持三种表情检测：哭 (cry)、笑 (happy)、生气 (angry)。

## 📋 项目简介

本项目实现了一个完整的嵌入式表情识别系统，包括：
- PyTorch 模型训练
- 模型导出 (ONNX/C 权重)
- C 语言推理库
- ESP32-S3 嵌入式部署
- **PC 端网页实时推理（新增）**

## ✨ 主要功能

### 嵌入式端
- 📷 实时摄像头采集 (OV2640)
- 🧠 深度学习表情识别 (SimpleCNN)
- 📡 WiFi 无线传输
- 🌐 Web 界面实时显示
- 🔧 支持 SD 卡存储模型

### PC 端网页（新增）
- 🖥️ Flask HTTP 服务器
- 📸 摄像头实时识别
- 🖼️ 图片上传识别
- 😊 表情 emoji 显示
- 💬 推荐话语反馈
- 📜 历史记录（最近 10 条）

## 🏗️ 项目结构

```
rlsb/
├── README.md                          # 项目说明文档
├── .gitignore                         # Git 忽略文件
│
├── emotion_inference/                 # C 语言推理库
│   ├── CMakeLists.txt                 # CMake 构建配置
│   ├── README.md                      # 推理库说明
│   ├── include/
│   │   ├── emotion_types.h            # 数据类型定义
│   │   ├── emotion_inference_task.h   # 推理任务 API
│   │   ├── emotion_preprocess.h       # 图像预处理
│   │   └── emotion_postprocess.h      # Softmax 后处理
│   └── src/
│       ├── emotion_inference_task.c   # 推理任务实现
│       ├── emotion_preprocess.c       # 预处理实现
│       └── emotion_postprocess.c      # 后处理实现
│
├── train_emotion_model/               # 模型训练工具
│   ├── train_emotion_torch.py         # PyTorch 训练脚本
│   ├── train_emotion.py               # TensorFlow 训练脚本
│   ├── convert_to_tflite.py           # ONNX/TFLite 转换
│   ├── export_weights.py              # C 权重导出
│   ├── prepare_fer2013.py             # FER2013 数据准备
│   ├── prepare_data.py                # 数据预处理
│   ├── test_model.py                  # 模型测试
│   ├── emotion_server.py              # Flask 推理服务器 ⭐新增
│   ├── emotion_web/                   # 网页前端 ⭐新增
│   │   └── index.html                 # 网页界面
│   └── models/
│       └── best_model.pth             # 训练好的模型
│
├── models/                            # 导出模型文件
│   ├── best_model.pth                 # PyTorch 模型
│   ├── emotion_model.onnx             # ONNX 模型
│   └── emotion_model_weights.h        # C 权重头文件
│
├── fer2013_3class/                    # 训练数据集
│   ├── train/
│   │   ├── cry/                       # 哭表情
│   │   ├── happy/                     # 笑表情
│   │   └── angry/                     # 生气表情
│   └── val/
│       ├── cry/
│       ├── happy/
│       └── angry/
│
└── fer2013_data/                      # 原始 FER2013 数据
```

## 🚀 快速开始

### 1. 环境准备

```bash
# Python 环境
python --version  # 需要 Python 3.8+

# 安装依赖
pip install torch torchvision numpy opencv-python matplotlib flask
```

### 2. 数据准备

```bash
# 准备 FER2013 数据集
python train_emotion_model/prepare_fer2013.py --input fer2013_data --output fer2013_3class
```

### 3. 模型训练

```bash
# 使用 PyTorch 训练
python train_emotion_model/train_emotion_torch.py --data_dir fer2013_3class --train --simple

# 训练参数
# --epochs 50          # 训练轮数
# --batch_size 32      # 批次大小
# --lr 0.001           # 学习率
```

### 4. 模型测试

```bash
# 测试模型性能
python train_emotion_model/test_model.py --image test_samples/sample.jpg
```

### 5. 导出模型

```bash
# 导出为 ONNX 格式
python train_emotion_model/convert_to_tflite.py

# 导出为 C 权重头文件
python train_emotion_model/export_weights.py
```

## 🌐 PC 端网页测试（新增）

### 启动服务器

```bash
cd train_emotion_model
python emotion_server.py
```

服务器启动后会在控制台显示访问地址：
```
==================================================
Emotion Recognition Server
==================================================
Server URL: http://0.0.0.0:5000
Open in browser to start using!
==================================================
```

### 访问网页

打开浏览器访问 **http://127.0.0.1:5000**

### 功能说明

#### 1. 摄像头实时识别
- 点击"开启摄像头"按钮
- 系统每 1 秒自动识别一次表情
- 实时显示识别结果和置信度

#### 2. 图片上传识别
- 切换到"上传图片"标签
- 选择图片文件
- 点击"开始识别"按钮

#### 3. 识别结果展示
- 😊 **表情 emoji** - 直观显示识别结果
- 📊 **置信度百分比** - 三种表情的概率分布
- 💬 **推荐话语** - 根据表情给出温暖回应

| 表情 | Emoji | 推荐话语 |
|------|-------|----------|
| happy | 😊 | 你看起来很开心！分享一下你的快乐吧！ |
| cry | 😢 | 一切都会好起来的，加油！ |
| angry | 😠 | 深呼吸，冷静一下，别生气。 |

#### 4. 历史记录
- 自动保存最近 10 条识别记录
- 显示表情、置信度和时间
- 页面刷新后清空

## 📊 模型信息

| 属性 | 值 |
|------|-----|
| 架构 | SimpleCNN |
| 输入尺寸 | 96x96 RGB |
| 输出类别 | 3 (cry, happy, angry) |
| 参数量 | ~2.1M |
| 模型大小 | ~10MB (ONNX), ~16MB (C 权重) |
| 量化 | INT8 |

## 🔧 API 文档

### C 语言推理库

```c
// 初始化
void emotion_inference_init(void);

// 推理
emotion_result_t emotion_inference_infer(uint8_t* image, int width, int height);

// 获取结果
const char* emotion_get_class_name(emotion_class_t cls);
float emotion_get_confidence(emotion_class_t cls);
```

### Python 训练 API

```python
# 训练模型
train_model(data_dir='fer2013_3class', epochs=50, batch_size=32)

# 导出权重
export_weights(model_path='models/best_model.pth', output='models/emotion_model_weights.h')
```

### Flask 服务器 API

```python
# POST /predict - 图片上传推理
# 请求体：{"image": "base64_encoded_image"}
# 响应：{"emotion": "happy", "confidence": 0.85, "probabilities": {...}}

# GET /stream - 摄像头视频流 (MJPEG)
# GET /history - 获取历史记录
# POST /clear_history - 清空历史记录
```

## 📦 硬件要求

### ESP32-S3 开发板
- ESP32-S3-WROOM-1 (8MB PSRAM 或 16MB PSRAM)
- OV2640 摄像头模块
- MicroSD 卡座 (可选，用于存储模型)

### 连接方式

#### SDIO 模式 (推荐)
| ESP32-S3 | SD 卡 |
|----------|------|
| GPIO 12  | CMD  |
| GPIO 11  | CLK  |
| GPIO 13  | D0   |
| GPIO 14  | D1   |
| GPIO 15  | D2   |
| GPIO 16  | D3   |

#### SPI 模式
| ESP32-S3 | SD 卡 |
|----------|------|
| GPIO 10  | CS   |
| GPIO 11  | CLK  |
| GPIO 12  | MISO |
| GPIO 13  | MOSI |

## 📝 依赖项

### Python
- torch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.21.0
- opencv-python >= 4.5.0
- matplotlib >= 3.5.0
- Pillow >= 9.0.0
- **flask >= 3.0.0** ⭐新增

### ESP-IDF
- ESP-IDF v5.0+
- CMake 3.16+

## 📄 许可证

MIT License

## 🙏 致谢

- [FER2013 数据集](https://www.kaggle.com/datasets/msambare/fer2013)
- [ESP-IDF](https://docs.espressif.com/projects/esp-idf/)
- [PyTorch](https://pytorch.org/)

## 📮 联系方式

如有问题，请提交 Issue 或联系开发者。
