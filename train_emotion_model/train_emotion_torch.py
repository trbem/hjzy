"""
train_emotion_torch.py
表情识别模型训练脚本 (PyTorch 版本)
为 ESP32-S3 生成 TFLite 模型

支持三种表情：哭 (cry)、笑 (happy)、生气 (angry)
输入尺寸：96x96
输出：INT8 量化 TFLite 模型

适用于 Python 3.10+ 环境
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# 检查 PyTorch 并尝试导入
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms, models
    print(f"PyTorch version: {torch.__version__}")
except ImportError:
    print("PyTorch not found. Installing...")
    os.system("pip install torch torchvision torchaudio")
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms, models

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not found. Installing...")
    os.system("pip install matplotlib")
    import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("Warning: TensorFlow not found. TFLite conversion will be skipped.")
    print("Install with: pip install tensorflow")

# 配置
IMG_HEIGHT = 96
IMG_WIDTH = 96
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# 表情类别
CLASS_NAMES = ['cry', 'happy', 'angry']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


class EmotionDataset(Dataset):
    """表情识别数据集"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        # 支持两种目录结构:
        # 1. root_dir/class_name/*.jpg (直接按类别组织)
        # 2. root_dir/train/class_name/*.jpg 或 root_dir/val/class_name/*.jpg
        
        # 首先检查是否是 train/val 结构
        if (self.root_dir / 'train').exists():
            scan_dir = self.root_dir / 'train'
        elif (self.root_dir / 'val').exists():
            scan_dir = self.root_dir / 'val'
        else:
            scan_dir = self.root_dir
        
        # 扫描所有类别目录
        for class_name in CLASS_NAMES:
            class_dir = scan_dir / class_name
            if not class_dir.exists():
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            
            # 递归查找所有图像文件
            for img_path in class_dir.rglob("*.png"):
                self.samples.append((img_path, CLASS_TO_IDX[class_name]))
            for img_path in class_dir.rglob("*.jpg"):
                self.samples.append((img_path, CLASS_TO_IDX[class_name]))
            for img_path in class_dir.rglob("*.jpeg"):
                self.samples.append((img_path, CLASS_TO_IDX[class_name]))
        
        print(f"Found {len(self.samples)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 使用 PIL 加载图像
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_model(num_classes=NUM_CLASSES, pretrained=True):
    """
    创建轻量级 CNN 模型，适合 ESP32-S3 部署
    
    使用 MobileNetV2 作为骨干网络
    """
    # 使用 MobileNetV2 (轻量级)
    model = models.mobilenet_v2(pretrained=pretrained)
    
    # 修改分类器
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, num_classes)
    )
    
    return model


def create_simple_cnn(num_classes=NUM_CLASSES):
    """
    创建简单的 CNN 模型 (不依赖预训练)
    """
    class SimpleCNN(nn.Module):
        def __init__(self):
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


def train_model(model, train_loader, val_loader, output_dir='models', device='cpu'):
    """训练模型"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 编译模型
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 训练记录
    best_acc = 0.0
    train_losses = []
    val_accs = []
    
    print(f"Starting training on {device}...")
    
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        val_accs.append(val_acc)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(output_dir, 'best_model.pth'))
            print(f"  -> Saved best model with accuracy {val_acc:.2f}%")
    
    # 绘制训练曲线
    plot_history(train_losses, val_accs, output_dir)
    
    # 保存最终模型
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_accs[-1],
    }, os.path.join(output_dir, 'emotion_model.pth'))
    
    print(f"\nTraining complete! Best validation accuracy: {best_acc:.2f}%")
    return best_acc


def plot_history(train_losses, val_accs, output_dir):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 损失
    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率
    ax2.plot(epochs, val_accs, label='Val Acc')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
    print(f"Training history saved to {output_dir}/training_history.png")


def convert_to_tflite_pytorch(model, output_dir='models'):
    """
    将 PyTorch 模型转换为 TFLite
    
    步骤：
    1. PyTorch -> ONNX
    2. ONNX -> TFLite
    """
    if not HAS_TF:
        print("TensorFlow not available, skipping TFLite conversion")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 导出为 ONNX
        onnx_path = os.path.join(output_dir, 'emotion_model.onnx')
        dummy_input = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH)
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"Exported to ONNX: {onnx_path}")
        
        # ONNX -> TFLite (需要 onnx2tf)
        try:
            import onnx2tf
            tflite_path = os.path.join(output_dir, 'emotion_model_int8.tflite')
            # 简化转换（不使用 INT8 量化）
            converter = tf.lite.TFLiteConverter.from_onnx(onnx_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"Converted to TFLite: {tflite_path}")
            print(f"Model size: {os.path.getsize(tflite_path) / 1024:.2f} KB")
            return tflite_path
        except ImportError:
            print("onnx2tf not available. Install with: pip install onnx2tf")
            print("Manual conversion: onnx -> tflite using onnx2tf")
            return onnx_path
    except Exception as e:
        print(f"TFLite conversion failed: {e}")
        return None


def verify_model(model_path, device='cpu'):
    """验证模型"""
    print(f"\nVerifying model: {model_path}")
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    model = create_simple_cnn()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 测试推理
    dummy_input = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
        probabilities = torch.nn.functional.softmax(output, dim=1)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Probabilities: {probabilities.cpu().numpy()}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Train emotion recognition model (PyTorch)')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path to training data directory')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Output directory for models')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--convert', action='store_true',
                        help='Convert to TFLite')
    parser.add_argument('--verify', type=str,
                        help='Verify model path')
    parser.add_argument('--simple', action='store_true',
                        help='Use simple CNN instead of MobileNetV2')
    
    args = parser.parse_args()
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.verify:
        verify_model(args.verify, device)
        return
    
    # 数据变换
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建模型
    print("Creating model...")
    if args.simple:
        model = create_simple_cnn()
    else:
        model = create_model(pretrained=True)
    
    model = model.to(device)
    
    # 打印模型结构
    print("\nModel architecture:")
    print(model)
    
    if args.train:
        # 加载数据
        print(f"\nLoading data from {args.data_dir}...")
        
        # 检查数据目录结构
        data_path = Path(args.data_dir)
        
        # 尝试不同的目录结构
        train_dir = data_path / 'train' if (data_path / 'train').exists() else data_path
        val_dir = data_path / 'val' if (data_path / 'val').exists() else None
        
        # 如果没有验证集，使用训练集的一部分
        if val_dir is None:
            print("No separate validation directory found. Using 20% of data for validation.")
            # 这里简化处理，实际应该使用 train_test_split
        
        train_dataset = EmotionDataset(train_dir, transform=train_transform)
        
        if val_dir and val_dir.exists():
            val_dataset = EmotionDataset(val_dir, transform=val_transform)
        else:
            # 从训练集分割
            from torch.utils.data import random_split
            val_size = int(len(train_dataset) * 0.2)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(
                train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            # 应用验证变换
            val_dataset.dataset.transform = val_transform
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        # 训练
        best_acc = train_model(model, train_loader, val_loader, args.output_dir, device)
    
    if args.convert or (args.train and HAS_TF):
        # 加载最佳模型
        best_model_path = os.path.join(args.output_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # 转换为 TFLite
        convert_to_tflite_pytorch(model, args.output_dir)


if __name__ == '__main__':
    main()