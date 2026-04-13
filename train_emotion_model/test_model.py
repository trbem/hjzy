"""
test_model.py
模型测试脚本 - 评估表情识别模型性能

功能：
- 在验证集上评估准确率
- 显示混淆矩阵
- 可视化错误样本
- 测试单张图片推理
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError:
    print("Installing matplotlib...")
    os.system("pip install matplotlib")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

try:
    from sklearn.metrics import confusion_matrix, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not found. Confusion matrix will not be available.")
    print("Install with: pip install scikit-learn")

# 配置
IMG_HEIGHT = 96
IMG_WIDTH = 96
NUM_CLASSES = 3
BATCH_SIZE = 32

# 表情类别
CLASS_NAMES = ['cry', 'happy', 'angry']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}


class EmotionDataset(Dataset):
    """表情识别数据集"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        # 支持 train/val 结构
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
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, str(img_path)


def create_simple_cnn(num_classes=NUM_CLASSES):
    """创建简单的 CNN 模型"""
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


def create_mobilenet(num_classes=NUM_CLASSES):
    """创建 MobileNetV2 模型"""
    model = models.mobilenet_v2(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, num_classes)
    )
    return model


def evaluate_model(model_path, data_dir, use_simple=False):
    """
    在验证集上评估模型
    
    Args:
        model_path: 模型文件路径
        data_dir: 数据目录
        use_simple: 是否使用简单 CNN 架构
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    print(f"\nLoading model from {model_path}")
    
    if use_simple:
        model = create_simple_cnn()
    else:
        model = create_mobilenet()
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded. Validation accuracy at save: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    
    # 数据变换 (验证集变换)
    val_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载验证集
    print(f"\nLoading validation data from {data_dir}...")
    val_dataset = EmotionDataset(data_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 评估
    all_preds = []
    all_labels = []
    all_probs = []
    correct_samples = defaultdict(list)
    error_samples = defaultdict(list)
    
    total = 0
    correct = 0
    total_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    print("\nEvaluating...")
    with torch.no_grad():
        for images, labels, paths in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # 记录正确和错误样本
            for i in range(len(labels)):
                pred = predicted[i].item()
                label = labels[i].item()
                path = paths[i]
                
                if pred == label:
                    correct_samples[label].append((path, probs[i][pred].item()))
                else:
                    error_samples[label].append((path, pred, probs[i][pred].item()))
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    
    print(f"\n{'='*50}")
    print(f"Validation Results:")
    print(f"{'='*50}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    # 各类别准确率
    print(f"\nPer-class Accuracy:")
    print(f"{'-'*50}")
    
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for label, pred in zip(all_labels, all_preds):
        class_total[label] += 1
        if label == pred:
            class_correct[label] += 1
    
    for idx, class_name in enumerate(CLASS_NAMES):
        c = class_correct[idx]
        t = class_total[idx]
        acc = 100.0 * c / t if t > 0 else 0
        print(f"  {class_name:10s}: {acc:6.2f}% ({c}/{t})")
    
    # 混淆矩阵
    if HAS_SKLEARN:
        print(f"\nConfusion Matrix:")
        print(f"{'-'*50}")
        cm = confusion_matrix(all_labels, all_preds)
        print(cm)
        
        # 可视化混淆矩阵
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=CLASS_NAMES,
               yticklabels=CLASS_NAMES,
               title='Confusion Matrix',
               ylabel='True Label',
               xlabel='Predicted Label')
        
        # 在单元格中显示数值
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150)
        print(f"\nConfusion matrix saved to: confusion_matrix.png")
    
    # 分类报告
    if HAS_SKLEARN:
        print(f"\nClassification Report:")
        print(f"{'-'*50}")
        report = classification_report(all_labels, all_preds, 
                                       target_names=CLASS_NAMES,
                                       digits=4)
        print(report)
    
    # 显示错误样本
    print(f"\nError Analysis:")
    print(f"{'-'*50}")
    total_errors = sum(len(v) for v in error_samples.values())
    print(f"Total errors: {total_errors}")
    
    # 保存错误样本列表
    with open('error_samples.txt', 'w') as f:
        for true_label, errors in error_samples.items():
            for error in errors:
                path = error[0]
                pred = error[1]
                prob = error[2]
                f.write(f"True: {CLASS_NAMES[true_label]}, "
                       f"Predicted: {CLASS_NAMES[pred]}, "
                       f"Prob: {prob:.4f}, Path: {path}\n")
    print(f"Error samples saved to: error_samples.txt")
    
    # 可视化一些错误样本
    visualize_errors(error_samples, num_samples=9)
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'class_accuracy': {CLASS_NAMES[i]: 100.0 * class_correct[i] / class_total[i] 
                          for i in range(NUM_CLASSES) if class_total[i] > 0},
        'confusion_matrix': cm if HAS_SKLEARN else None
    }


def visualize_errors(error_samples, num_samples=9):
    """可视化错误样本"""
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    samples = []
    for true_label, errors in error_samples.items():
        for error in errors:
            samples.append((true_label, error))
    
    # 随机选择 num_samples 个错误样本
    np.random.seed(42)
    if len(samples) > num_samples:
        indices = np.random.choice(len(samples), num_samples, replace=False)
        samples = [samples[i] for i in indices]
    else:
        num_samples = len(samples)
    
    for i, (true_label, error) in enumerate(samples):
        if i >= 9:
            break
        
        path = error[0]
        pred = error[1]
        prob = error[2]
        
        # 加载图像
        img = Image.open(path).convert('RGB')
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        
        axes[i].imshow(img)
        axes[i].set_title(f"True: {CLASS_NAMES[true_label]}\n"
                         f"Pred: {CLASS_NAMES[pred]} ({prob:.2f})")
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for j in range(i + 1, 9):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig('error_samples_visualization.png', dpi=150)
    print(f"Error visualization saved to: error_samples_visualization.png")
    plt.close()


def test_single_image(model_path, image_path, use_simple=False):
    """
    测试单张图片推理
    
    Args:
        model_path: 模型文件路径
        image_path: 图像文件路径
        use_simple: 是否使用简单 CNN 架构
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    if use_simple:
        model = create_simple_cnn()
    else:
        model = create_mobilenet()
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 加载图像
    print(f"Loading image: {image_path}")
    img = Image.open(image_path).convert('RGB')
    original_img = img.copy() if hasattr(img, 'copy') else img
    
    # 变换
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidences = probs.cpu().numpy()[0]
    
    # 显示结果
    print(f"\n{'='*50}")
    print(f"Prediction Results:")
    print(f"{'='*50}")
    
    # 按置信度排序
    sorted_idx = np.argsort(confidences)[::-1]
    
    for idx in sorted_idx:
        class_name = CLASS_NAMES[idx]
        confidence = confidences[idx] * 100
        print(f"  {class_name:10s}: {confidence:5.2f}%")
    
    # 获取预测结果
    pred_idx = sorted_idx[0]
    pred_class = CLASS_NAMES[pred_idx]
    pred_conf = confidences[pred_idx]
    
    # 根据表情生成话语
    if pred_class == 'happy':
        message = "你看起来很开心！分享一下你的快乐吧！"
    elif pred_class == 'cry':
        message = "一切都会好起来的，加油！"
    else:  # angry
        message = "深呼吸，冷静一下，别生气。"
    
    print(f"\nSuggested Response:")
    print(f"  \"{message}\"")
    
    return pred_class, pred_conf, message


def main():
    parser = argparse.ArgumentParser(description='Test emotion recognition model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model file (.pth)')
    parser.add_argument('--data_dir', type=str, default='fer2013_3class',
                        help='Path to data directory')
    parser.add_argument('--image', type=str,
                        help='Path to single image for inference')
    parser.add_argument('--simple', action='store_true',
                        help='Use simple CNN instead of MobileNetV2')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    if args.image:
        # 单图推理模式
        if not os.path.exists(args.image):
            print(f"Error: Image file not found: {args.image}")
            sys.exit(1)
        test_single_image(args.model_path, args.image, args.simple)
    else:
        # 完整评估模式
        if not os.path.exists(args.data_dir):
            print(f"Error: Data directory not found: {args.data_dir}")
            sys.exit(1)
        evaluate_model(args.model_path, args.data_dir, args.simple)


if __name__ == '__main__':
    main()