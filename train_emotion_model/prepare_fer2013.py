"""
prepare_fer2013.py
下载并准备 FER-2013 数据集用于表情识别训练

FER-2013 包含 7 种表情：
- angry (生气) - 对应我们的类别
- disgust
- fear
- happy (开心) - 对应我们的类别
- sad
- surprise
- neutral

我们将使用 angry, happy 两种表情，并创建 cry 类别（使用 sad + fear 近似）
"""

import os
import sys
import csv
import numpy as np
from pathlib import Path
from PIL import Image

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print("请安装 scikit-learn: pip install scikit-learn")
    sys.exit(1)

# 配置
FER2013_URL = "https://raw.githubusercontent.com/fer2013/fer2013/master/fer2013.csv"
OUTPUT_DIR = "fer2013_data"
IMG_SIZE = 96

# 表情类别映射
# FER-2013: 0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral
# 我们的类别：cry (sad+fear), happy, angry
CLASS_MAPPING = {
    0: 'angry',    # angry -> angry
    2: 'cry',      # fear -> cry
    3: 'happy',    # happy -> happy
    4: 'cry',      # sad -> cry
}

FER2013_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


def download_fer2013(output_path):
    """下载 FER-2013 数据集"""
    import urllib.request
    
    print(f"Downloading FER-2013 dataset from {FER2013_URL}...")
    try:
        urllib.request.urlretrieve(FER2013_URL, output_path)
        print(f"Downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please download manually from: https://www.kaggle.com/datasets/msambare/fer2013")
        return False


def parse_fer2013(csv_path, output_dir):
    """解析 FER-2013 CSV 文件并保存为图像"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建类别子目录
    class_dirs = {}
    for class_name in set(CLASS_MAPPING.values()):
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        class_dirs[class_name] = class_dir
    
    print(f"Parsing {csv_path}...")
    
    image_count = 0
    skipped_count = 0
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        
        for row_idx, row in enumerate(reader):
            if len(row) < 2:
                continue
                
            pixels = row[0]
            usage = row[1] if len(row) > 1 else 'Training'
            
            # 解析表情类别
            try:
                emotion = int(pixels.split()[0]) if ' ' in pixels else int(pixels)
            except:
                continue
            
            # 检查是否在我们的映射中
            if emotion not in CLASS_MAPPING:
                skipped_count += 1
                continue
            
            class_name = CLASS_MAPPING[emotion]
            
            # 解析像素数据
            try:
                pixel_list = [int(x) for x in pixels.split()]
                image = np.array(pixel_list, dtype=np.uint8).reshape(48, 48)
            except:
                skipped_count += 1
                continue
            
            # 缩放到 96x96 使用 PIL
            image_pil = Image.fromarray(image)
            image_resized = image_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            image = np.array(image_resized)
            
            # 保存图像
            img_path = os.path.join(class_dirs[class_name], f"{usage}_{row_idx}.png")
            Image.fromarray(image).save(img_path)
            image_count += 1
            
            if image_count % 1000 == 0:
                print(f"  Processed {image_count} images...")
    
    print(f"\nCompleted!")
    print(f"  Total images: {image_count}")
    print(f"  Skipped (unused classes): {skipped_count}")
    
    # 显示各类别数量
    for class_name, class_dir in class_dirs.items():
        count = len([f for f in os.listdir(class_dir) if f.endswith('.png')])
        print(f"  {class_name}: {count} images")


def verify_dataset(output_dir):
    """验证数据集结构"""
    print(f"\nVerifying dataset at {output_dir}...")
    
    total_images = 0
    for class_name in ['angry', 'happy', 'cry']:
        class_dir = os.path.join(output_dir, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg'))])
            total_images += count
            print(f"  {class_name}: {count} images")
        else:
            print(f"  {class_name}: NOT FOUND")
    
    print(f"  Total: {total_images} images")
    return total_images > 0


def create_train_val_split(output_dir, val_ratio=0.2):
    """创建训练集和验证集子目录"""
    # 对于 FER-2013，CSV 中已经有 Training 和 PublicTest/PrivateTest 标记
    # 我们使用文件名中的 usage 标记来分离
    
    for class_name in ['angry', 'happy', 'cry']:
        class_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        # 创建 train 和 val 子目录
        train_dir = os.path.join(class_dir, 'train')
        val_dir = os.path.join(class_dir, 'val')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # 移动文件
        files = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg'))]
        train_files, val_files = train_test_split(files, test_size=val_ratio, random_state=42)
        
        for f in train_files:
            os.rename(os.path.join(class_dir, f), os.path.join(train_dir, f))
        for f in val_files:
            os.rename(os.path.join(class_dir, f), os.path.join(val_dir, f))
    
    print("Created train/val split")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare FER-2013 dataset')
    parser.add_argument('--download', action='store_true', help='Download dataset')
    parser.add_argument('--csv', type=str, help='Path to fer2013.csv')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR, help='Output directory')
    parser.add_argument('--split', action='store_true', help='Create train/val split')
    
    args = parser.parse_args()
    
    csv_path = None
    
    # 下载数据集
    if args.download:
        temp_csv = os.path.join(OUTPUT_DIR, 'fer2013.csv')
        if download_fer2013(temp_csv):
            csv_path = temp_csv
    
    # 使用提供的 CSV 路径
    if args.csv:
        csv_path = args.csv
    
    if csv_path and os.path.exists(csv_path):
        parse_fer2013(csv_path, args.output)
    else:
        print("No CSV file provided. Usage:")
        print("  python prepare_fer2013.py --download")
        print("  python prepare_fer2013.py --csv /path/to/fer2013.csv")
    
    # 验证数据集
    if verify_dataset(args.output):
        # 创建训练/验证分割
        if args.split:
            create_train_val_split(args.output)
        
        print("\nDataset preparation complete!")
        print(f"Use --output {args.output} with train_emotion.py")
    else:
        print("\nDataset verification failed!")


if __name__ == '__main__':
    main()