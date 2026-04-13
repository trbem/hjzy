"""
prepare_data.py
准备 FER-2013 数据集用于 3 类表情训练

原始类别 (7 类):
- Angry (愤怒) -> angry
- Disgust (厌恶) -> cry (近似)
- Fear (恐惧) -> cry
- Happy (幸福) -> happy
- Sad (悲伤) -> cry
- Surprise (惊讶) -> happy (近似，因为惊讶和 happy 都有积极的面部特征)
- Neutral (中性) -> cry (近似，中性表情更接近平静/悲伤)

目标类别 (3 类):
- cry (哭/悲伤): Sad + Fear + Disgust + Neutral
- happy (笑/幸福): Happy + Surprise
- angry (生气): Angry
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# 配置
SOURCE_DIR = Path("C:/Users/admin/Desktop/archive_13/Training/Training")
OUTPUT_DIR = Path("fer2013_3class")

# 类别映射
CLASS_MAPPING = {
    'Angry': 'angry',
    'Disgust': 'cry',
    'Fear': 'cry',
    'Happy': 'happy',
    'Sad': 'cry',
    'Suprise': 'happy',  # Surprise -> happy
    'Neutral': 'cry',    # Neutral -> cry
}

def prepare_dataset():
    """准备数据集"""
    print("Preparing FER-2013 dataset for 3-class emotion recognition...")
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    
    # 创建输出目录
    for class_name in ['cry', 'happy', 'angry']:
        (OUTPUT_DIR / 'train' / class_name).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'val' / class_name).mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    stats = {'train': {'cry': 0, 'happy': 0, 'angry': 0},
             'val': {'cry': 0, 'happy': 0, 'angry': 0}}
    
    # 处理训练集
    train_source = SOURCE_DIR
    if train_source.exists():
        print("\nProcessing Training set...")
        for source_class, target_class in CLASS_MAPPING.items():
            source_dir = train_source / source_class
            if not source_dir.exists():
                print(f"  Skipping {source_class} (not found)")
                continue
            
            images = list(source_dir.glob("*.*"))
            print(f"  {source_class} -> {target_class}: {len(images)} images")
            
            # 90% 训练，10% 验证
            split_idx = int(len(images) * 0.9)
            
            for i, img_path in enumerate(tqdm(images, desc=f"  {source_class}")):
                # 确定目标目录
                if i < split_idx:
                    target_dir = OUTPUT_DIR / 'train' / target_class
                else:
                    target_dir = OUTPUT_DIR / 'val' / target_class
                
                # 复制文件
                target_path = target_dir / img_path.name
                shutil.copy2(img_path, target_path)
                
                # 更新统计
                split_name = 'train' if i < split_idx else 'val'
                stats[split_name][target_class] += 1
    
    # 处理测试集
    test_source = SOURCE_DIR / 'Testing'
    if test_source.exists():
        print("\nProcessing Testing set...")
        for source_class, target_class in CLASS_MAPPING.items():
            source_dir = test_source / source_class
            if not source_dir.exists():
                continue
            
            images = list(source_dir.glob("*.*"))
            print(f"  {source_class} -> {target_class}: {len(images)} images")
            
            # 全部作为验证集
            for img_path in tqdm(images, desc=f"  {source_class}"):
                target_dir = OUTPUT_DIR / 'val' / target_class
                target_path = target_dir / f"test_{img_path.name}"
                shutil.copy2(img_path, target_path)
                stats['val'][target_class] += 1
    
    # 打印统计
    print("\n" + "="*50)
    print("Dataset Statistics:")
    print("="*50)
    for split in ['train', 'val']:
        print(f"\n{split.upper()}:")
        total = 0
        for class_name in ['cry', 'happy', 'angry']:
            count = stats[split][class_name]
            total += count
            print(f"  {class_name}: {count}")
        print(f"  Total: {total}")
    
    print("\n" + "="*50)
    print(f"Dataset prepared at: {OUTPUT_DIR}")
    print("="*50)


if __name__ == '__main__':
    prepare_dataset()