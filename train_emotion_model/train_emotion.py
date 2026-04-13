"""
train_emotion.py
表情识别模型训练脚本
为 ESP32-S3 生成 TFLite 模型

支持三种表情：哭 (cry)、笑 (happy)、生气 (angry)
输入尺寸：96x96
输出：INT8 量化 TFLite 模型
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.lite import Optimizations

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 配置
IMG_HEIGHT = 96
IMG_WIDTH = 96
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 50

# 表情类别
CLASS_NAMES = ['cry', 'happy', 'angry']

# 数据增强配置
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
], name="data_augmentation")


def create_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES):
    """
    创建轻量级 CNN 模型，适合 ESP32-S3 部署
    
    架构:
    - 输入：96x96x3
    - 3 个卷积块 (Conv + BatchNorm + ReLU + MaxPool)
    - 2 个全连接层
    - 输出：3 类 Softmax
    """
    model = models.Sequential([
        # 数据增强
        data_augmentation,
        
        # 输入归一化
        layers.Rescaling(1./255, input_shape=input_shape),
        
        # 卷积块 1
        layers.Conv2D(32, (3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 卷积块 2
        layers.Conv2D(64, (3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 卷积块 3
        layers.Conv2D(128, (3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 全连接层
        layers.Flatten(),
        layers.Dense(128, use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes),
    ])
    
    return model


def load_data(data_dir):
    """
    从目录加载数据
    目录结构:
    data/
    ├── cry/
    │   ├── img1.jpg
    │   └── ...
    ├── happy/
    └── angry/
    """
    print(f"Loading data from {data_dir}...")
    
    # 使用 image_dataset_from_directory 自动加载
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='int'
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='int'
    )
    
    # 优化性能
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds


def train_model(model, train_ds, val_ds, output_dir='models'):
    """训练模型"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 编译模型
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # 回调函数
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001
        ),
        ModelCheckpoint(
            os.path.join(output_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    # 训练
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # 绘制训练曲线
    plot_history(history, output_dir)
    
    return history


def plot_history(history, output_dir):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 准确率
    ax1.plot(history.history['accuracy'], label='Train Acc')
    ax1.plot(history.history['val_accuracy'], label='Val Acc')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # 损失
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()


def convert_to_tflite(model, model_path, output_dir='models'):
    """
    转换为 TFLite 格式并进行 INT8 量化
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 转换为 TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # 2. 设置代表数据集用于 INT8 量化
    def representative_dataset():
        # 需要一些样本数据进行量化
        for _ in range(100):
            image = np.random.rand(1, IMG_HEIGHT, IMG_WIDTH, 3).astype(np.float32) * 255
            yield [image]
    
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    # 3. 转换
    tflite_model = converter.convert()
    
    # 4. 保存
    tflite_path = os.path.join(output_dir, 'emotion_model_int8.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"INT8 TFLite model saved to {tflite_path}")
    print(f"Model size: {os.path.getsize(tflite_path) / 1024:.2f} KB")
    
    return tflite_path


def verify_model(tflite_path, test_image_path=None):
    """验证 TFLite 模型"""
    # 加载模型
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # 获取输入输出详情
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\nModel Verification:")
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input type: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output type: {output_details[0]['dtype']}")
    
    # 测试推理
    test_input = np.random.rand(1, IMG_HEIGHT, IMG_WIDTH, 3).astype(np.uint8) * 255
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(f"Test output: {output_data}")
    
    return True


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train emotion recognition model')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path to training data directory')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Output directory for models')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--convert', action='store_true',
                        help='Convert to TFLite')
    parser.add_argument('--verify', type=str,
                        help='Verify TFLite model path')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_model(args.verify)
        return
    
    # 创建模型
    print("Creating model...")
    model = create_model()
    model.summary()
    
    if args.train:
        # 加载数据
        train_ds, val_ds = load_data(args.data_dir)
        
        # 训练
        history = train_model(model, train_ds, val_ds, args.output_dir)
        
        # 保存 Keras 模型
        model.save(os.path.join(args.output_dir, 'emotion_model.h5'))
        print(f"Keras model saved to {args.output_dir}/emotion_model.h5")
    
    if args.convert or args.train:
        # 加载最佳模型
        best_model_path = os.path.join(args.output_dir, 'best_model.h5')
        if os.path.exists(best_model_path):
            model = keras.models.load_model(best_model_path)
        
        # 转换为 TFLite
        convert_to_tflite(model, best_model_path, args.output_dir)


if __name__ == '__main__':
    main()