# Emotion Inference Component for ESP32-S3
# TFLite Micro based emotion recognition

cmake_minimum_required(VERSION 3.16)

set(COMPONENT_NAME emotion_inference)

# Source files
set(COMPONENT_SRCS
    "src/emotion_inference_task.c"
    "src/emotion_preprocess.c"
    "src/emotion_postprocess.c"
)

# Include directories
set(COMPONENT_ADD_INCLUDEDIRS
    "include"
)

# Dependencies
set(COMPONENT_REQUIRES
    esp_timer
    esp_lcd
    esp_lcd_ili9341
    esp_camera
    tflite_micro
)

# Register component
register_component()

# Target-specific settings
target_compile_definitions(${COMPONENT_TARGET} PRIVATE
    CONFIG_EMOTION_INPUT_WIDTH=96
    CONFIG_EMOTION_INPUT_HEIGHT=96
    CONFIG_EMOTION_INPUT_CHANNELS=3
)

# Link libraries
target_link_libraries(${COMPONENT_TARGET} PUBLIC
    -lm
)