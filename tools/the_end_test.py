from mmdeploy_runtime import Segmentor
import cv2
import numpy as np
from PIL import Image
import os
import math

# 定义图片和掩码文件夹的路径
image_folder = '../data/all'
result_folder = '../test_result'

# 获取文件夹中的所有图片文件名
image_files = os.listdir(image_folder)

# create a classifier
segmentor = Segmentor(model_path='../work_dirs/segformer_tensorrt_fp16_static-512x512_area', device_name='cuda', device_id=0)
i = 0
# 遍历所有图片和掩码文件
for image_filename in image_files:
    i += 1
    # 读取图片
    image = Image.open(os.path.join(image_folder, image_filename))

    # 将图片转换为numpy数组
    image_np = np.array(image)

    # perform inference
    annotations_array = segmentor(image_np)
    annotations_array = annotations_array.astype(np.uint8)

    # 获取标签为4的区域掩码
    mask_tab_4 = annotations_array == 1

    # 使用连通组件分析计算独立连通区域
    num_labels, labels = cv2.connectedComponents(np.uint8(mask_tab_4))

    # 将标签为4的区域转换为红色
    image_np[mask_tab_4] = [255, 0, 0]  # RGB红色

    # 使用cv2.findContours()函数查找轮廓
    contours, _ = cv2.findContours(np.uint8(mask_tab_4), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历每个轮廓并绘制边界框
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 绘制绿色边界框

    # 将处理后的图片保存
    result_image = Image.fromarray(image_np)
    result_image.save(os.path.join(result_folder, f'result_{i}.jpg'))

    # 打印独立连通区域数量
    print(f"图像 {i+1} 中标签为4的独立区域数量: {num_labels - 1}")
