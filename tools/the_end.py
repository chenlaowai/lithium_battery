from PIL import Image, ImageDraw
import numpy as np
import os
import cv2
import math
from mmdeploy_runtime import Segmentor
import time

def filter_area_lines(mask_tab, source_lines):
    target_lines = []
    for line in source_lines:
        for x1, y1, x2, y2 in line:
            # 检查线段的起点和终点是否在标签为sealant的掩码区域中
            if mask_tab[int(y1), int(x1)] and mask_tab[int(y2), int(x2)]:
                target_lines.append(line)
    target_lines = np.array(target_lines)
    return target_lines

def filter_slope_lines(source_lines, true_slope):
    target_lines = []
    for line in source_lines:
        for x1, y1, x2, y2 in line:
            # 计算直线斜率
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
            # 根据斜率范围筛选直线
            if true_slope > 1:
                if abs(slope) > true_slope:
                    target_lines.append(line)
            else:
                if abs(slope) < true_slope:
                    target_lines.append(line)
    # 将直线列表转换为numpy数组
    target_lines = np.array(target_lines)
    return target_lines

# 筛选最边上直线，location_nums为二元数组，[0, 5]表从最靠下的五条水平直线取最长直线，[1, -5]表从最靠左的五条竖直直线取最长直线
def filter_target_line(source_lines, location_nums: list):
    length = 0.0
    target_locations = []
    for line in source_lines:
        for x1, y1, x2, y2 in line:
            if location_nums[0]:
                location = (x1 + x2) / 2
                target_locations.append(location)
            else:
                location = (y1 + y2) / 2
                target_locations.append(location)
    # 将直线位置与直线线段对应起来
    target_lines_with_locations = list(zip(target_locations, source_lines))
    # 根据直线位置进行排序(其中使用key参数来指定排序的依据，告诉sort函数按照每个元组的第一个元素（位置参数）进行排序）
    target_lines_with_locations.sort(key=lambda x: x[0], reverse=True)
    if location_nums[1] > 0:
        # 保留位置最右或下的五条直线线段
        top_lines = target_lines_with_locations[:location_nums[1]]
    else:
        # 保留位置最靠左或上的五条直线线段
        top_lines = target_lines_with_locations[location_nums[1]:]
    # 筛选出较长的直线线段
    if len(top_lines) == 0:
        return None, None
    for _, line in top_lines:
        for x1, y1, x2, y2 in line:
            if length < np.sqrt((x2 - x1)**2 + (y2 - y1)**2):
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                target_line = line
    for x1, y1, x2, y2 in target_line:
        slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
        angle_line = abs(math.degrees(math.atan(slope)))
    target_line = np.array(target_line)
    return target_line, angle_line

# 直线延申和收缩
def target_line_post_process(source_line, auxiliary_line_1, auxiliary_line_2):
    target_line = source_line
    if auxiliary_line_1 is None or auxiliary_line_2 is None or source_line is None:
        return target_line
    if (target_line[0][0] - target_line[0][2]) != 0:
        angle = abs(target_line[0][1] - target_line[0][3]) / abs(target_line[0][0] - target_line[0][2])
    else:
        angle = 100000000.0
    if angle > 1:
        auxiliary_point_1 = (auxiliary_line_1[0][1] + auxiliary_line_1[0][3]) / 2
        auxiliary_point_2 = (auxiliary_line_2[0][1] + auxiliary_line_2[0][3]) / 2
        if target_line[0][1] > target_line[0][3]:
            target_line[0][1] = auxiliary_point_1 if auxiliary_point_1 > auxiliary_point_2 else auxiliary_point_2
            target_line[0][3] = auxiliary_point_2 if auxiliary_point_1 > auxiliary_point_2 else auxiliary_point_1
            if target_line[0][0] > target_line[0][2]:
                target_line[0][0] = target_line[0][0] + (target_line[0][1] - source_line[0][1]) / angle
                target_line[0][2] = target_line[0][2] - (source_line[0][3] - target_line[0][3]) / angle
                dist_x = target_line[0][0] - target_line[0][2]
                target_line[0][0] = target_line[0][0] - dist_x * 0.1
                target_line[0][2] = target_line[0][2] + dist_x * 0.1
            else:
                target_line[0][0] = target_line[0][0] - (target_line[0][1] - source_line[0][1]) / angle
                target_line[0][2] = target_line[0][2] + (source_line[0][3] - target_line[0][3]) / angle
                dist_x = target_line[0][2] - target_line[0][0]
                target_line[0][0] = target_line[0][0] + dist_x * 0.1
                target_line[0][2] = target_line[0][2] - dist_x * 0.1
            dist_y = target_line[0][1] - target_line[0][3]
            target_line[0][1] = target_line[0][1] - dist_y * 0.1
            target_line[0][3] = target_line[0][3] + dist_y * 0.1
        else:
            target_line[0][1] = auxiliary_point_2 if auxiliary_point_1 > auxiliary_point_2 else auxiliary_point_1
            target_line[0][3] = auxiliary_point_1 if auxiliary_point_1 > auxiliary_point_2 else auxiliary_point_2
            if target_line[0][0] > target_line[0][2]:
                target_line[0][0] = target_line[0][0] + (source_line[0][1] - target_line[0][1]) / angle
                target_line[0][2] = target_line[0][2] - (target_line[0][3] - source_line[0][3]) / angle
                dist_x = target_line[0][0] - target_line[0][2]
                target_line[0][0] = target_line[0][0] - dist_x * 0.1
                target_line[0][2] = target_line[0][2] + dist_x * 0.1
            else:
                target_line[0][0] = target_line[0][0] - (source_line[0][1] - target_line[0][1]) / angle
                target_line[0][2] = target_line[0][2] + (target_line[0][3] - source_line[0][3]) / angle
                dist_x = target_line[0][2] - target_line[0][0]
                target_line[0][0] = target_line[0][0] + dist_x * 0.1
                target_line[0][2] = target_line[0][2] - dist_x * 0.1
            dist_y = target_line[0][3] - target_line[0][1]
            target_line[0][3] = target_line[0][3] - dist_y * 0.1
            target_line[0][1] = target_line[0][1] + dist_y * 0.1
    else:
        auxiliary_point_1 = (auxiliary_line_1[0][0] + auxiliary_line_1[0][2]) / 2
        auxiliary_point_2 = (auxiliary_line_2[0][0] + auxiliary_line_2[0][2]) / 2
        if target_line[0][0] > target_line[0][2]:
            target_line[0][0] = auxiliary_point_1 if auxiliary_point_1 > auxiliary_point_2 else auxiliary_point_2
            target_line[0][2] = auxiliary_point_2 if auxiliary_point_1 > auxiliary_point_2 else auxiliary_point_1
            if target_line[0][1] > target_line[0][3]:
                target_line[0][1] = target_line[0][1] + (target_line[0][0] - source_line[0][0]) * angle
                target_line[0][3] = target_line[0][3] - (source_line[0][2] - target_line[0][2]) * angle
                dist_y = target_line[0][1] - target_line[0][3]
                target_line[0][1] = target_line[0][1] - dist_y * 0.1
                target_line[0][3] = target_line[0][3] + dist_y * 0.1
            else:
                target_line[0][1] = target_line[0][1] - (target_line[0][0] - source_line[0][0]) * angle
                target_line[0][3] = target_line[0][3] + (source_line[0][2] - target_line[0][2]) * angle
                dist_y = target_line[0][3] - target_line[0][1]
                target_line[0][1] = target_line[0][1] + dist_y * 0.1
                target_line[0][3] = target_line[0][3] - dist_y * 0.1
            dist_x = target_line[0][0] - target_line[0][2]
            target_line[0][0] = target_line[0][0] - dist_x * 0.1
            target_line[0][2] = target_line[0][2] + dist_x * 0.1
        else:
            target_line[0][0] = auxiliary_point_2 if auxiliary_point_1 > auxiliary_point_2 else auxiliary_point_1
            target_line[0][2] = auxiliary_point_1 if auxiliary_point_1 > auxiliary_point_2 else auxiliary_point_2
            if target_line[0][1] > target_line[0][3]:
                target_line[0][1] = target_line[0][1] + (source_line[0][0] - target_line[0][0]) * angle
                target_line[0][3] = target_line[0][3] - (target_line[0][2] - source_line[0][2]) * angle
                dist_y = target_line[0][1] - target_line[0][3]
                target_line[0][1] = target_line[0][1] - dist_y * 0.1
                target_line[0][3] = target_line[0][3] + dist_y * 0.1
            else:
                target_line[0][1] = target_line[0][1] - (source_line[0][0] - target_line[0][0]) * angle
                target_line[0][3] = target_line[0][3] + (target_line[0][2] - source_line[0][2]) * angle
                dist_y = target_line[0][3] - target_line[0][1]
                target_line[0][1] = target_line[0][1] + dist_y * 0.1
                target_line[0][3] = target_line[0][3] - dist_y * 0.1
            dist_x = target_line[0][2] - target_line[0][0]
            target_line[0][2] = target_line[0][2] - dist_x * 0.1
            target_line[0][0] = target_line[0][0] + dist_x * 0.1
    return target_line

def target_line_post_processs(source_line, auxiliary_line_1, auxiliary_line_2):
    target_line = source_line
    if auxiliary_line_1 is None or auxiliary_line_2 is None:
        return target_line
    if (target_line[0][0] - target_line[0][2]) != 0:
        angle = abs(target_line[0][1] - target_line[0][3]) / abs(target_line[0][0] - target_line[0][2])
    else:
        angle = 100000000.0
    if angle > 1: # 竖直线
        if target_line[0][0] > target_line[0][2]:
            a =1




def draw_line(draw_object, target_line, line_thickness):
    if target_line is not None:
        for line in target_line:
            x1, y1, x2, y2 = line
            draw_object.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=line_thickness)

def draw_contours(image, mask_tab, color):
    # 使用cv2.findContours()函数查找轮廓
    contours, _ = cv2.findContours(np.uint8(mask_tab), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历每个轮廓并绘制边界框
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)  # 绘制绿色边界框



# 定义图片和掩码文件夹的路径
image_folder = '../data/test_image'
result_folder = '../test_result_1'

# 获取文件夹中的所有图片文件名
image_files = os.listdir(image_folder)

# create a classifier
segmentor_1 = Segmentor(model_path='../work_dirs/segformer_tensorrt_fp16_static-512x512_area', device_name='cuda', device_id=0)
segmentor_2 = Segmentor(model_path='../work_dirs/segformer_tensorrt_fp16_static-512x512_lidianchi', device_name='cuda', device_id=0)
i = 0
# 遍历所有图片和掩码文件
for image_filename in image_files:
    # 记录代码开始时间
    start_time = time.time()

    i += 1
    # 读取图片
    image = Image.open(os.path.join(image_folder, image_filename))

    # 将图片转换为numpy数组
    image_np = np.array(image)

    # perform inference
    annotations_array = segmentor_1(image_np)
    annotations_array = annotations_array.astype(np.uint8)

    # 膨胀掩码区域
    kernel = np.ones((20, 20), np.uint8)
    dilate_tab = cv2.dilate(annotations_array, kernel, iterations=1)

    # 根据掩码文件中的像素来创建掩码区域的mask
    mask_tab_circle = annotations_array == 1
    mask_tab_tab_up = dilate_tab == 2
    mask_tab_tab_down = dilate_tab == 3
    mask_tab_sealant = dilate_tab == 4
    mask_tab_ear = dilate_tab == 6

    # 创建LSD检测器
    lsd = cv2.createLineSegmentDetector(sigma_scale=0.6, quant=2.0, ang_th=12.5, log_eps=0, density_th=0.99)

    # 灰度化处理
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # 直线检测
    lines, _, _, _ = lsd.detect(gray)

    # 取出各区域的直线
    # 保存标签为sealant的掩码区域中的直线线段
    sealant_lines = filter_area_lines(mask_tab_sealant, lines)

    # 保存标签为ear的掩码区域中的直线线段
    ear_lines = filter_area_lines(mask_tab_ear, lines)

    # 保存标签为tab_down的掩码区域中的直线线段
    tab_down_lines = filter_area_lines(mask_tab_tab_down, lines)

    # 筛选各区域竖直方向的直线
    # sealant区域的竖直线
    vertical_sealant_lines = filter_slope_lines(sealant_lines, 5)

    # tab_down区域的竖直线
    vertical_tab_down_lines = filter_slope_lines(tab_down_lines, 5)

    # ear区域的竖直线
    vertical_ear_lines = filter_slope_lines(ear_lines, 5)

    # 筛选各区域水平方向的直线
    # sealant区域的水平线
    horizontal_sealant_lines = filter_slope_lines(sealant_lines, 0.2)

    # ear区域的水平线
    horizontal_ear_lines = filter_slope_lines(ear_lines, 0.2)

    # tab_down区域的水平线
    horizontal_tab_down_lines = filter_slope_lines(tab_down_lines, 0.2)

    # 目标直线获取
    # 筛选出sealant区域最左侧且较长的竖直线
    target_sealant_left_line, angle_sealant_left = filter_target_line(vertical_sealant_lines, [1, -1])

    # 筛选出sealant区域最右侧且较长的竖直线
    target_sealant_right_line, angle_sealant_right = filter_target_line(vertical_sealant_lines, [1, 1])

    # 筛选出sealant区域最上侧且较长的横线
    target_sealant_up_line, _ = filter_target_line(horizontal_sealant_lines, [0, -1])

    # 筛选出sealant区域最下侧且较长的横线
    target_sealant_down_line, _ = filter_target_line(horizontal_sealant_lines, [0, 5])

    # 筛选出tab_down区域最右侧且较长的竖直线
    target_tab_down_left_line, angle_tab_down_left = filter_target_line(vertical_tab_down_lines, [1, -1])

    # 筛选出tab_down区域最左侧且较长的竖直线
    target_tab_down_right_line, angle_tab_down_right = filter_target_line(vertical_tab_down_lines, [1, 1])

    # 筛选出tab_down区域底层且较长的竖直线
    target_tab_down_down_line, _ = filter_target_line(horizontal_tab_down_lines, [0, 5])

    # 筛选出ear区域底层水平直线
    target_ear_down_line, _ = filter_target_line(horizontal_ear_lines, [0, 1])

    # 筛选出ear区域顶层水平直线
    target_ear_up_line, _ = filter_target_line(horizontal_ear_lines, [0, -5])

    # 筛选出ear区域左侧竖直线
    target_ear_left_line, _ = filter_target_line(vertical_ear_lines, [1, -5])

    # 筛选出ear区域右侧竖直线
    target_ear_right_line, _ = filter_target_line(vertical_ear_lines, [1, 5])

    # 目标直线后处理
    # ear区域左侧竖直线
    target_ear_left_line = target_line_post_process(target_ear_left_line, target_ear_up_line, target_ear_down_line)

    # ear区域右侧竖直线
    target_ear_right_line = target_line_post_process(target_ear_right_line, target_ear_up_line, target_ear_down_line)

    # ear区域底层直线
    target_ear_down_line = target_line_post_process(target_ear_down_line, target_ear_left_line, target_ear_right_line)

    # sealant区域顶层直线
    target_sealant_up_line = target_line_post_process(target_sealant_up_line, target_sealant_left_line, target_sealant_right_line)

    # sealant区域底层直线
    target_sealant_down_line = target_line_post_process(target_sealant_down_line, target_sealant_left_line, target_sealant_right_line)

    # sealant区域左侧直线
    target_sealant_left_line = target_line_post_process(target_sealant_left_line, target_sealant_up_line, target_sealant_down_line)

    # sealant区域右侧直线
    target_sealant_right_line = target_line_post_process(target_sealant_right_line, target_sealant_up_line, target_sealant_down_line)

    # tab_down区域左侧直线
    target_tab_down_left_line = target_line_post_process(target_tab_down_left_line, target_tab_down_down_line, target_sealant_down_line)

    # tab_down区域右侧直线
    target_tab_down_right_line = target_line_post_process(target_tab_down_right_line, target_tab_down_down_line, target_sealant_down_line)

    # 检测
    annotations_array = segmentor_2(image_np)
    annotations_array = annotations_array.astype(np.uint8)

    # 缺陷掩码归类
    mask_tab_tab_question = annotations_array == 1
    mask_tab_foreign_object = annotations_array == 2
    mask_tab_metal_chip = annotations_array == 3
    mask_tab_ear_foldup = annotations_array == 4
    mask_tab_ear_damaged = annotations_array == 5
    mask_tab_weld_crack = annotations_array == 6

    # 使用连通组件分析计算独立连通区域
    num_labels_circle, labels_circle = cv2.connectedComponents(np.uint8(mask_tab_circle))
    num_labels_tab_question, labels_tab_question = cv2.connectedComponents(np.uint8(mask_tab_tab_question))
    num_labels_foreign_object, labels_foreign_object = cv2.connectedComponents(np.uint8(mask_tab_foreign_object))
    num_labels_metal_chip, labels_metal_chip = cv2.connectedComponents(np.uint8(mask_tab_metal_chip))
    num_labels_ear_foldup, labels_ear_foldup = cv2.connectedComponents(np.uint8(mask_tab_ear_foldup))
    num_labels_ear_damaged, labels_ear_damaged = cv2.connectedComponents(np.uint8(mask_tab_ear_damaged))
    num_labels_weld_crack, labels_weld_crack = cv2.connectedComponents(np.uint8(mask_tab_weld_crack))

    # 画缺陷掩码图
    image_np[mask_tab_circle] = [255, 0, 0]
    image_np[mask_tab_tab_question] = [244, 108, 59]
    image_np[mask_tab_foreign_object] = [0, 255, 0]
    image_np[mask_tab_metal_chip] = [0, 85, 255]
    image_np[mask_tab_ear_foldup] = [255, 255, 0]
    image_np[mask_tab_ear_damaged] = [255, 85, 255]
    image_np[mask_tab_weld_crack] = [151, 255, 248]

    # 画缺陷的框
    draw_contours(image_np, mask_tab_circle, (255, 0, 0))
    draw_contours(image_np, mask_tab_tab_question, (244, 108, 59))
    draw_contours(image_np, mask_tab_foreign_object, (0, 255, 0))
    draw_contours(image_np, mask_tab_metal_chip, (0, 85, 255))
    draw_contours(image_np, mask_tab_ear_foldup, (255, 255, 0))
    draw_contours(image_np, mask_tab_ear_damaged, (255, 85, 255))
    draw_contours(image_np, mask_tab_weld_crack, (151, 255, 248))

    if target_sealant_up_line is None or target_ear_down_line is None:
        dist = None
        width = None
    else:
        dist = (target_sealant_up_line[0][1] + target_sealant_up_line[0][3]) / 2 - (
                    target_ear_down_line[0][1] + target_ear_down_line[0][3]) / 2
        width = abs(target_sealant_up_line[0][0] - target_sealant_up_line[0][2]) / 0.8
    # 记录代码结束时间
    end_time = time.time()
    # 计算代码执行时间差
    execution_time = end_time - start_time
    # 添加信息
    if (num_labels_tab_question + num_labels_foreign_object + num_labels_metal_chip + num_labels_ear_foldup + num_labels_ear_damaged + num_labels_weld_crack - 6) == 0:
        cv2.putText(image_np, "OK", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)
    else:
        cv2.putText(image_np, "NG", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 6)
    cv2.putText(image_np, f"Angle_of_left_tab: {angle_tab_down_left}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(image_np, f"Angle_of_right_tab: {angle_tab_down_right}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(image_np, f"Angle_of_left_sealant: {angle_sealant_left}", (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(image_np, f"Angle_of_right_sealant: {angle_sealant_right}", (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(image_np, f"Dist_for_ear_to_sealant: {dist}", (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(image_np, f"Width_sealant: {width}", (10, 550), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(image_np, f"Inference_time: {execution_time}", (10, 620), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(image_np, f"Num_circle: {num_labels_circle - 1}", (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(image_np, f"Image: {i}", (10, 840), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 2)
    cv2.putText(image_np, f"Location: wait for...", (10, 910), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 2)
    cv2.putText(image_np, f"Tab_question: {num_labels_tab_question - 1}", (10, 1090), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(image_np, f"Foreign_objecct: {num_labels_foreign_object-1}", (10, 1160), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(image_np, f"Metal_chip: {num_labels_metal_chip - 1}", (10, 1230), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(image_np, f"Ear_foldup: {num_labels_ear_foldup - 1}", (10, 1300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(image_np, f"Ear_damaged: {num_labels_ear_damaged - 1}", (10, 1370), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(image_np, f"Weld_crack: {num_labels_weld_crack - 1}", (10, 1440), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    # 绘制直线
    image_end = image_np.copy()
    # Convert numpy image to PIL Image
    image_end_pil = Image.fromarray(image_end)
    # Create draw objects for drawing thicker lines
    draw_end = ImageDraw.Draw(image_end_pil)
    # 画线
    draw_line(draw_end, target_sealant_up_line, 5)
    draw_line(draw_end, target_sealant_down_line, 5)
    draw_line(draw_end, target_sealant_left_line, 5)
    draw_line(draw_end, target_sealant_right_line, 5)
    draw_line(draw_end, target_ear_down_line, 5)
    draw_line(draw_end, target_ear_left_line, 5)
    draw_line(draw_end, target_ear_right_line, 5)
    draw_line(draw_end, target_tab_down_left_line, 5)
    draw_line(draw_end, target_tab_down_right_line, 5)

    # 将处理后的图片保存
    image_end_pil.save(os.path.join(result_folder, f'end_{i}.jpg'))

    print(f"第 {i} 张图片中sealant和ear的距离为: {dist}")
    print(f"第 {i} 张图片中sealant左直线的角度为: {angle_sealant_left}")
    print(f"第 {i} 张图片中sealant右直线的角度为: {angle_sealant_right}")
    print(f"第 {i} 张图片中tab_down左直线的角度为: {angle_tab_down_left}")
    print(f"第 {i} 张图片中tab_down右直线的角度为: {angle_tab_down_right}")
