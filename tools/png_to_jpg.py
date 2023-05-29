import os
import json
from PIL import Image

# 转换png为jpg
img_folder = '../data/test_image'
for img_file in os.listdir(img_folder):
    if img_file.endswith('.png'):
        img_path = os.path.join(img_folder, img_file)
        img = Image.open(img_path)
        img = img.convert('RGB')
        img_file_jpg = img_file[:-3] + 'jpg'
        img_path_jpg = os.path.join(img_folder, img_file_jpg)
        img.save(img_path_jpg)
        os.remove(img_path)


