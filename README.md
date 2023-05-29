1.训练模型
mmseg具体用法跟官方一致
2.部署模型
在tools/pytorch_to_tensorrt.py文件中写入模型路径和工作路径（即保存路径）即可生成部署模型
3.生成缺陷检测图片
tools/the_end.py写入源图片、生成图片路径，部署模型路径即可。
