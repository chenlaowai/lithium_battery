from mmdeploy.apis import torch2onnx
from mmdeploy.apis.tensorrt import onnx2tensorrt
from mmdeploy.backend.sdk.export_info import export2SDK
import os

img = '../data/test_image/ear_foldup1.jpg'
work_dir = '../work_dirs/segformer_tensorrt_fp16_static-512x512_area'
save_file = 'end2end.onnx'
deploy_cfg = '../configs/mmdeploy_config/segmentation_tensorrt-fp16_static-512x512.py'
model_cfg = '../work_dirs/segformer_mit-b1_40k_512x512_voc_area_new_1/segformer_mit-b1_40k_512x512_voc_area_new_1.py'
model_checkpoint = '../work_dirs/segformer_mit-b1_40k_512x512_voc_area_new_1/iter_40000.pth'
device = 'cuda'

# 1. convert model to IR(onnx)
torch2onnx(img, work_dir, save_file, deploy_cfg, model_cfg,
  model_checkpoint, device)

# 2. convert IR to tensorrt
onnx_model = os.path.join(work_dir, save_file)
save_file = 'end2end.engine'
model_id = 0
device = 'cuda'
onnx2tensorrt(work_dir, save_file, model_id, deploy_cfg, onnx_model, device)

# 3. extract pipeline info for sdk use (dump-info)
export2SDK(deploy_cfg, model_cfg, work_dir, pth=model_checkpoint, device=device)