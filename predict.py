import os
import shutil
import torch

from model import MattingNetwork
from inference import convert_video

model_path = 'checkpoints/rvm_resnet50.pth'
input_source = 'data/val2022/Hisense/img'
output_source = 'data/output'

model = MattingNetwork('resnet50').eval().cuda()
model.load_state_dict(torch.load(model_path))

if output_source != None:
    if os.path.exists(output_source):
        shutil.rmtree(output_source)
    os.makedirs(output_source)

convert_video(
    model,
    input_source = input_source,
    output_type = 'png_sequence',
    output_composition = output_source,
    downsample_ratio = None,
    progress = True
)