from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
from PIL import Image, ImageDraw
from mmpose.registry import VISUALIZERS
import numpy as np
from mmcv.image import imread

register_all_modules()

config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
checkpoint_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
model = init_model(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'

# please prepare an image with person
results = inference_topdown(model, 'demo.jpg')

model.cfg.visualizer.radius = 10
model.cfg.visualizer.alpha = 1.0
model.cfg.visualizer.line_width = 3

visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.set_dataset_meta(
    model.dataset_meta, skeleton_style="mmpose")

visualizer.add_datasample(
        'result',
        imread("demo.jpg", channel_order='rgb'),
        data_sample=results,
        draw_gt=False,
        draw_bbox=True,
        kpt_thr=0.3,
        draw_heatmap=False,
        show_kpt_idx=True,
        skeleton_style="mmpose",
        show=False,
        out_file="demo_skeleton.png")