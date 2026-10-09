import numpy as np
import pyexr
import os
import sys
from PIL import Image
from utils import tone_mapping

scene = 'san-miguel'
scene_accum_dir = os.path.join('dataset', scene, 'acc_colors')
scene_inputs_dir = os.path.join('dataset', scene, 'inputs')
experiment_name = 'Fixed+Rep+AFF+Aux'
results_dir = os.path.join('results', experiment_name, scene)
crop_size = 128
frame_num = 28 #0-59
top_left_x = 768
top_left_y = 282
coord = f'({top_left_x},{top_left_y})'
save_dir = os.path.join('saved_crops', experiment_name, scene, f'{frame_num}', coord)

for dir in [save_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

for dir in [scene_accum_dir, scene_inputs_dir, results_dir]:
    if not os.path.exists(dir):
        print(f'Vital directory: {dir} does not exist!')
        sys.exit(-1)

if top_left_x < 0 or top_left_x > (1280 - crop_size):
    print(f'Invalid top left x coord provided!')
    sys.exit(-1)

if top_left_y < 0 or top_left_y > (720 - crop_size):
    print(f'Invalid top left y coord provided!')
    sys.exit(-1)

if frame_num >= 59 or frame_num < 0:
    print(f'Invalid frame number provided!')
    sys.exit(-1)

result_path = os.path.join(results_dir, f'{frame_num}.exr')
albedo_path = os.path.join(scene_inputs_dir, f'albedo{frame_num}.exr')
non_ta_path = os.path.join(scene_inputs_dir, f'color{frame_num}.exr')
ta_path = os.path.join(scene_accum_dir, f'color{frame_num}.exr')
depth_path = os.path.join(scene_inputs_dir, f'depth{frame_num}.exr')
reference_path = os.path.join(scene_inputs_dir, f'reference{frame_num}.exr')
shading_normal_path = os.path.join(scene_inputs_dir, f'shading_normal{frame_num}.exr')
world_position_path = os.path.join(scene_inputs_dir, f'world_position{frame_num}.exr')

low_x = top_left_x
low_y = top_left_y
up_x = top_left_x + crop_size
up_y = top_left_y + crop_size

result = tone_mapping(pyexr.read(result_path)[low_y:up_y,low_x:up_x,:3]) * 255
albedo = tone_mapping(pyexr.read(albedo_path)[low_y:up_y,low_x:up_x,:3]) * 255
non_ta = tone_mapping(pyexr.read(non_ta_path)[low_y:up_y,low_x:up_x,:3]) * 255
ta = tone_mapping(pyexr.read(ta_path)[low_y:up_y,low_x:up_x,:3]) * 255
depth = tone_mapping(pyexr.read(depth_path)[low_y:up_y,low_x:up_x,:1]) * 255
reference = tone_mapping(pyexr.read(reference_path)[low_y:up_y,low_x:up_x,:3]) * 255
shading_normal = tone_mapping(pyexr.read(shading_normal_path)[low_y:up_y,low_x:up_x,:3]) * 255
# shading_normal = shading_normal*0.5 + 0.5 # [-1, 1] -> [0, 1]
world_position = tone_mapping(pyexr.read(world_position_path)[low_y:up_y,low_x:up_x,:3]) * 255

result = result.astype(np.uint8)
albedo = albedo.astype(np.uint8)
non_ta = non_ta.astype(np.uint8)
ta = ta.astype(np.uint8)
depth = depth.astype(np.uint8)
depth = depth.reshape(depth.shape[0], depth.shape[1])
reference = reference.astype(np.uint8)
shading_normal = shading_normal.astype(np.uint8)
shading_normal = shading_normal.astype(np.uint8)
world_position = world_position.astype(np.uint8)

# pyexr.write(os.path.join(save_dir, f'result.exr'), result)
# pyexr.write(os.path.join(save_dir, f'albedo.exr'), albedo)
# pyexr.write(os.path.join(save_dir, f'non_ta.exr'), non_ta)
# pyexr.write(os.path.join(save_dir, f'ta.exr'), ta)
# pyexr.write(os.path.join(save_dir, f'depth.exr'), depth)
# pyexr.write(os.path.join(save_dir, f'reference.exr'), reference)
# pyexr.write(os.path.join(save_dir, f'shading_normal.exr'), shading_normal)
# pyexr.write(os.path.join(save_dir, f'world_position.exr'), world_position)

result_png = Image.fromarray(result)
albedo_png = Image.fromarray(albedo)
non_ta_png = Image.fromarray(non_ta)
ta_png = Image.fromarray(ta)
depth_png = Image.fromarray(depth, mode='L')
reference_png = Image.fromarray(reference)
shading_normal_png = Image.fromarray(shading_normal)
world_position_png = Image.fromarray(world_position)

# save as png
result_png.save(os.path.join(save_dir, f'result.png'))
albedo_png.save(os.path.join(save_dir, f'albedo.png'))
non_ta_png.save(os.path.join(save_dir, f'non_ta.png'))
ta_png.save(os.path.join(save_dir, f'ta.png'))
depth_png.save(os.path.join(save_dir, f'depth.png'))
reference_png.save(os.path.join(save_dir, f'reference.png'))
shading_normal_png.save(os.path.join(save_dir, f'shading_normal.png'))
world_position_png.save(os.path.join(save_dir, f'world_position.png'))

print('Success!')