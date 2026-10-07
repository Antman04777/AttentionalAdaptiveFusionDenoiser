import numpy as np
import pyexr
import os
import sys

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

result = pyexr.read(result_path)[low_y:up_y,low_x:up_x,:3]
albedo = pyexr.read(albedo_path)[low_y:up_y,low_x:up_x,:3]
non_ta = pyexr.read(non_ta_path)[low_y:up_y,low_x:up_x,:3]
ta = pyexr.read(ta_path)[low_y:up_y,low_x:up_x,:3]
depth = pyexr.read(depth_path)[low_y:up_y,low_x:up_x,:1]
reference = pyexr.read(reference_path)[low_y:up_y,low_x:up_x,:3]
shading_normal = pyexr.read(shading_normal_path)[low_y:up_y,low_x:up_x,:3]
world_position = pyexr.read(world_position_path)[low_y:up_y,low_x:up_x,:3]

pyexr.write(os.path.join(save_dir, f'result.exr'), result)
pyexr.write(os.path.join(save_dir, f'albedo.exr'), albedo)
pyexr.write(os.path.join(save_dir, f'non_ta.exr'), non_ta)
pyexr.write(os.path.join(save_dir, f'ta.exr'), ta)
pyexr.write(os.path.join(save_dir, f'depth.exr'), depth)
pyexr.write(os.path.join(save_dir, f'reference.exr'), reference)
pyexr.write(os.path.join(save_dir, f'shading_normal.exr'), shading_normal)
pyexr.write(os.path.join(save_dir, f'world_position.exr'), world_position)

print('Success!')