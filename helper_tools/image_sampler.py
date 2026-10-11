import numpy as np
import pyexr
import os
import sys
from PIL import Image
from utils import tone_mapping

scene = 'sponza-moving-light'
scene_accum_dir = os.path.join('dataset', scene, 'acc_colors')
scene_inputs_dir = os.path.join('dataset', scene, 'inputs')
our_experiment_name = 'Fixed+Rep+AFF+Aux'
adfn_experiment_name = 'AD'

ours_results_dir = os.path.join('results', our_experiment_name, scene)
adfn_results_dir = os.path.join('results', adfn_experiment_name, scene)
not_our_results_dir = os.path.join('wskpn_results', scene)

crop_size = 128
frame_num = 50 #0-59
top_left_x = 858
top_left_y = 463
top_left_y = 720 - top_left_y
coord = f'({top_left_x},{top_left_y})'
save_dir = os.path.join('saved_crops', our_experiment_name, scene, f'{frame_num}', coord)

for dir in [save_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

for dir in [scene_accum_dir, scene_inputs_dir, ours_results_dir, adfn_results_dir, not_our_results_dir]:
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

low_x = top_left_x
low_y = top_left_y
up_x = top_left_x + crop_size
up_y = top_left_y + crop_size

albedo_path = os.path.join(scene_inputs_dir, f'albedo{frame_num}.exr')
non_ta_path = os.path.join(scene_inputs_dir, f'color{frame_num}.exr')
ta_path = os.path.join(scene_accum_dir, f'color{frame_num}.exr')
reference_path = os.path.join(scene_inputs_dir, f'reference{frame_num}.exr')

non_ta = pyexr.read(non_ta_path)[low_y:up_y,low_x:up_x,:3]
albedo = pyexr.read(albedo_path)[low_y:up_y,low_x:up_x,:3]
ta = pyexr.read(ta_path)[low_y:up_y,low_x:up_x,:3]
reference = pyexr.read(reference_path)[low_y:up_y,low_x:up_x,:3]

ta *= albedo
non_ta *= albedo

non_ta = tone_mapping(non_ta) * 255
albedo = tone_mapping(albedo) * 255
ta = tone_mapping(ta) * 255
reference = tone_mapping(reference) * 255

albedo = albedo.astype(np.uint8)
non_ta = non_ta.astype(np.uint8)
ta = ta.astype(np.uint8)
reference = reference.astype(np.uint8)

albedo_png = Image.fromarray(albedo)
non_ta_png = Image.fromarray(non_ta)
ta_png = Image.fromarray(ta)
reference_png = Image.fromarray(reference)

non_ta_png.save(os.path.join(save_dir, f'non_ta.png'))
ta_png.save(os.path.join(save_dir, f'ta.png'))
reference_png.save(os.path.join(save_dir, f'reference.png'))

for result_dir in [ours_results_dir, adfn_results_dir]:
    result_path = os.path.join(result_dir, f'{frame_num}.exr')
    
    result = pyexr.read(result_path)[low_y:up_y,low_x:up_x,:3]

    result = tone_mapping(result) * 255

    result = result.astype(np.uint8)

    result_png = Image.fromarray(result)

    # save as png
    # result_png.save(os.path.join(save_dir, f'{result_dir.partition('_')[0]}.png'))
    # result_png.save(os.path.join(save_dir, f'{}.png'))
    if result_dir == ours_results_dir:
        result_png.save(os.path.join(save_dir, f'ours.png'))
    elif result_dir == adfn_results_dir:
        result_png.save(os.path.join(save_dir, f'adfn.png'))

# other papers' results
wskpn = Image.open(os.path.join(not_our_results_dir, 'Ours.png'))
wskpn = np.asarray(wskpn)
bmfr = Image.open(os.path.join(not_our_results_dir, 'BMFR.png'))
bmfr = np.asarray(bmfr)
nbgd = Image.open(os.path.join(not_our_results_dir, 'NBGD.png'))
nbgd = np.asarray(nbgd)
svgf = Image.open(os.path.join(not_our_results_dir, 'SVGF.png'))
svgf = np.asarray(svgf)

wskpn_png = Image.fromarray(wskpn[low_y:up_y,low_x:up_x,:3])
bmfr_png = Image.fromarray(bmfr[low_y:up_y,low_x:up_x,:3])
nbgd_png = Image.fromarray(nbgd[low_y:up_y,low_x:up_x,:3])
svgf_png = Image.fromarray(svgf[low_y:up_y,low_x:up_x,:3])

wskpn_png.save(os.path.join(save_dir, f'wskpn.png'))
bmfr_png.save(os.path.join(save_dir, f'bmfr.png'))
nbgd_png.save(os.path.join(save_dir, f'nbgd.png'))
svgf_png.save(os.path.join(save_dir, f'svgf.png'))

print('Success!')