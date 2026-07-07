import numpy as np
import pyexr
import os
from tqdm import tqdm
# import sys
import re

scene_names = ["classroom", "living-room", "san-miguel", "sponza-glossy", "sponza-moving-light", "sponza"]

for scene in scene_names:
    camera_matrices = np.zeros((60, 4, 4))
    with open(os.path.join("dataset", "cameras", scene+".h"), "r") as file:
        camera_idx = 0
        row_idx = 0
        for line in file.readlines():
            floats = re.findall(r'-?\d+.\d+', line)
            if len(floats) > 0:
                camera_matrices[camera_idx, row_idx] = np.array(floats, dtype=np.float32)
                row_idx += 1
            if row_idx == 4:
                camera_idx += 1
                row_idx = 0

    for k in tqdm(range(60)):
        world_space_position_img = pyexr.read(os.path.join("dataset", scene, "inputs", "world_position"+str(k)+".exr"))
        
        final_depth = np.zeros(shape=(world_space_position_img.shape[0], world_space_position_img.shape[1], 1))

        world_position = np.array([0., 0., 0., 1.])

        # the camera matrix is comprised of the multiplication of the projection matrix and the view matrix defined by the BMFR people
        camera_matrix = camera_matrices[k].T    # Transpose is required for correct clip space transformation

        for i in range(world_space_position_img.shape[0]):
            for j in range(world_space_position_img.shape[1]):
                world_position[:3] = world_space_position_img[i, j]
                clip_space_position = np.dot(camera_matrix,world_position)
                ndc_depth = (clip_space_position[2] / clip_space_position[3])
                final_depth[i, j] = ndc_depth

        inv_final_depth = 1.0 / final_depth
        inv_min_depth = 1.0 / np.min(final_depth)
        inv_max_depth = 1.0 / np.max(final_depth)
        final_depth = (inv_final_depth - inv_min_depth) / (inv_max_depth - inv_min_depth)

        pyexr.write(os.path.join("dataset", scene, "inputs", "depth"+str(k)+".exr"), final_depth)