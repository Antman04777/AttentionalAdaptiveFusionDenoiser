import numpy as np
import os
import torch
import pyexr
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf


class RawDataReader:

    def __init__(self):

        folder_name = os.path.join("dataset")
        scene_names = ["classroom"]


        num_frames_per_scene = 5

        irradiance_TA_file_names = [os.path.join(folder_name, scene, "acc_colors", "colors"+str(i)+".exr") for scene in scene_names for i in range (num_frames_per_scene)]

        irradiance_NonTA_file_names = [os.path.join(folder_name, scene, "inputs", "colors"+str(i)+".exr") for scene in scene_names for i in range (num_frames_per_scene)]

        albedo_file_names = [os.path.join(folder_name, scene, "inputs", "albedo"+str(i)+".exr") for scene in scene_names for i in range (num_frames_per_scene)]

        normal_file_names = [os.path.join(folder_name, scene, "inputs", "shading_normal"+str(i)+".exr") for scene in scene_names for i in range (num_frames_per_scene)]

        depth_file_names = [os.path.join(folder_name, scene, "inputs", "depth"+str(i)+".exr") for scene in scene_names for i in range (num_frames_per_scene)]

        reference_file_names = [os.path.join(folder_name, scene, "inputs", "reference"+str(i)+".exr") for scene in scene_names for i in range (num_frames_per_scene)]

        self.train_inputs=[] 
        self.test_inputs=[]

        self.train_references=[]
        self.test_references=[]

        for i in tqdm(range(len(reference_file_names))):
            
            irradiance_TA_image = pyexr.read(irradiance_TA_file_names[i])[:,:,3]
            irradiance_NonTA_image = pyexr.read(irradiance_NonTA_file_names[i])[:,:,3]
            albedo_image = pyexr.read(albedo_file_names[i])
            reference_image=pyexr.read(reference_file_names[i])
            normal_image=pyexr.read(normal_file_names[i])
            normal_image = normal_image*0.5 + 0.5
            depth_image = pyexr.read(depth_file_names[i])[:,:, 0:1]
            depth_image = (depth_image - np.min(depth_image))/ (np.max(depth_image) - np.min(depth_image))

            inputs = np.concatenate(irradiance_NonTA_image,
                                    irradiance_TA_image,
                                    albedo_image,
                                    normal_image,
                                    depth_image, 
                                    axis = 2)
            
            if i < (num_frames_per_scene*len(scene_names-1)):
                
                self.train_inputs.append(inputs)
                self.train_references.append(reference_image)

            else:

                self.test_inputs.append(inputs)
                self.test_references.append(reference_image)


class BMFRFinalDataset(Dataset):
    def __init__(self, raw_data, use_train=False, use_val = False, use_test=False):
        self.raw_data=raw_data
        self.use_train=use_train
        self.use_val=use_val
        self.use_test=use_test

    def __getitem__(self, index):
        if self.use_test:
            return tf.to_tensor(self.raw_data.test_inputs[index]), tf.to_tensor(self.raw_data.test_references[index]) 
        else:
            return tf.to_tensor(self.raw_data.train_inputs[index]), tf.to_tensor(self.raw_data.train_references[index]) #wskpn used float 32
        
    def __len__(self):
        if self.use_train:
            return len(self.raw_data.train_references)
        else:
            return len(self.raw_data.test_references)
        
    

