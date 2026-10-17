import numpy as np
import os
import torch
import pyexr
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf

# from memory_profiler import profile

class RawDataReader:

    def __init__(self, is_training=True, hold_out_scene=None):
        self.train_inputs = None
        self.train_references = None
        self.test_inputs = None
        self.test_references = None
        self.IMG_HEIGHT = 720
        self.IMG_WIDTH = 1280

        self.NUM_INPUT_CHANNELS = 16
        self.NUM_REFERENCE_CHANNELS = 3

        self.scene_names = ["classroom", "living-room", "san-miguel", "sponza-glossy", "sponza-moving-light", "sponza"]
        self.scenes_to_be_loaded = None
        self.num_frames_per_scene = 60

        self.is_training = is_training
        self.hold_out_scene = hold_out_scene
        
        if self.is_training:
            self.scenes_to_be_loaded = [scene for scene in self.scene_names if scene != self.hold_out_scene]
        else:
            self.scenes_to_be_loaded = [self.hold_out_scene]

        self.NUM_TRAIN_IMAGES = self.num_frames_per_scene * (len(self.scenes_to_be_loaded) - 1)
        self.NUM_VALIDATION_IMAGES = 5 * (len(self.scenes_to_be_loaded) - 1)
        self.NUM_TEST_IMAGES = self.num_frames_per_scene

        self.train_input_shape = (self.NUM_TRAIN_IMAGES, self.IMG_HEIGHT, self.IMG_WIDTH, self.NUM_INPUT_CHANNELS)
        self.train_reference_shape = (self.NUM_TRAIN_IMAGES, self.IMG_HEIGHT, self.IMG_WIDTH, self.NUM_REFERENCE_CHANNELS)
        self.test_input_shape = (self.NUM_TEST_IMAGES, self.IMG_HEIGHT, self.IMG_WIDTH, self.NUM_INPUT_CHANNELS)
        self.test_reference_shape = (self.NUM_TEST_IMAGES, self.IMG_HEIGHT, self.IMG_WIDTH, self.NUM_REFERENCE_CHANNELS)

        self.load_scenes()

    def load_scenes(self):
        folder_name = os.path.join("dataset")

        irradiance_TA_file_names = [os.path.join(folder_name, scene, "acc_colors", "color"+str(i)+".exr") for scene in self.scenes_to_be_loaded for i in range (self.num_frames_per_scene)]
        irradiance_non_TA_file_names = [os.path.join(folder_name, scene, "inputs", "color"+str(i)+".exr") for scene in self.scenes_to_be_loaded for i in range (self.num_frames_per_scene)]
        albedo_file_names = [os.path.join(folder_name, scene, "inputs", "albedo"+str(i)+".exr") for scene in self.scenes_to_be_loaded for i in range (self.num_frames_per_scene)]
        normal_file_names = [os.path.join(folder_name, scene, "inputs", "shading_normal"+str(i)+".exr") for scene in self.scenes_to_be_loaded for i in range (self.num_frames_per_scene)]
        world_position_file_names = [os.path.join(folder_name, scene, "inputs", "world_position"+str(i)+".exr") for scene in self.scenes_to_be_loaded for i in range (self.num_frames_per_scene)]
        depth_file_names = [os.path.join(folder_name, scene, "inputs", "depth"+str(i)+".exr") for scene in self.scenes_to_be_loaded for i in range (self.num_frames_per_scene)]
        reference_file_names = [os.path.join(folder_name, scene, "inputs", "reference"+str(i)+".exr") for scene in self.scenes_to_be_loaded for i in range (self.num_frames_per_scene)]

        self.train_inputs = []
        self.train_references = []
        self.validation_inputs = []
        self.validation_references = []
        self.test_inputs = []
        self.test_references = []

        for i in tqdm(range(len(reference_file_names))):
            
            irradiance_TA_image = pyexr.read(irradiance_TA_file_names[i])[:,:,:3]
            irradiance_non_TA_image = pyexr.read(irradiance_non_TA_file_names[i])[:,:,:3]
            albedo_image = pyexr.read(albedo_file_names[i])
            reference_image=pyexr.read(reference_file_names[i])

            normal_image=pyexr.read(normal_file_names[i])
            normal_image = normal_image*0.5 + 0.5

            world_position_image = pyexr.read(world_position_file_names[i])

            depth_image = pyexr.read(depth_file_names[i])[:,:, 0:1]
            depth_image = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image))

            inputs = np.concatenate([irradiance_non_TA_image,
                                    irradiance_TA_image,
                                    albedo_image,
                                    normal_image,
                                    world_position_image,
                                    depth_image], 
                                    axis=2)
            
            if self.is_training:
                if i % self.num_frames_per_scene >= (self.num_frames_per_scene - 5):
                    self.validation_inputs.append(inputs)
                    self.validation_references.append(reference_image)
                    continue
                self.train_inputs.append(inputs)
                self.train_references.append(reference_image)

            else:
                self.test_inputs.append(inputs)
                self.test_references.append(reference_image)

class BMFRFinalDataset(Dataset):
    def __init__(self, raw_data=None, use_train=False, use_val = False, use_test=False, crops_per_training_frame=80, crops_per_validation_frame=20, crop_size=128):
        self.raw_data=raw_data
        self.use_train=use_train
        self.use_val=use_val
        self.use_test=use_test
        self.crops_per_training_frame = crops_per_training_frame
        self.crops_per_validation_frame = crops_per_validation_frame
        self.crop_size = crop_size

        def rotate90(input):
            return torch.rot90(input, 1, dims=(1, 2))

        def rotate270(input):
            return torch.rot90(input, -1, dims=(1, 2))

        self.augmentation_operations = [rotate90, rotate270, tf.hflip, tf.vflip]

    def transform(self, input, reference):
        if self.use_train or self.use_val:
            rand_x = np.random.randint(self.raw_data.IMG_HEIGHT - self.crop_size)
            rand_y = np.random.randint(self.raw_data.IMG_WIDTH - self.crop_size)

            input_cropped = tf.to_tensor(input[rand_x : rand_x + self.crop_size, rand_y : rand_y + self.crop_size].astype(np.float32))
            reference_cropped = tf.to_tensor(reference[rand_x : rand_x + self.crop_size, rand_y : rand_y + self.crop_size].astype(np.float32))

            if np.random.rand() > 0.5:
                operation = np.random.choice(self.augmentation_operations)
                input_cropped = operation(input_cropped)
                reference_cropped = operation(reference_cropped)
                    
        elif self.use_test:
            input_cropped = tf.to_tensor(input.astype(np.float32))
            reference_cropped = tf.to_tensor(reference.astype(np.float32))

        return input_cropped, reference_cropped
    
    def __getitem__(self, index):
        if self.use_test:
            frame_index = index
            return self.transform(self.raw_data.test_inputs[frame_index], self.raw_data.test_references[frame_index])
        
        elif self.use_train:
            frame_index = index // self.crops_per_training_frame
            train_in = self.raw_data.train_inputs[frame_index]
            train_ref = self.raw_data.train_references[frame_index]
            ret = self.transform(train_in, train_ref)

            return ret
        
        elif self.use_val:
            frame_index = index // self.crops_per_validation_frame
            return self.transform(self.raw_data.validation_inputs[frame_index], self.raw_data.validation_references[frame_index])
    
    '''
    def __getitems__(self, indices):
        final_list = []
        for i in indices:
            if self.use_test:
                frame_index = i
                input, reference = self.crop(self.raw_data.test_inputs[frame_index], self.raw_data.test_references[frame_index])
            elif self.use_train:
                frame_index = i // self.crops_per_training_frame
                # input_size = self.raw_data.IMG_HEIGHT * self.raw_data.IMG_WIDTH * self.raw_data.NUM_INPUT_CHANNELS
                # reference_size = self.raw_data.IMG_HEIGHT * self.raw_data.IMG_WIDTH * self.raw_data.NUM_REFERENCE_CHANNELS
                # offset_input = frame_index * self.raw_data.IMG_HEIGHT * self.raw_data.IMG_WIDTH * self.raw_data.NUM_INPUT_CHANNELS
                # offset_reference = frame_index * self.raw_data.IMG_HEIGHT * self.raw_data.IMG_WIDTH * self.raw_data.NUM_REFERENCE_CHANNELS

                # train_in = np.memmap('memmap_dataset/train_inputs.npy', dtype=np.float32, mode='r', shape=self.raw_data.train_input_shape, offset=offset_input)
                # train_ref = np.memmap('memmap_dataset/train_references.npy', dtype=np.float32, mode='r', shape=self.raw_data.train_reference_shape, offset=offset_reference)

                train_in = self.raw_data.train_inputs[frame_index]
                train_ref = self.raw_data.train_references[frame_index]

                input, reference = self.crop(train_in, train_ref)
                del train_in, train_ref
            elif self.use_val:
                frame_index = i // self.crops_per_validation_frame
                train_in = self.raw_data.train_inputs[frame_index]
                test_in = self.raw_data.train_references[frame_index]
                input, reference = self.crop(train_in, test_in)
                del train_in, test_in

            final_list.append((input, reference))

        return final_list
    '''

    def __len__(self):
        if self.use_train:
            return (self.raw_data.NUM_TRAIN_IMAGES - self.raw_data.NUM_VALIDATION_IMAGES) * self.crops_per_training_frame
        elif self.use_val:
            return self.raw_data.NUM_VALIDATION_IMAGES * self.crops_per_validation_frame
        elif self.use_test:
            return self.raw_data.NUM_TEST_IMAGES
        
    
class DatasetPreloader:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.next_input = None
        self.next_reference = None

        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_reference = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_reference = None
            return
        
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.next_input = self.next_input.to(device=self.device, non_blocking=True)
                self.next_reference = self.next_reference.to(device=self.device, non_blocking=True)

    def __iter__(self):
        return self

    def __next__(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)

        input = self.next_input
        reference = self.next_reference

        if input is None:
            raise StopIteration

        if self.stream is not None:
            input.record_stream(torch.cuda.current_stream())
            reference.record_stream(torch.cuda.current_stream())

        self.preload()

        return input, reference

