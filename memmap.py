import numpy as np
from dataloader import RawDataReader

import psutil
    
if __name__ == '__main__':
    process = psutil.Process()
    print(process.memory_info().rss)
    data = RawDataReader()

    print(process.memory_info().rss)

    part = data.test_inputs
    print(part)

    # train_shape = (data.NUM_TRAIN_IMAGES, data.IMG_WIDTH, data.IMG_HEIGHT, data.NUM_INPUT_CHANNELS)
    # test_shape = (data.NUM_TEST_IMAGES, data.IMG_WIDTH, data.IMG_HEIGHT, data.NUM_REFERENCE_CHANNELS)
    
    # mmap_train_i = np.memmap('memmap_dataset/train_inputs.npy', dtype=np.float32, mode='r', shape=train_shape)
    # mmap_train_r = np.memmap('memmap_dataset/train_references.npy', dtype=np.float32, mode='r', shape=train_shape)
    # mmap_test_i = np.memmap('memmap_dataset/test_inputs.npy', dtype=np.float32, mode='r', shape=test_shape)
    # mmap_test_r = np.memmap('memmap_dataset/test_references.npy', dtype=np.float32, mode='r', shape=test_shape)

    # train_i = mmap_train_i.__array__()
    # train_r = mmap_train_r.__array__()
    # test_i =  mmap_test_i.__array__()
    # test_r =  mmap_test_r.__array__()   

    # print()
    # print(train_i.shape)
    # print(train_r.shape)
    # print(test_i.shape)
    # print(test_r.shape)

    print(process.memory_info().rss)