import numpy as np
from dataloader import RawDataReader

if __name__ == '__main__':
    data = RawDataReader()

    train_inputs = np.array(data.train_inputs)
    train_references = np.array(data.train_references)
    test_inputs = np.array(data.test_inputs)
    test_references = np.array(data.test_references)

    print(train_inputs.shape)
    print(train_references.shape)
    print(test_inputs.shape)
    print(test_references.shape)

    train_inputs.tofile('memmap_dataset/train_inputs.npy')
    train_references.tofile('memmap_dataset/train_references.npy')
    test_inputs.tofile('memmap_dataset/test_inputs.npy')
    test_references.tofile('memmap_dataset/test_references.npy')

    mmap_train_i = np.memmap('memmap_dataset/train_inputs.npy', dtype=np.float32, mode='r')
    mmap_train_r = np.memmap('memmap_dataset/train_references.npy', dtype=np.float32, mode='r')
    mmap_test_i = np.memmap('memmap_dataset/test_inputs.npy', dtype=np.float32, mode='r')
    mmap_test_r = np.memmap('memmap_dataset/test_references.npy', dtype=np.float32, mode='r')

    train_i = mmap_train_i.__array__()
    train_r = mmap_train_r.__array__()
    test_i =  mmap_test_i.__array__()
    test_r =  mmap_test_r.__array__()   

    print()
    print(train_i.shape)
    print(train_r.shape)
    print(test_i.shape)
    print(test_r.shape)

    train_i = train_i.reshape(train_inputs.shape)
    train_r = train_r.reshape(train_references.shape)
    test_i = test_i.reshape(test_inputs.shape)
    test_r = test_r.reshape(test_references.shape)

    print()
    print(train_i.shape)
    print(train_r.shape)
    print(test_i.shape)
    print(test_r.shape)