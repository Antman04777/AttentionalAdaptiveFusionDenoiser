import dataloader
import pyexr
import os

our_data = dataloader.RawDataReader()

#train_inputs = our_data.train_inputs 
#train_ref = our_data.train_references

test_inputs = our_data.test_inputs
test_ref = our_data.test_references

for i in range(len(test_inputs)):

    current_frame_test_inputs = test_inputs[i]
    current_frame_test_ref = test_ref[i]

    # irradiance_non_ta = current_frame_test_inputs[:,:,0:3]
    # albedo =  current_frame_test_inputs[:,:,3:6]
    # shading_normal = current_frame_test_inputs[:,:,6:9]
    # depth = current_frame_test_inputs[:,:,9:10]

    # pyexr.write(os.path.join("testing_data", "color"+str(i)+".exr"), irradiance_non_ta)
    # pyexr.write(os.path.join("testing_data", "albedo"+str(i)+".exr"), albedo)
    # pyexr.write(os.path.join("testing_data", "shading_normal"+str(i)+".exr"), shading_normal)
    # pyexr.write(os.path.join("testing_data", "depth"+str(i)+".exr"), depth)
    # pyexr.write(os.path.join("testing_data", "ref"+str(i)+".exr"), current_frame_test_ref)