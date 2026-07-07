import dataloader
import pyexr
import os

our_data = dataloader.RawDataReader

#train_inputs = our_data.train_inputs 
#train_ref = our_data.train_references

test_inputs = our_data.test_inputs
test_ref = our_data.test_references

for i in range(len(test_inputs)):

    current_frame_test_inputs = test_inputs[i]
    current_frame_test_ref = test_ref[i]

    current_feature_test_inp =  current_frame_test_inputs[:,:,6:9] #albedo
    

    pyexr.write(os.path.join("testing_data", "albedo"+str(i)+".exr"), current_feature_test_inp)
    pyexr.write(os.path.join("testing_data", "ref"+str(i)+".exr"), current_frame_test_ref)