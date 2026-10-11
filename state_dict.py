import torch
import os
import copy

experiment = 'Fixed+Rep+AFF+Aux'
scene = 'san-miguel'
model_state_dict = torch.load(os.path.join('checkpoints', experiment, scene, 'model.pt'))
# new_model_state_dict = model_state_dict.copy()
new_model_state_dict = copy.deepcopy(model_state_dict)

# print(state_dict)
for k,v in model_state_dict['model_state_dict'].items():
    if 'adaptive_fusion_conv.' in k:
        new_key = k.replace('adaptive_fusion_conv.', 'kernel_predictor_cnn.')
        new_model_state_dict['model_state_dict'][new_key] = new_model_state_dict['model_state_dict'].pop(k)

#for k,v in new_model_state_dict['model_state_dict'].items():
#    print(k)

torch.save({'model_state_dict': new_model_state_dict['model_state_dict'], 'optimizer_state_dict': new_model_state_dict['optimizer_state_dict']}, os.path.join(os.path.join('checkpoints', experiment, scene), "model_modified.pt"))