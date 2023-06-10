import torch
from torchvision import transforms

from PIL import Image

n_frame = 3

def update_cur_state(states_, state):

    for i in range(1, n_frame):
        states_[0, (i-1)*3:i*3] = states_[0, i*3:(i+1)*3]
        transforms.functional.to_pil_image(states_[0, (i-1)*3:i*3]).save(f"frame{i}.jpg")

    states_[0, (n_frame-1)*3:n_frame*3] = state
    transforms.functional.to_pil_image(states_[0, (n_frame-1)*3:n_frame*3]).save(f"frame{n_frame}.jpg")

    return states_

frame_state = torch.zeros((1, 9, 4, 4))
frame_state[0, 0:3, :, :] = 0.3
frame_state[0, 3:6, :, :] = 0.6
frame_state[0, 6:9, :, :] = 0.9

def get_i_image(states_, i):
    return states_[0, (i)*3:(i+1)*3]

[transforms.ToPILImage()(get_i_image(frame_state, i)[0]).save(f"test_frame{i}_before.jpg") for i in range(n_frame)]

frame_state = update_cur_state(frame_state, torch.ones((3, 4, 4)))

[transforms.ToPILImage()(get_i_image(frame_state, i)[0]).save(f"test_frame{i}_after.jpg") for i in range(n_frame)]