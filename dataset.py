import os
import numpy as np
import imageio
from utils import extract_patches,blur,upscale
from torch import from_numpy,stack

def dataset(dataset_train_path,batch_size,scale_factor):
    assert(os.path.exists(dataset_train_path))
    data = []
    for file in os.listdir(dataset_train_path):
        if file.endswith('.bmp'):
            filepath = os.path.join(dataset_train_path,file)
            img = imageio.imread(filepath).dot([0.299, 0.587, 0.114])
            patches = extract_patches(img,(36,36),0.166)

            data += [patches[idx] for idx in range(patches.shape[0])]

    mod_data = [from_numpy(np.expand_dims(blur(upscale(patch,scale_factor),scale_factor),0)).float() for patch in data]
    data = [from_numpy(np.expand_dims(upscale(patch,scale_factor),0)).float() for patch in data]
    l = len(data)
    for idx in range(0,l,batch_size):
        yield stack(mod_data[idx:min(idx+batch_size,l)]),stack(data[idx:min(idx+batch_size,l)])

