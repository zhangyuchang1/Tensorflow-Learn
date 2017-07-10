import numpy as np
data_dir = 'vgg16_pretrain/vgg16.npy'

data = np.load(data_dir, encoding='latin1').item()
print(data)