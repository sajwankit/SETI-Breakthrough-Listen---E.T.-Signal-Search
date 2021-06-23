import numpy as np
from PIL import Image
from PIL import ImagePath
import glob

import config
data_path = config.DATA_PATH
img_p = '/content/drive/MyDrive/SETI/input/train/d/d9c8576acaa0.npy'
image = np.load(img_p)
print(image.shape)