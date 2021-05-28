import numpy as np
import config
from multiprocessing import Pool
import glob

class DesignImage():
    def __init__(self, images_set):
        self.images_set = images_set
        self.image_paths = list(glob(f'{config.DATA_PATH}{self.images_set}/*/*.npy'))

    def concat_channels(self):

        def image_concat
    def yield_imgarr(self):
        for image_path in self.image_paths:
            yield np.load(image_path)



designImage = DesignImage(images_set = 'train')
with Pool as p:
    p.map(designImage.concat_channels, designImage.yield_imgarr)
