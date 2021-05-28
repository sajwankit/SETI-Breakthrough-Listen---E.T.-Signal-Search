import numpy as np
import config
from multiprocessing import Pool
import glob

class DesignImage():
    def __init__(self, images_set, channel_position_in_concat):
        self.images_set = images_set
        self.image_paths = list(glob(f'{config.DATA_PATH}{self.images_set}/*/*.npy'))
        # p{channel} means position of channel in the final spatial image,  p{channel}: -1 means a channel is excluded
        self.channel_position_in_spatial = [2, 1, 0, 3, 4, 5]

    def concat_channels_to_spatial(self, image_array, chl_pos_in_spatial):

        image_spatial = np.zeros((image_array[1]*3, image_array[2]*2))

        #pos_idx is positions index in the spatial image to be formed. 
        # 0 1
        # 2 3
        # 4 5
        for pos_idx in range(0, image_array.shape[0]):
            if pos_idx % 2 == 0:
                image_spatial[: image_array.shape[1], pos_idx*0.5: (pos_idx*0.5+1) * image_array.shape[2]] = image_array[chl_pos_in_spatial.index(pos_idx)]
            else:
                image_spatial[image_array.shape[1]: 2 * image_array.shape[1], (pos_idx-1)*0.5*image_array.shape[2]: ((pos_idx-1)*0.5+1)*image_array.shape[2]] = image_array[chl_pos_in_spatial.index(pos_idx)]

        return image_spatial

    def yield_image_array(self):
        for image_path in self.image_paths:
            yield np.load(image_path)



designImage = DesignImage(images_set = 'train')
with Pool as p:
    p.map(designImage.concat_channels_to_spatial, designImage.yield_image_array)
