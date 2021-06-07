import numpy as np
import config
from multiprocessing import Pool
import glob
import cv2
from tqdm import tqdm
import time
import os

s = time.time()

class DesignImage():
    def __init__(self, images_set, out_image_size = (256, 256), chl_pos_in_spatial = [0,1,2,3,4,5]):
        self.images_set = images_set
        self.image_paths = list(glob.glob(f'{config.DATA_PATH}{self.images_set}/*/*.npy'))
        # p{channel} means position of channel in the final spatial image,  p{channel}: -1 means a channel is excluded
        self.chl_pos_in_spatial = chl_pos_in_spatial
        self.out_image_size = out_image_size

    def simple_concat(self, image_path):
        image_array_name = image_path.split('/')[-1]
        image_array = np.load(image_path)
        image_array = image_array.astype(np.float32)
        
        #inverting off channels
        max_pix = np.amax(image_array)
        image_array[1] = max_pix - image_array[1]
        image_array[3] = max_pix - image_array[3]
        image_array[5] = max_pix - image_array[5]
        
        image_array = np.vstack(image_array)
        image_spatial =  cv2.resize(image_array, dsize=self.out_image_size, interpolation=cv2.INTER_AREA)
        
        if config.SAVE_IMAGE:
            try:
                os.mkdir(f'{config.RESIZED_IMAGE_PATH}{self.images_set}')
            except:
                pass
            np.save(f'{config.RESIZED_IMAGE_PATH}{self.images_set}/{image_array_name}', image_spatial)
        else:
            return image_spatial

        ##see progress of p.map
        global s
        if time.time()- s > 10:
            no_files_in_path = len(list(glob.glob(f'{config.RESIZED_IMAGE_PATH}{self.images_set}/*.npy')))
            print(f'{no_files_in_path} images done of {len(self.image_paths)}')
            s = time.time()

    def concat_channels_to_spatial(self, image_path):
        image_array_name = image_path.split('/')[-1]
        image_array = np.load(image_path)
        # cv2.imwrite(f'{config.RESIZED_IMAGE_PATH}tesdt.png', image_array)
        image_array_spatial = np.zeros((image_array.shape[1]*2, image_array.shape[2]*3))

        #pos_idx is positions index in the spatial image to be formed. 
        # ---- -256*3 .
        # 0 1         .
        # 2 3         .
        # 4 5       273*2
        # looping over pos_idx values to get corresponding channel 
        for ch_pos_idx in range(0, len(self.chl_pos_in_spatial)):
            if ch_pos_idx % 2 == 0:
                try:
                    channel = self.chl_pos_in_spatial.index(ch_pos_idx)
                    channel_image = image_array[channel,:,:]
                except:
                    channel_image = 0
                y_start = 0
                y_end = image_array.shape[1]
                x_start = int(ch_pos_idx*0.5) * image_array.shape[2]
                x_end = int((ch_pos_idx*0.5)+1) * image_array.shape[2]
                image_array_spatial[y_start: y_end, x_start: x_end] = channel_image
            else:
                try:
                    channel = self.chl_pos_in_spatial.index(ch_pos_idx)
                    channel_image = image_array[channel,:,:]
                except:
                    channel_image = 0
                y_start = image_array.shape[1]
                y_end = 2 * image_array.shape[1]
                x_start = int((ch_pos_idx-1)*0.5)*image_array.shape[2]
                x_end = int((ch_pos_idx-1)*0.5+1)*image_array.shape[2]
                image_array_spatial[y_start: y_end, x_start: x_end] = channel_image

        image_spatial =  cv2.resize(image_array_spatial, dsize=self.out_image_size, interpolation=cv2.INTER_AREA)

        if config.SAVE_IMAGE:
            np.save(f'{config.RESIZED_IMAGE_PATH}{self.images_set}/{image_array_name}', image_spatial)
        else:
            return image_spatial
        
        
        #see progress of p.map
        global s
        if time.time()- s > 10:
            no_files_in_path = len(list(glob.glob(f'{config.RESIZED_IMAGE_PATH}{self.images_set}/*.npy')))
            print(f'{no_files_in_path} images done of {len(self.image_paths)}')
            s = time.time()
        
        return image_spatial

    def yield_image_array(self):
        for image_path in self.image_paths:
            yield image_path.split('/')[-1], np.load(image_path)


if __name__ == "__main__":
    designImage = DesignImage(images_set = 'train', out_image_size= config.IMAGE_SIZE,  chl_pos_in_spatial = [0,1,2,3,4,5])
    # designImage.simple_concat(designImage.image_paths[0])
    
    with Pool() as p:
        p.map(designImage.simple_concat, designImage.image_paths)
    print('Done')
    
    #for image_array_name, image_array in tqdm(designImage.yield_image_array()):
    #    image_spatial = designImage.concat_channels_to_spatial(image_array)
    #    np.save(f'{config.RESIZED_IMAGE_PATH}{designImage.images_set}/{image_array_name}', image_spatial)
    