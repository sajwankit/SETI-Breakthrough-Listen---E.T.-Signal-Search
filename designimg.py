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
    def __init__(self, images_set, out_image_size = (256, 258), chl_pos_in_spatial = [0,1,2,3,4,5]):
        #out_image_size = (frequency, time)..cv2 needs as (x,y), np.array shape is (y,x)
        self.images_set = images_set
        self.image_paths = list(glob.glob(f'{config.DATA_PATH}{self.images_set}/*/*.npy'))
        # p{channel} means position of channel in the final spatial image,  p{channel}: -1 means a channel is excluded
        self.chl_pos_in_spatial = chl_pos_in_spatial
        self.out_image_size = out_image_size

    def simple_concat(self, image_path):
        image_array_name = image_path.split('/')[-1]
        image_array = np.load(image_path)
        image_array = image_array.astype(np.float32)      
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

    def yield_image_array(self):
        for image_path in self.image_paths:
            yield image_path.split('/')[-1], np.load(image_path)


if __name__ == "__main__":
    train = DesignImage(images_set = 'train', out_image_size= config.IMAGE_SIZE,  chl_pos_in_spatial = [0,1,2,3,4,5])
    test = DesignImage(images_set = 'test', out_image_size= config.IMAGE_SIZE,  chl_pos_in_spatial = [0,1,2,3,4,5])
    # designImage.simple_concat(designImage.image_paths[0])
    
    with Pool() as p:
        p.map(train.simple_concat, train.image_paths)
        p.map(test.simple_concat, test.image_paths)
    print('Done')
    
    #for image_array_name, image_array in tqdm(designImage.yield_image_array()):
    #    image_spatial = designImage.concat_channels_to_spatial(image_array)
    #    np.save(f'{config.RESIZED_IMAGE_PATH}{designImage.images_set}/{image_array_name}', image_spatial)
    