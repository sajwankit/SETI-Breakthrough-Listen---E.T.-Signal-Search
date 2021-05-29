import numpy as np
import config
from multiprocessing import Pool
import glob
import cv2
from tqdm import tqdm
import time

s = time.time()

class DesignImage():
    def __init__(self, images_set, out_image_size = (224, 224), chl_pos_in_spatial = [0,1,2,3,4,5]):
        self.images_set = images_set
        self.image_paths = list(glob.glob(f'{config.DATA_PATH}{self.images_set}/*/*.npy'))
        # p{channel} means position of channel in the final spatial image,  p{channel}: -1 means a channel is excluded
        self.chl_pos_in_spatial = chl_pos_in_spatial
        self.out_image_size = out_image_size

    def concat_channels_to_spatial(self, image_path):
        image_array_name = image_path.split('/')[-1]
        image_array = np.load(image_path)
        # cv2.imwrite(f'{config.RESIZED_IMAGE_PATH}tesdt.png', image_array)
        image_array_spatial = np.zeros((image_array.shape[1]*2, image_array.shape[2]*3))

        #pos_idx is positions index in the spatial image to be formed. 
        # ---- -273*2 .
        # 0 1         .
        # 2 3         .
        # 4 5       256*3
        # looping over pos_idx values to get corresponding channel 
        for ch_pos_idx in range(0, len(self.chl_pos_in_spatial)):
            if ch_pos_idx % 2 == 0:
                try:
                    channel = self.chl_pos_in_spatial.index(ch_pos_idx)
                    channel_image = image_array[channel,:,:]
                except:
                    channel_image = 0
                x_start = 0
                x_end = image_array.shape[1]
                y_start = int(ch_pos_idx*0.5) * image_array.shape[2]
                y_end = int((ch_pos_idx*0.5)+1) * image_array.shape[2]
                image_array_spatial[x_start: x_end, y_start: y_end] = channel_image
            else:
                try:
                    channel = self.chl_pos_in_spatial.index(ch_pos_idx)
                    channel_image = image_array[channel,:,:]
                except:
                    channel_image = 0
                x_start = image_array.shape[1]
                x_end = 2 * image_array.shape[1]
                y_start = int((ch_pos_idx-1)*0.5)*image_array.shape[2]
                y_end = int((ch_pos_idx-1)*0.5+1)*image_array.shape[2]
                image_array_spatial[x_start: x_end, y_start: y_end] = channel_image

        image_spatial =  cv2.resize(image_array_spatial, dsize=self.out_image_size, interpolation=cv2.INTER_AREA)
        np.save(f'{config.RESIZED_IMAGE_PATH}{self.images_set}/{image_array_name}', image_spatial)
        
        
        ##see progress of p.map
        global s
        if time.time()- s > 10:
          no_files_in_path = len(list(glob.glob(f'{config.RESIZED_IMAGE_PATH}{self.images_set}/*.npy')))
          print(f'{no_files_in_path} images done of {len(self.image_paths)}')
          s = time.time()
        
       # return image_spatial

    def yield_image_array(self):
        for image_path in self.image_paths:
            yield image_path.split('/')[-1], np.load(image_path)


if __name__ == "__main__":
    designImage = DesignImage(images_set = 'test', out_image_size= config.IMAGE_SIZE,  chl_pos_in_spatial = [0,1,2,3,4,5])
    
    
    with Pool(16) as p:
        p.map(designImage.concat_channels_to_spatial, designImage.image_paths)
    print('Done')
    
    #for image_array_name, image_array in tqdm(designImage.yield_image_array()):
    #    image_spatial = designImage.concat_channels_to_spatial(image_array)
    #    np.save(f'{config.RESIZED_IMAGE_PATH}{designImage.images_set}/{image_array_name}', image_spatial)
    