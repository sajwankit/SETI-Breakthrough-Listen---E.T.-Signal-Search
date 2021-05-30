import torch
import numpy as np
from PIL import Image
from PIL import ImagePath
import random


class ImageTransformer():
    def __init__(self, image_array):
        self.image_array = image_array
    def swap_channels(self, p = 0.3, ):
        
        #getting size of image_part of a channel in image_array
        init_shape = (273*2, 256*3)
        final_shape = self.image_array.shape
        chnl_shape = (final_shape//2, final_shape//3) #will be approx to note.


        chnls = {'pos_chnls': [0,2,4], 'neg_chnls': [1,3,5]}
        pos_to_swap = chnls['pos_chnls'].remove(random.choice(chnls['pos_chnls']))
        neg_to_swap = chnls['neg_chnls'].remove(random.choice(chnls['neg_chnls']))
        swap_ops = ['pos_to_swap', 'neg_to_swap', 'both_swap']
        with random.choice(swap_ops) as to_swap:
            if to_swap == 'pos_to_swap':
                temp_image_arr = np.zeros(chnl_shape)


                y_start = 0
                y_end = self.image_array.shape[1]
                x_start = int(ch_pos_idx*0.5) * self.image_array.shape[2]
                x_end = int((ch_pos_idx*0.5)+1) * self.image_array.shape[2]
                image_array_spatial[y_start: y_end, x_start: x_end] = channel_image



        x = chnl_shape[1]
        y = chnl_shape[0]

        c0 = self.image_array[:x, :y]
        c1 = self.image_array[:x, y:]
        c2 = self.image_array[x:2*x, :y]
        c3 = self.image_array[x:2*x, y:]
        c4 = self.image_array[2*x:, :y]
        c5 = self.image_array[2*x:, y:]

        for c in range(6):
            pos_to_swap_copy = pos_to_swap 
            if c in pos_to_swap:
                pos_to_swap.remove(c)
        if c == 0:
            self.image_array[:x, :y] = 

        chnl_positions = [0, 1, 2, 3, 4, 5] # value gives position of channel in spatial image, range gives us the channel indices
        for ch_pos_idx in range(0, len(chnl_positions)):
            if ch_pos_idx % 2 == 0:
                try:
                    channel = chnl_positions.index(ch_pos_idx)
                    channel_image = self.image_array[channel,:,:]
                except:
                    channel_image = 0
                y_start = 0
                y_end = self.image_array.shape[1]
                x_start = int(ch_pos_idx*0.5) * self.image_array.shape[2]
                x_end = int((ch_pos_idx*0.5)+1) * self.image_array.shape[2]
                image_array_spatial[y_start: y_end, x_start: x_end] = channel_image
            else:
                try:
                    channel = chnl_positions.index(ch_pos_idx)
                    channel_image = image_array[channel,:,:]
                except:
                    channel_image = 0
                y_start = image_array.shape[1]
                y_end = 2 * image_array.shape[1]
                x_start = int((ch_pos_idx-1)*0.5)*image_array.shape[2]
                x_end = int((ch_pos_idx-1)*0.5+1)*image_array.shape[2]
                image_array_spatial[y_start: y_end, x_start: x_end] = channel_image

class SetiDataset:
    def __init__(self, image_paths, targets = None, resize=None, augmentations = None):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resizeself.image_array
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, item):
        # image = Image.open(self.image_paths[item])
        image = np.load(self.image_paths[item])

        # #use when using resized images
        # image = image.reshape(1,image.shape[0],image.shape[1])


        if self.targets is not None:
            targets = self.targets[item]

        if self.resize is not None:
            image = image.resize(self.resize[1], self.resize[0], resample = Image.BILINEAR)
        
        image = np.array(image)
        
        if self.augmentations is not None:
            augmented = self.augmentations(image = image)
            image = augmented['image']
        
        #pytorch expects channelHeightWidth instead of HeightWidthChannel
        # image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        if self.targets is not None:
            return{'images': torch.tensor(image, dtype = torch.float), 
                    'targets': torch.tensor(targets, dtype = torch.long)}
        else:
            return{'images': torch.tensor(image, dtype = torch.float)}

