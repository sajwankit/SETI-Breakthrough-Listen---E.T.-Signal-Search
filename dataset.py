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
        swap_op = random.choice(['pos_to_swap', 'neg_to_swap', 'both_swap'])

        x = chnl_shape[1]
        y = chnl_shape[0]

        image_patches = [self.image_array[:x, :y], 
                        self.image_array[:x, y:2*y],
                        self.image_array[x:2*x, :y],
                        self.image_array[x:2*x, y:2*y],
                        self.image_array[2*x:3*x, :y],
                        self.image_array[2*x:3*x, y:2*y]]
        out_image_array = self.image_array
        if swap_op == 'pos_to_swap' or swap_op == 'both_swap':
            if pos_to_swap[0] == 0:
                out_image_array[:x, :y] = image_patches[pos_to_swap[1]]

            if pos_to_swap[0] == 2:
                out_image_array[x:2*x, :y] = image_patches[pos_to_swap[1]]

            if pos_to_swap[0] == 4:
                out_image_array[2*x:, :y] = image_patches[pos_to_swap[1]]


        if swap_op == 'neg_to_swap' or swap_op == 'both_swap':
            if neg_to_swap[0] == 1:
                out_image_array[:x, y:2*y] = image_patches[neg_to_swap[1]]

            if neg_to_swap[0] == 3:
                out_image_array[x:2*x, y:2*y] = image_patches[neg_to_swap[1]]

            if neg_to_swap[0] == 5:
                out_image_array[2*x:3*x, y:2*y]= image_patches[neg_to_swap[1]]
        
        return out_image_array



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

