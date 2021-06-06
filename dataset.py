import torch
import numpy as np
from PIL import Image
from PIL import ImagePath
import random
import config
import albumentations as A

class ImageTransformer():
    def __init__(self, image_array):
        self.image_array = image_array

    def __init__(self, image_array):
        self.image_array = image_array

    def random_brightness_contrast(self, bl = [0, 0], cl = [0.5, 0.5], p =1):
        transform = A.RandomBrightnessContrast(brightness_limit = bl, contrast_limit = cl, p = p)
        trans_image_array = transform(image = self.image_array.astype(np.float32))['image']
        return trans_image_array
    
    def sharpen(self, alpha = [0, 0], lightness = [0.5, 0.5], p =1):
        transform = A.Sharpen(alpha = alpha, lightness = lightness, p=p)
        trans_image_array = transform(image = self.image_array.astype(np.float32))['image']
        return trans_image_array

    def swap_channels(self, p = 0.3, ):
        
        # init_shape = (273*2, 256*3) (f, t)
        final_shape = self.image_array.shape
        chnl_shape = (final_shape[0]//1, final_shape[1]//6) #will be approx to note.


        chnls = {'pos_chnls': [0,2,4], 'neg_chnls': [1,3,5]}
        chnls['pos_chnls'].remove(random.choice(chnls['pos_chnls']))
        chnls['neg_chnls'].remove(random.choice(chnls['neg_chnls']))
        swap_op = random.choice(['pos_chnls', 'neg_chnls', 'both_swap'])

        f = chnl_shape[1]
        t = chnl_shape[0]

        # image_patches = [self.image_array[c:(c+1)*t, : f]], c = 0, 1, 2 ,3, 4, 5

        trans_image_array = np.copy(self.image_array)
        if swap_op == 'pos_chnls' or swap_op == 'both_swap':
            c1 = chnls['pos_chnls'][0]
            c2 = chnls['pos_chnls'][1]
            trans_image_array[c1:(c1+1)*t, : f] = self.image_array[c2:(c2+1)*t, : f]
            trans_image_array[c2:(c2+1)*t, : f] = self.image_array[c1:(c1+1)*t, : f]

        if swap_op == 'neg_chnls' or swap_op == 'both_swap':
            c1 = chnls['neg_chnls'][0]
            c2 = chnls['neg_chnls'][1]
            trans_image_array[c1:(c1+1)*t, : f] = self.image_array[c2:(c2+1)*t, : f]
            trans_image_array[c2:(c2+1)*t, : f] = self.image_array[c1:(c1+1)*t, : f] 

        return trans_image_array

    def drop_channels(self, p = 0.3,):
        #getting size of image_part of a channel in image_array
        # init_shape = (273*2, 256*3)
        final_shape = self.image_array.shape
        chnl_shape = (final_shape[0]//2, final_shape[1]//3) #will be approx to note.

        chnls = {'pos_chnls': [0,2,4], 'neg_chnls': [1,3,5]}
        chnls_to_remove = random.sample(chnls['neg_chnls'], random.choice([1,2]))

        x = chnl_shape[1]
        y = chnl_shape[0]

        # image_patches = [self.image_array[ :y, :x], 0
        #                 self.image_array[ y:2*y, :x], 1
        #                 self.image_array[:y, x:2*x], 2
        #                 self.image_array[y:2*y, x:2*x], 3
        #                 self.image_array[:y, 2*x:3*x], 4
        #                 self.image_array[y:2*y, 2*x:3*x]] 5
        
        out_image_array = self.image_array
        for c in chnls_to_remove:
            if c == 1:
                out_image_array[ y:2*y, :x] = 0
            if c == 3:
                out_image_array[y:2*y, x:2*x] = 0
            if c == 5:
                out_image_array[y:2*y, 2*x:3*x] = 0
        # np.save(f'/content/drive/MyDrive/SETI/drop.npy', out_image_array)
        return out_image_array

class SetiDataset:
    def __init__(self, image_paths, targets = None, ids = None, resize=None, augmentations = None):
        self.image_paths = image_paths
        self.targets = targets
        self.ids = ids
        self.resize = resize
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, item):
        # image = Image.open(self.image_paths[item])
        image = np.load(self.image_paths[item])
                    
        id = self.ids[item]
        
        if not config.ORIG_IMAGE:
            # #use when using resized images
            image = image.reshape(1,image.shape[0],image.shape[1])
        else:
            #converting 6 channels to 1 for original image, inverting off channels
            max_pix = np.amax(image)
            image[1] = max_pix - image[1]
            image[3] = max_pix - image[3]
            image[5] = max_pix - image[5]
            image = np.vstack(image).transpose((1, 0))
            image = image.reshape(1,image.shape[0],image.shape[1])


        if self.targets is not None:
            target = self.targets[item]

        if self.resize is not None:
            image = image.resize(self.resize[1], self.resize[0], resample = Image.BILINEAR)
        
        image = np.array(image)

#        if np.random.uniform(0, 1)>0.5:
#            image = ImageTransformer(image).swap_channels()
#        
#        if np.random.uniform(0, 1)>0.5:
#            image = ImageTransformer(image).drop_channels()

        if self.augmentations is not None:
            augmented = self.augmentations(image = image)
            image = augmented['image']
        
        #pytorch expects channelHeightWidth instead of HeightWidthChannel
        # image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        if self.targets is not None:
            return{'images': torch.tensor(image, dtype = torch.float), 
                    'targets': torch.tensor(target, dtype = torch.long),
                  'ids': torch.tensor(id, dtype = torch.int32)}
        else:
            return{'images': torch.tensor(image, dtype = torch.float),
                  'ids': torch.tensor(id, dtype = torch.int32)}

# SetiDataset(['/content/drive/MyDrive/SETI/resized_images/128128/train/ecb6df8c6f71.npy'], targets = None, resize=None, augmentations = None)[0]
