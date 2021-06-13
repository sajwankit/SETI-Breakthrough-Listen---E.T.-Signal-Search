import torch
import numpy as np
from PIL import Image
from PIL import ImagePath
import random
import config
import albumentations as A
import cv2
import glob

class ImageTransform():
    def __init__(self):
        pass
    def normalize(self, image):
        # normalise image with 0 mean, 1 std
        
        return (image - np.mean(image)) / (np.std(image)).astype(np.float32)
    
    def minmax_norm(self, image):
        # min-max to bring image in range 0,1. albumentations requires it.
        return ((image - np.min(image))/(np.max(image) - np.min(image)))
    
    def flip(self,image, p=0.5):
#         transform = A.Compose([
# #             A.OneOf([
# #                     A.RandomBrightnessContrast(brightness_limit = [-0.3,0.2], contrast_limit = [-0.3,0.2], p =0.75),
# #                     A.Sharpen(alpha = [0.1,0.4], lightness = [0.6, 1], p=0.75),
# #             ]),
#             A.HorizontalFlip(p=1),
# #             A.ShiftScaleRotate(shift_limit_x=(-0.08, 0.08), scale_limit=0, rotate_limit=0,
# #                                 p=1)
#                     ])
       
#         trans_image_array = transform(image = self.minmax_norm(np.copy(self.image_array)))['image']
        if np.random.uniform(0, 1) <= p: 
            if np.random.uniform(0, 1) <= 0.5:
                trans_image_array = np.fliplr(image)
            else:
                trans_image_array = np.flip(image)  
            return trans_image_array
        else:
            return image

class SetiNeedleDataset:
    def __init__(self, image_paths, targets = None, ids = None, resize=None, augmentations = None):
        self.image_paths = image_paths #glob.glob(f'{config.NEEDLE_PATH}*/{needle_type}/*.png')
        self.targets = targets
        self.ids = ids
        self.resize = resize
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, item):
        # image = Image.open(self.image_paths[item])
        imt = ImageTransform()
        image = imt.normalize(cv2.imread(self.image_paths[item], cv2.IMREAD_GRAYSCALE))
        
        id = self.ids[item]
            
        if self.targets is not None:
            target = self.targets[item]

        if self.resize is not None:
            image = image.resize(self.resize[1], self.resize[0], resample = Image.BILINEAR)

        imt = ImageTransform()
# #         image = imt.apply_ext_needle()
        if self.augmentations:
            image = imt.flip(image = image, p = 0.5)
#             image = imt.swap_channels(image = image, p = 0.65)
#             image = imt.drop_channels(image = image, p = 0.25)
# #         print('1ds', np.mean(image), np.std(image))
# #         image =  imt.normalize(cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_AREA))

#         if config.INVERT_OFF_CHANNELS:
#             #inverting off channels
#             chnl_shape = (config.IMAGE_SIZE[1]//6, config.IMAGE_SIZE[0]//1) #will be approx to note.(time,freq)
#             f = chnl_shape[1]
#             t = chnl_shape[0]
#             # image_patches = [self.image_array[c:(c+1)*t, : f]], c = 0, 1, 2 ,3, 4, 5
#             chnls_to_invert = [1, 3, 5]
#             max_pix = np.amax(image)
#             for c in chnls_to_invert:
#                 image[c*t:(c+1)*t, : f] = max_pix - image[c*t:(c+1)*t, : f]
#         image = imt.normalize(image)
            
        image = image.reshape(1,image.shape[0],image.shape[1])
        image = np.repeat(image, 3, axis = 0)
        #pytorch expects channelHeightWidth instead of HeightWidthChannel
        # image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    
        if self.targets is not None:
            return{'images': torch.tensor(image.copy(), dtype = torch.float), 
                    'targets': torch.tensor(target, dtype = torch.long),
                  'ids': torch.tensor(id, dtype = torch.int32)}
        else:
            return{'images': torch.tensor(image.copy(), dtype = torch.float),
                  'ids': torch.tensor(id, dtype = torch.int32)}

# i = SetiDataset([f'{config.DATA_PATH}train/1/1a0a41c753e1.npy'], targets = [1], ids =[0], resize=None, augmentations = None)[0]

# i = SetiDataset([f'/content/drive/MyDrive/SETI/resized_images/256256/train/1a0a41c753e1.npy'], targets = [1], ids =[0], resize=None, augmentations = None)[0]
# print(i)