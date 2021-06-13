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
    def __init__(self, image_array):
        self.image_array = image_array

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
            trans_image_array = np.fliplr(image)
            return trans_image_array
        else:
            return image

    def swap_channels(self, image, p = 0.3):
        if np.random.uniform(0, 1) <= p:
            # init_shape = (t, f)
            final_shape = image.shape
            chnl_shape = (final_shape[0]//6, final_shape[1]//1) #will be approx to note.


            chnls = {'pos_chnls': [0,2,4], 'neg_chnls': [1,3,5]}
            chnls['pos_chnls'].remove(random.choice(chnls['pos_chnls']))
            chnls['neg_chnls'].remove(random.choice(chnls['neg_chnls']))
            swap_op = random.choice(['pos_chnls', 'neg_chnls', 'both_swap'])

            f = chnl_shape[1]
            t = chnl_shape[0]

            # image_patches = [self.image_array[c:(c+1)*t, : f]], c = 0, 1, 2 ,3, 4, 5
            trans_image_array = np.copy(image)
            if swap_op == 'pos_chnls' or swap_op == 'both_swap':
                c1 = chnls['pos_chnls'][0]
                c2 = chnls['pos_chnls'][1]
#                 print(f'swapping{c1}{c2}')
                trans_image_array[c1*t:(c1+1)*t, : f] = image[c2*t:(c2+1)*t, : f]
                trans_image_array[c2*t:(c2+1)*t, : f] = image[c1*t:(c1+1)*t, : f]

            if swap_op == 'neg_chnls' or swap_op == 'both_swap':
                c1 = chnls['neg_chnls'][0]
                c2 = chnls['neg_chnls'][1]
#                 print(f'swapping{c1}{c2}')
                trans_image_array[c1*t:(c1+1)*t, : f] = image[c2*t:(c2+1)*t, : f]
                trans_image_array[c2*t:(c2+1)*t, : f] = image[c1*t:(c1+1)*t, : f] 

            return trans_image_array
        else:
            return image.astype(np.float32)

    def drop_channels(self, image, p = 0.3,):
        if np.random.uniform(0, 1) <= p:
            # init_shape = (t, f)
            final_shape = image.shape
            chnl_shape = (final_shape[0]//6, final_shape[1]//1) #will be approx to note.

            chnls = {'pos_chnls': [0,2,4], 'neg_chnls': [1,3,5]}
            chnls_to_remove = random.sample(chnls['neg_chnls'], random.choice([1,2]))

            f = chnl_shape[1]
            t = chnl_shape[0]

            # image_patches = [self.image_array[c:(c+1)*t, : f]], c = 0, 1, 2 ,3, 4, 5
            trans_image_array = np.copy(image)
            for c in chnls_to_remove:
                trans_image_array[c*t:(c+1)*t, : f] = 0.25*image[c*t:(c+1)*t, : f]

            return trans_image_array
        else:
            return image.astype(np.float32)
    
    def add_needle(self, chls_to_add_needle, needle_img, needle_mask):
        fimg = np.copy(self.image_array)
        final_shape = fimg.shape
        chnl_shape = (final_shape[0]//6, final_shape[1]//1) #will be approx to note.
        f = chnl_shape[1]
        t = chnl_shape[0]
        
        for chl in chls_to_add_needle:
            fimg[chl*t:(chl+1)*t, : f][needle_mask] = self.normalize(needle_img[needle_mask] + fimg[chl*t:(chl+1)*t, : f][needle_mask])
        return self.normalize(fimg).astype(np.float32)

    def apply_ext_needle(self):
        ftarget_type = random.choice([0, 1])
        needle_type = random.choice([
        1,
        2,
        5,
        ])
        
        # needle_target_encoding = {
#             0'brightpixel':[1, 0, 0, 0, 0, 0, 0],
#             1'narrowband': [0, 1, 0, 0, 0, 0, 0],
#             2'narrowbanddrd': [0, 0, 1, 0, 0, 0, 0],
#             3'noise': [0, 0, 0, 1, 0, 0, 0], 
#             4'squarepulsednarrowband': [0, 0, 0, 0, 1, 0, 0],
#             5'squiggle': [0, 0, 0, 0, 0, 1, 0],
#             6'squigglesquarepulsednarrowband': [0, 0, 0, 0, 0, 0, 1]
#             }
        needle_path = random.choice(glob.glob(f'{config.NEEDLE_PATH}*_{needle_type}.npy'))
        needle_mask_path = f'{config.NEEDLE_PATH}mask_{needle_path.split('/')[-1]}'
        print(needle_path, needle_mask_path)
        needle_img = np.load(needle_path)
        needle_mask = np.load(needle_mask_path)
        print(needle_img.shape, needle_mask.shape)
        if ftarget_type == 1:
            chls_to_add_needle = random.sample([0, 2, 4], random.choice([1, 2, 3]))
            trans_image_array = self.add_needle(chls_to_add_needle, needle_img, needle_mask)
        else:
#             needle_img = np.amax(needle_img) - needle_img
            chls_to_add_needle = random.sample([1, 3, 5], random.choice([1, 2, 3]))
            trans_image_array = self.add_needle(chls_to_add_needle, needle_img, needle_mask)
        return trans_image_array

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
                
        if config.ORIG_IMAGE:
#           converting 6 channels to 1 for original image, inverting off channels
            image = np.vstack(image)
            image = image.astype(np.float32)
            
        if self.targets is not None:
            target = self.targets[item]

        if self.resize is not None:
            image = image.resize(self.resize[1], self.resize[0], resample = Image.BILINEAR)

            
            
        imt = ImageTransform(image)
        if config.APPLY_NEEDLE:
            image = imt.apply_ext_needle()
        if self.augmentations:
            image = imt.flip(image = image, p = 0.5)
            image = imt.swap_channels(image = image, p = 0.65)
            image = imt.drop_channels(image = image, p = 0.25)
#         print('1ds', np.mean(image), np.std(image))
#         image =  imt.normalize(cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_AREA))

        if config.INVERT_OFF_CHANNELS:
            #inverting off channels
            chnl_shape = (config.IMAGE_SIZE[1]//6, config.IMAGE_SIZE[0]//1) #will be approx to note.(time,freq)
            f = chnl_shape[1]
            t = chnl_shape[0]
            # image_patches = [self.image_array[c:(c+1)*t, : f]], c = 0, 1, 2 ,3, 4, 5
            chnls_to_invert = [1, 3, 5]
            max_pix = np.amax(image)
            for c in chnls_to_invert:
                image[c*t:(c+1)*t, : f] = max_pix - image[c*t:(c+1)*t, : f]
        image = imt.normalize(image)
        
        
        
        image = image.reshape(1,image.shape[0],image.shape[1])
        
        #pytorch expects channelHeightWidth instead of HeightWidthChannel
        # image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    
        if self.targets is not None:
            return{'images': torch.tensor(image, dtype = torch.float), 
                    'targets': torch.tensor(target, dtype = torch.long),
                  'ids': torch.tensor(id, dtype = torch.int32)}
        else:
            return{'images': torch.tensor(image, dtype = torch.float),
                  'ids': torch.tensor(id, dtype = torch.int32)}

# i = SetiDataset([f'{config.DATA_PATH}train/1/1a0a41c753e1.npy'], targets = [1], ids =[0], resize=None, augmentations = None)[0]

# i = SetiDataset([f'/content/drive/MyDrive/SETI/resized_images/256256/train/1a0a41c753e1.npy'], targets = [1], ids =[0], resize=None, augmentations = None)[0]
# print(i)