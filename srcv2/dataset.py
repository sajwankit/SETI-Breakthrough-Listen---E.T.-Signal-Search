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

    def normalize(self, image):
        # normalise image with 0 mean, 1 std
        return (image - np.mean(image)) / (np.std(image)).astype(np.float32)
    
    def normalize_xy(self, image):
        image = (image - np.mean(image, axis = 0, keepdims = True))/np.std(image, axis = 0, keepdims = True)
        image = (image - np.mean(image, axis = 1, keepdims = True))/np.std(image, axis = 1, keepdims = True)
        return image
    
    def normalize_ft(self, image, tstacks = 6, fstacks = 1, p=0.5):
        if np.random.uniform(0, 1) <= p: 
            final_shape = image.shape
            chnl_shape = (final_shape[0]//tstacks, final_shape[1]//fstacks) #will be approx to note.
            f = chnl_shape[1]
            t = chnl_shape[0]
            trans_image_array = np.copy(image)
            for t_ in range(tstacks):
                for f_ in range(fstacks):
                    trans_image_array[t_*t:(t_+1)*t, f_*f:(f_+1)*f] = self.normalize_xy(trans_image_array[t_*t:(t_+1)*t, f_*f:(f_+1)*f])
            return trans_image_array
        else:
            return image
    
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
            chnl_needle_mask = needle_mask[chl*t:(chl+1)*t, : f]
            fimg[chl*t:(chl+1)*t, : f][chnl_needle_mask] = (np.random.uniform(0.5, 0.7)*needle_img[chl*t:(chl+1)*t, : f][chnl_needle_mask] + fimg[chl*t:(chl+1)*t, : f][chnl_needle_mask])
        return self.normalize(fimg).astype(np.float32)

    def apply_ext_needle(self):
        ftarget_type = np.random.choice([0, 1], p = [0.35, 0.65])
        needle_type = np.random.choice([1, 2, 5], p = [0.33, 0.33, 0.34])
        
        # needle_target_encoding = {
#             0'brightpixel':[1, 0, 0, 0, 0, 0, 0],
#             1'narrowband': [0, 1, 0, 0, 0, 0, 0],
#             2'narrowbanddrd': [0, 0, 1, 0, 0, 0, 0],
#             3'noise': [0, 0, 0, 1, 0, 0, 0], 
#             4'squarepulsednarrowband': [0, 0, 0, 0, 1, 0, 0],
#             5'squiggle': [0, 0, 0, 0, 0, 1, 0],
#             6'squigglesquarepulsednarrowband': [0, 0, 0, 0, 0, 0, 1]
#             }
        
        needle_mask_path = random.choice(glob.glob(f'{config.NEEDLE_PATH}mask_*_{needle_type}.npy'))
        needle_path = needle_mask_path.replace('mask_', '')
        
#         print(needle_path, needle_mask_path)
        needle_img = np.load(needle_path)
        needle_mask = np.load(needle_mask_path)
#         print(needle_img.shape, needle_mask.shape)
        if ftarget_type == 1:
            chls_to_add_needle = random.sample([0, 2, 4], random.choice([1, 2, 3]))
            trans_image_array = self.add_needle(chls_to_add_needle, needle_img, needle_mask)
        else:
#             needle_img = np.amax(needle_img) - needle_img
            chls_to_add_needle = random.sample([1, 3, 5], random.choice([1, 2, 3]))
            trans_image_array = self.add_needle(chls_to_add_needle, needle_img, needle_mask)
        return trans_image_array, ftarget_type 

class SetiDataset:
    def __init__(self, df, pred=False, augmentations = None):
        self.df = df
        self.resize = resize
        self.augmentations = augmentations
        self.pred = pred

    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, item):

        image = np.load(self.df.loc[item, 'image_path'])
        
        dfidx = self.df.loc[item, 'orig_index']
                
        if config.IMAGE_TYPE == 'orig':
#           converting 6 channels to 1 for original image, inverting off channels
            image = np.vstack(image)
            image = image.astype(np.float32)
            
        if not self.pred:
            target = self.df.loc[item, 'target']
        

        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        '''
        Use following when 6 channels concatenated and 3 channel image generated
        '''
        # imt = ImageTransform()
        
        # if config.APPLY_NEEDLE:
        #     if target == 0 and np.random.uniform(0,1) <=0.55:
        #         image, target = imt.apply_ext_needle()
        
        # if self.augmentations:
        #     image = imt.flip(image = image, p = 0.5)
        #     image = imt.swap_channels(image = image, p = 0.65)
        #     image = imt.drop_channels(image = image, p = 0.25)

        # image0 = np.copy(image)
        # if config.INVERT_OFF_CHANNELS:
        #     '''
        #     inverting off channels
        #     '''
        #     chnl_shape = (config.IMAGE_SIZE[1]//6, config.IMAGE_SIZE[0]//1) #will be approx to note.(time,freq)
        #     f = chnl_shape[1]
        #     t = chnl_shape[0]
        #     '''
        #     image_patches = [self.image_array[c:(c+1)*t, : f]], c = 0, 1, 2 ,3, 4, 5
        #     '''
        #     chnls_to_invert = [1, 3, 5]
        #     for c in chnls_to_invert:
        #         image0[c*t:(c+1)*t, : f] = np.amax(image0[c*t:(c+1)*t, : f]) - image0[c*t:(c+1)*t, : f]
        #     image0 = imt.normalize(image0, )

        # image1 = np.copy(image)
        # image1 = imt.normalize(image1)

        # image2 = imt.normalize_ft(image, p=1)
        # if config.INVERT_OFF_CHANNELS:
        #     '''
        #     inverting off channels
        #     '''
        #     chnl_shape = (config.IMAGE_SIZE[1]//6, config.IMAGE_SIZE[0]//1) #will be approx to note.(time,freq)
        #     f = chnl_shape[1]
        #     t = chnl_shape[0]
        #     '''
        #     image_patches = [self.image_array[c:(c+1)*t, : f]], c = 0, 1, 2 ,3, 4, 5
        #     '''
        #     chnls_to_invert = [1, 3, 5]
        #     for c in chnls_to_invert:
        #         image2[c*t:(c+1)*t, : f] = np.amax(image2[c*t:(c+1)*t, : f]) - image2[c*t:(c+1)*t, : f]
        #     image2 = imt.normalize(image2, )
        
        
        # image3ch = np.zeros((3, image.shape[0], image.shape[1]))
        # image3ch[0] = image0.reshape(1,image0.shape[0],image0.shape[1])
        # image3ch[1] = image1.reshape(1,image1.shape[0],image1.shape[1])
        # image3ch[2] = image2.reshape(1,image2.shape[0],image2.shape[1])

        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        '''
        Use when single channel image passed
        '''  
        image = image.reshape(1,image.shape[0],image.shape[1])

        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
        if not self.pred:
            return{'images': torch.tensor(image, dtype = torch.float), 
                    'targets': torch.tensor(target, dtype = torch.long),
                  'ids': torch.tensor(dfidx, dtype = torch.int32)}
        else:
            return{'images': torch.tensor(image, dtype = torch.float),
                  'ids': torch.tensor(dfidx, dtype = torch.int32)}