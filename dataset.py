import torch
import numpy as np
from PIL import Image
from PIL import ImagePath

class SetiDataset:
    def __init__(self, image_paths, targets = None, resize=None, augmentations = None):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
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

