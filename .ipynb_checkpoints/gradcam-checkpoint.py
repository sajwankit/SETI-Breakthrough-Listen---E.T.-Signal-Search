import pytorch_grad_cam as pgc
from torchvision.models import resnet50
import cv2
import glob
import random
import torch
import numpy as np

model = resnet50(pretrained = True)
target_layer = model.layer4[-1]

paths = glob.glob('/mnt/gfs/gv1/project_sonar_data/seti/ext/ext_needles/primary_small/train/squiggle/*.png')[:128*4]
images = []
for path in paths:
    image = cv2.imread(path)
    image = np.amax(image) - image
    image = (image/ np.max(image)).reshape(image.shape[2], image.shape[0], image.shape[1])
    images.append(image)

input_tensor = torch.tensor(np.array(images), dtype = torch.float)
target_category = 1

cam = pgc.GradCAM(model = model, target_layer = target_layer, use_cuda = False)
grayscale_cam = cam(input_tensor = input_tensor, target_category = target_category)

grayscale_cam = grayscale_cam[0, :]
