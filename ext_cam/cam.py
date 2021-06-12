import pandas as pd
import numpy as np
import torch
from sklearn import metrics

import config
import dataset
import engine
import models
import seedandlog
import os
import cv2
from tqdm import tqdm
import torch
import glob

if __name__ == '__main__':
    seedandlog.seed_torch(seed=config.SEED)

    data_path = config.DATA_PATH
    device = config.DEVICE
    bs = config.BATCH_SIZE
    target_size = config.TARGET_SIZE

    model = models.Model(pretrained = False)
    model.to(device)
    state = torch.load(f'{config.MODEL_OUTPUT_PATH}{config.MODEL_LOAD_FOR_INFER}_{config.MODEL_NAME}_bs{bs}_size{config.IMAGE_SIZE[0]}_dt{config.DATETIME}.pth')


    needle_target_encoding = {
                'brightpixel':[1, 0, 0, 0, 0, 0, 0],
                'narrowband': [0, 1, 0, 0, 0, 0, 0],
                'narrowbanddrd': [0, 0, 1, 0, 0, 0, 0],
                'noise': [0, 0, 0, 1, 0, 0, 0], 
                'squarepulsednarrowband': [0, 0, 0, 0, 1, 0, 0],
                'squiggle': [0, 0, 0, 0, 0, 1, 0],
                'squigglesquarepulsednarrowband': [0, 0, 0, 0, 0, 0, 1]
                }

    test_image_paths = glob.glob(f'{config.DATA_PATH}test/*/*.png')
    test_targets = [needle_target_encoding[x.split('/')[-2]] for x in test_image_paths]
    test_ids = [int((x.split('/')[-1]).split('_')[0]) for x in test_image_paths]

    test_dataset = dataset.SetiNeedleDataset(image_paths = test_image_paths, ids = test_ids, targets = test_targets)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                                            shuffle=False, 
                                        num_workers=4)

    model.load_state_dict(state['model'])                                    
    

    print(model)
    model.eval()
    finalconv_name = 'layer4'
    # hook the feature extractor
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    model.model.layer4[1]._modules.get('conv2').register_forward_hook(hook_feature)
    print(model.model.layer4[1]._modules.get('conv2'))
    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())
    fc_params = list(model.model._modules.get('last_linear').parameters())
    weight = np.squeeze(fc_params[0].cpu().data.numpy())
    print(fc_params, '\n', weight)
    def returnCAM(feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 256x256
        size_upsample = (384, 512)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
            cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

    predictions = engine.predict(test_loader, model, device)


    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])







#     # simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception

# import io
# import requests
# from PIL import Image
# from torchvision import models, transforms
# from torch.autograd import Variable
# from torch.nn import functional as F
# import numpy as np
# import cv2
# import pdb

# # input image
# LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
# IMG_URL = 'http://media.mlive.com/news_impact/photo/9933031-large.jpg'

# # networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
# model_id = 1
# if model_id == 1:
#     net = models.squeezenet1_1(pretrained=True)
#     finalconv_name = 'features' # this is the last conv layer of the network
# elif model_id == 2:
#     net = models.resnet18(pretrained=True)
#     finalconv_name = 'layer4'
# elif model_id == 3:
#     net = models.densenet161(pretrained=True)
#     finalconv_name = 'features'

# net.eval()

# # hook the feature extractor
# features_blobs = []
# def hook_feature(module, input, output):
#     features_blobs.append(output.data.cpu().numpy())

# net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# # get the softmax weight
# params = list(net.parameters())
# weight_softmax = np.squeeze(params[-2].data.numpy())

# def returnCAM(feature_conv, weight_softmax, class_idx):
#     # generate the class activation maps upsample to 256x256
#     size_upsample = (256, 256)
#     bz, nc, h, w = feature_conv.shape
#     output_cam = []
#     for idx in class_idx:
#         cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
#         cam = cam.reshape(h, w)
#         cam = cam - np.min(cam)
#         cam_img = cam / np.max(cam)
#         cam_img = np.uint8(255 * cam_img)
#         output_cam.append(cv2.resize(cam_img, size_upsample))
#     return output_cam


# normalize = transforms.Normalize(
#    mean=[0.485, 0.456, 0.406],
#    std=[0.229, 0.224, 0.225]
# )
# preprocess = transforms.Compose([
#    transforms.Resize((224,224)),
#    transforms.ToTensor(),
#    normalize
# ])

# response = requests.get(IMG_URL)
# img_pil = Image.open(io.BytesIO(response.content))
# img_pil.save('test.jpg')

# img_tensor = preprocess(img_pil)
# img_variable = Variable(img_tensor.unsqueeze(0))
# logit = net(img_variable)

# # download the imagenet category list
# classes = {int(key):value for (key, value)
#           in requests.get(LABELS_URL).json().items()}

# h_x = F.softmax(logit, dim=1).data.squeeze()
# probs, idx = h_x.sort(0, True)
# probs = probs.numpy()
# idx = idx.numpy()

# # output the prediction
# for i in range(0, 5):
#     print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

# # generate class activation mapping for the top1 prediction
# CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

# # render the CAM and output
# print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
# img = cv2.imread('test.jpg')
# height, width, _ = img.shape
# heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
# result = heatmap * 0.3 + img * 0.5
# cv2.imwrite('CAM.jpg', result)

