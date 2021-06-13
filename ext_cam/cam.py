import numpy as np
import torch
import config
import dataset
import models
import seedandlog
import cv2
import torch
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

outsize = config.OUT_IMAGE_SIZE

seedandlog.seed_torch(seed=config.SEED)
imt = dataset.ImageTransform()
def returnCAM(feature_conv, weight, class_idx):
    # generate the class activation maps upsample
    size_upsample = (512, 384) #original image size used for train, of ext needle
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for i in range(bz):
        cam = weight[class_idx[i]].dot(feature_conv[i, :, :, :].reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        output_cam.append(imt.normalize(cv2.resize(cam, size_upsample)))
    return output_cam

class SaveFeatures():
    def __init__(self, m):
        """ Extract pretrained activations"""
        self.hook = m.register_forward_hook(self.hook_fn)
        self.features = None
    def hook_fn(self, module, input, output):
        self.features = output.data.cpu().numpy()
    def remove(self):
        self.hook.remove()

data_path = config.DATA_PATH
device = config.DEVICE
bs = config.BATCH_SIZE
target_size = config.TARGET_SIZE

model = models.Model(pretrained = False)
model.to(device)
state = torch.load(f'{config.MODEL_OUTPUT_PATH}{config.MODEL_LOAD_FOR_INFER}_{config.MODEL_NAME}_bs{bs}_size{config.IMAGE_SIZE[0]}_dt{config.DATETIME}.pth')

model.load_state_dict(state['model'])                                    

for param in model.parameters():
    param.requires_grad = False

model.eval()

conv_layer = model.model.layer4[1]._modules.get('conv2')
print(conv_layer)

fc_params = list(model.model._modules.get('last_linear').parameters())
weight = np.squeeze(fc_params[0].cpu().data.numpy())

# needle_target_encoding = {
#             'brightpixel':[1, 0, 0, 0, 0, 0, 0],
#             'narrowband': [0, 1, 0, 0, 0, 0, 0],
#             'narrowbanddrd': [0, 0, 1, 0, 0, 0, 0],
#             'noise': [0, 0, 0, 1, 0, 0, 0], 
#             'squarepulsednarrowband': [0, 0, 0, 0, 1, 0, 0],
#             'squiggle': [0, 0, 0, 0, 0, 1, 0],
#             'squigglesquarepulsednarrowband': [0, 0, 0, 0, 0, 0, 1]
#             }

# test_image_paths = glob.glob(f'{config.DATA_PATH}valid/*/*.png')
# test_targets = [needle_target_encoding[x.split('/')[-2]] for x in test_image_paths]
# test_ids = [int((x.split('/')[-1]).split('_')[0]) for x in test_image_paths]

# test_dataset = dataset.SetiNeedleDataset(image_paths = test_image_paths, ids = test_ids, targets = test_targets)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
#                                         shuffle=False, 
#                                     num_workers=4)



# for i, data in enumerate(test_loader):
#     images = data['images']
#     targets = data['targets']
#     ids = data['ids']

#     images = images.to(device, dtype = torch.float)
#     activated_features = SaveFeatures(conv_layer)

#     outputs = model(images)
#     outputs = torch.sigmoid(outputs).detach().cpu().numpy().tolist()

#     class_predictions = np.argmax(np.array(outputs), axis = 1)
#     heatmaps = returnCAM(activated_features.features, weight, class_predictions)

#     heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
#     result = heatmap * 0.3 + img * 0.5
#     cv2.imwrite('CAM.jpg', result)

def generate_heatmaps(images):
    # input: batch of image arrays of shape : (bs, 3, image_size[0], image_size[1])
    images = images.to(device, dtype = torch.float)
    activated_features = SaveFeatures(conv_layer)

    outputs = model(images)
    outputs = torch.sigmoid(outputs).detach().cpu().numpy().tolist()

    class_predictions_acc = np.max(np.array(outputs), axis = 1)
    class_predictions = np.argmax(np.array(outputs), axis = 1)
    heatmaps = returnCAM(activated_features.features, weight, class_predictions)
    return heatmaps, class_predictions, class_predictions_acc

masks = []
needle_images = []
def generate_mask(heatmaps, images, class_predictions=None, class_predictions_acc= None):
    images = np.array(images[:, 0, :, :])
    heatmaps = np.array(heatmaps)
    for i in tqdm(range(len(heatmaps))):
        if class_predictions_acc[i]>=0.99:
            image = imt.normalize(cv2.resize(images[i], dsize = outsize, interpolation=cv2.INTER_AREA))
            heatmap = cv2.resize(heatmaps[i], dsize = outsize, interpolation=cv2.INTER_AREA)
            mask = heatmap>0
            
            if config.SAVE_NEEDLES:
                np.save( f'{config.NEEDLE_PATH}{i}_{class_predictions[i]}.npy', image)
                np.save( f'{config.NEEDLE_PATH}mask_{i}_{class_predictions[i]}.npy', mask)
            else:
                masks.append(mask)
                needle_images.append(image)
        
def plot_heatmaps(masks, needle_images):
    for i in range(len(needle_images)):
        plt.imshow(needle_images[i])
        plt.imshow(masks[i], alpha=0.4, cmap='jet')
        plt.show()
        print('\n\n')

# test_images_paths = [
#     f'{config.DATA_PATH}train/squigglesquarepulsednarrowband/3_squigglesquarepulsednarrowband.png',
#     f'{config.DATA_PATH}train/squiggle/88_squiggle.png',
#     f'{config.DATA_PATH}train/squarepulsednarrowband/125_squarepulsednarrowband.png',
#     f'{config.DATA_PATH}train/narrowbanddrd/118_narrowbanddrd.png',
#     f'{config.DATA_PATH}train/narrowband/92_narrowband.png'
# ]

# needle_target_encoding = {
#             'brightpixel':[1, 0, 0, 0, 0, 0, 0],
#             'narrowband': [0, 1, 0, 0, 0, 0, 0],
#             'narrowbanddrd': [0, 0, 1, 0, 0, 0, 0],
#             'noise': [0, 0, 0, 1, 0, 0, 0], 
#             'squarepulsednarrowband': [0, 0, 0, 0, 1, 0, 0],
#             'squiggle': [0, 0, 0, 0, 0, 1, 0],
#             'squigglesquarepulsednarrowband': [0, 0, 0, 0, 0, 0, 1]
#             }

def glob_needle():
    needles = []
    types = [
        'narrowband',
        'narrowbanddrd',
#         'squarepulsednarrowband',
        'squiggle',
#         'squigglesquarepulsednarrowband'
        ]
    for t in types:
        needles.extend(glob.glob(f'{config.DATA_PATH}valid/{t}/*.png')+glob.glob(f'{config.DATA_PATH}test/{t}/*.png'))
    return needles
test_images_paths = glob_needle()[:]
# print(test_images_paths)

test_images = []
for p in test_images_paths:
    image = imt.normalize(cv2.imread(p, cv2.IMREAD_GRAYSCALE))
    image = image.reshape(1,image.shape[0],image.shape[1])
    image = np.repeat(image, 3, axis = 0)
    test_images.append(image)
test_images = torch.tensor(test_images, dtype = torch.float)

heatmaps, class_predictions, class_predictions_acc = generate_heatmaps(test_images)
generate_mask(heatmaps, test_images, class_predictions, class_predictions_acc)
plot_heatmaps(masks, needle_images)

