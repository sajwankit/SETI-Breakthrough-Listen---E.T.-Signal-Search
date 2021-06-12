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

seedandlog.seed_torch(seed=config.SEED)

def returnCAM(feature_conv, weight, class_idx):
    # generate the class activation maps upsample
    size_upsample = (512, 384)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight[idx].dot(feature_conv[0,:, :, ].reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        output_cam.append(cv2.resize(cam_img, size_upsample))
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

model.load_state_dict(state['model'])                                    

for param in model.parameters():
    param.requires_grad = False

model.eval()

conv_layer = model.model.layer4[1]._modules.get('conv2')
print(conv_layer)

fc_params = list(model.model._modules.get('last_linear').parameters())
weight = np.squeeze(fc_params[0].cpu().data.numpy())

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

    class_predictions = np.argmax(np.array(outputs), axis = 1)
    heatmaps = returnCAM(activated_features.features, weight, class_predictions)
    return heatmaps, class_predictions

def plot_heatmaps(heatmaps, images, class_predictions=None):
    
    for i, hm in enumerate(heatmaps):
        # hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        # result = hm * 0.3 + images[i] * 0.5
        print(class_predictions[i])
        plt.imshow(images[i][0].reshape(-1,512))
        plt.imshow(hm, alpha=0.4, cmap='jet')
        plt.show()
        print('\n\n')

test_images_paths = [
    '/content/primary_small/train/squigglesquarepulsednarrowband/3_squigglesquarepulsednarrowband.png',
    '/content/primary_small/train/squiggle/88_squiggle.png',
    '/content/primary_small/train/squarepulsednarrowband/125_squarepulsednarrowband.png',
    '/content/primary_small/train/narrowbanddrd/118_narrowbanddrd.png',
    '/content/primary_small/train/narrowband/92_narrowband.png'
]
test_images = []
for p in test_images_paths:
    imt = dataset.ImageTransform()
    image = imt.normalize(cv2.imread(p, cv2.IMREAD_GRAYSCALE))
    image = image.reshape(1,image.shape[0],image.shape[1])
    image = np.repeat(image, 3, axis = 0)
    test_images.append(image)
test_images = torch.tensor(test_images, dtype = torch.float)

heatmaps, class_predictions = generate_heatmaps(test_images)
plot_heatmaps(heatmaps, test_images, class_predictions)

