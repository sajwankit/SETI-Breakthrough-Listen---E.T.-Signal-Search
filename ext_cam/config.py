import datetime
import pytz
import os

tz_NY = pytz.timezone('Asia/Kolkata')
datetime_NY = datetime.datetime.now(tz_NY)


# DATETIME = datetime_NY.strftime('%m%d%H')

#for inference
DATETIME = '061303'

i = 0
input_path = ['/mnt/gfs/gv1/project_sonar_data/seti/ext/ext_needles/primary_small/', '/content/drive/MyDrive/SETI/primary_small/']
DATA_PATH = input_path[i]

ORIG_IMAGE = True
IMAGE_SIZE = (384,512) # (freq, time): aligning with cv2, not to confuse with np.array shape. note: to fix: this is wrong should be (512, 384), code works as this is redundant.check.dont use.
if not ORIG_IMAGE:
    IMAGE_SIZE = (256,258) # (freq, time): aligning with cv2, not to confuse with np.array shape
    resize_image_path = [f'/mnt/gfs/gv1/project_sonar_data/seti/resized_images_seti/{IMAGE_SIZE[0]}{IMAGE_SIZE[1]}/',
                         f'/content/drive/MyDrive/SETI/resized_images/{IMAGE_SIZE[0]}{IMAGE_SIZE[1]}/']



    RESIZED_IMAGE_PATH = resize_image_path[i]
    try:
        os.makedirs(RESIZED_IMAGE_PATH[:-1])
    except:
        print(f'error creating {RESIZED_IMAGE_PATH[:-1]}')
    SAVE_IMAGE = True

OUT_IMAGE_SIZE = (256, 258)    
SAVE_NEEDLES = False

INVERT_OFF_CHANNELS = True

SEED = 42
DEBUG = False
MIXED_PRECISION = True
MIXUP = False
MIXUP_APLHA = 1

needle_path = ['/mnt/gfs/gv1/project_sonar_data/seti/needles/', '/content/drive/MyDrive/SETI/ext_needle/']
NEEDLE_PATH = needle_path[i]
LOAD_SAVED_MODEL = False
DEVICE = 'cuda'
EPOCHS = 25
BATCH_SIZE = 8
TARGET_SIZE = 7
FOLDS = 4
MODEL_NAME = 'resnet18'
CHANNELS = 1
MODEL_LOAD_FOR_INFER = 'acc'

LEARNING_RATE = 5e-5
FACTOR = 0.1
PATIENCE = 2
EPS = 1e-8

out_path = ['/home/asajw/seti_models/cam/', '/content/drive/MyDrive/SETI/output/']
MODEL_OUTPUT_PATH = out_path[i]

AUG = 'SwapDropFlip'

log_path = ['/home/asajw/SETI/output/cam/', '/content/SETI/output/cam/']
foldername = f'{MODEL_NAME}_dt{DATETIME}'
LOG_DIR = f'{os.path.join(log_path[i], foldername)}/'
try:
    os.mkdir(LOG_DIR[:-1])
except:
    print('folder exists, make sure this call is from inference.py')

INFER = True



