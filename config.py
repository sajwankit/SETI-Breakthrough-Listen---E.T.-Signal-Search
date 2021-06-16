import datetime
import pytz
import os

tz_NY = pytz.timezone('Asia/Kolkata')
datetime_NY = datetime.datetime.now(tz_NY)


DATETIME = datetime_NY.strftime('%m%d%H')

#for inference
# DATETIME = '061604'

i = 0
input_path = ['/mnt/gfs/gv1/project_sonar_data/seti/', '/content/drive/MyDrive/SETI/input/']
DATA_PATH = input_path[i]

ORIG_IMAGE = False
IMAGE_SIZE = (256,273) # (freq, time): aligning with cv2, not to confuse with np.array shape
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

INVERT_OFF_CHANNELS = True

SEED = 42
DEBUG = False
MIXED_PRECISION = True
MIXUP = True
MIXUP_APLHA = 1

needle_path = ['/mnt/gfs/gv1/project_sonar_data/seti/needles/', '/content/drive/MyDrive/SETI/ext_needle/']
NEEDLE_PATH = needle_path[i]
APPLY_NEEDLE = False
LOAD_SAVED_MODEL = False
DEVICE = 'cuda'
EPOCHS = 40
BATCH_SIZE = 32
TARGET_SIZE = 1
FOLDS = 4
MODEL_NAME = 'resnet18d'
CHANNELS = 1
MODEL_LOAD_FOR_INFER = 'auc'


CURRENT_EPOCH = 0
OHEM_LOSS = False
OHEM_RATE = 0.7

SCHEDULER = 'CosineAnnealingWarmRestarts'
LEARNING_RATE = 1e-4
FACTOR = 0.1
PATIENCE = 2
EPS = 1e-8

out_path = ['/home/asajw/seti_models/', '/content/drive/MyDrive/SETI/output/']
MODEL_OUTPUT_PATH = out_path[i]

AUG = 'SwapDropFlip'

SAVED_MODEL_NAME = f'{MODEL_NAME}_bs{BATCH_SIZE}_AllChl{IMAGE_SIZE[0]}{IMAGE_SIZE[1]}_mixup{MIXUP}_aug{AUG}_ohem{OHEM_LOSS}_scd{SCHEDULER}_normalized_ft_dt{DATETIME}'

log_path = ['/home/asajw/SETI/output/', '/content/SETI/output/']

LOG_DIR = f'{os.path.join(log_path[i], SAVED_MODEL_NAME)}/'
try:
    os.mkdir(LOG_DIR[:-1])
except:
    print('folder exists, make sure this call is from inference.py')

INFER = True



