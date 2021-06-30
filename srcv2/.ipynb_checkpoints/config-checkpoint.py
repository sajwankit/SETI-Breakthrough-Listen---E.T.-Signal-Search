import os
i = 0


'''
REQUIRED INPUT PATHS
'''
input_path = ['/mnt/gfs/gv1/project_sonar_data/seti/', '/content/drive/MyDrive/SETI/input/', '/kaggle/input/256258normed/']
DATA_PATH = input_path[i]

ORIG_IMAGE = False
IMAGE_SIZE = (256,273) # (freq, time): aligning with cv2, not to confuse with np.array shape
if not ORIG_IMAGE:
    NORM_IMAGE_SIZE = (256,273)
    norm_image_path = [f'/mnt/gfs/gv1/project_sonar_data/seti/normalized_images_seti/{NORM_IMAGE_SIZE[0]}{NORM_IMAGE_SIZE[1]}/',
                         f'/content/drive/MyDrive/SETI/normalized_images_seti/{NORM_IMAGE_SIZE[0]}{NORM_IMAGE_SIZE[1]}/',
                        f'/kaggle/working/256258normed/']
    NORM_IMAGE_PATH = norm_image_path[i]
    try:
        os.makedirs(NORM_IMAGE_PATH[:-1])
    except:
        print(f'error creating {NORM_IMAGE_PATH[:-1]}')
    SAVE_IMAGE = True
    
    
    IMAGE_SIZE = (256,258) # (freq, time): aligning with cv2, not to confuse with np.array shape
    resize_image_path = [f'/mnt/gfs/gv1/project_sonar_data/seti/resized_images_seti/{IMAGE_SIZE[0]}{IMAGE_SIZE[1]}/',
                         f'/content/drive/MyDrive/SETI/resized_images/{IMAGE_SIZE[0]}{IMAGE_SIZE[1]}/',
                        f'/kaggle/working/256258normed/']

    RESIZED_IMAGE_PATH = resize_image_path[i]
    try:
        os.makedirs(RESIZED_IMAGE_PATH[:-1])
    except:
        print(f'error creating {RESIZED_IMAGE_PATH[:-1]}')
    SAVE_IMAGE = True
    
    
        
        
        
'''
BASIC PARAMETERS
'''    
SEED = 42
DEBUG = False
MIXED_PRECISION = True
LOAD_SAVED_MODEL = False
DEVICE = 'cuda'


'''
MODEL PARAMETERS
'''
FOLDS = 4
EPOCHS = 10
BATCH_SIZE = 32
TARGET_SIZE = 1
NET = 'SeResNet'
MODEL_NAME = 'legacy_seresnet18'
CHANNELS = 3
MODEL_LOAD_FOR_INFER = 'auc'
DROPOUT = False

CURRENT_EPOCH = 0
OHEM_LOSS = False
OHEM_RATE = 0.7

OPTIMIZER='Adam'
SCHEDULER = 'CosineAnnealingWarmRestarts'
INIT_LEARNING_RATE = 1e-4
ETA_MIN = 1e-8
FACTOR = 0.1
PATIENCE = 2
EPS = 1e-6
T_0 = EPOCHS//4
T_MAX = 7


'''
AUGMENTATION PARAMETERS
'''
OVERSAMPLE = 3
MIXUP = True
MIXUP_APLHA = 1
INVERT_OFF_CHANNELS = True
needle_path = ['/mnt/gfs/gv1/project_sonar_data/seti/needles/', '/content/drive/MyDrive/SETI/ext_needle/', '']
NEEDLE_PATH = needle_path[i]
APPLY_NEEDLE = False
AUG = 'SwapDropFlip'

'''
INFERENCE MODE
'''
INFER = True #SET TO FALSE IF REQUIRE JUST OOF CSV


'''
OUTPUT PATH FOR LOGS, SUBMISSIONS AND MODEL OUTPUT
'''
out_path = ['/home/asajw/seti_models/', '/content/drive/MyDrive/SETI/output/',
           '/kaggle/working/']
MODEL_OUTPUT_PATH = out_path[i]


SAVED_MODEL_NAME = f'{NET}_{MODEL_NAME}_bs{BATCH_SIZE}_AllChl{IMAGE_SIZE[0]}{IMAGE_SIZE[1]}_mixup{MIXUP}_aug{AUG}_ups{OVERSAMPLE}_scd{SCHEDULER}_dropout{DROPOUT}_InvOrigNorm_epoch{EPOCHS}'

log_path = ['/home/asajw/SETI/output/', '/content/SETI/output/', '/kaggle/working/']

LOG_DIR = f'{os.path.join(log_path[i], SAVED_MODEL_NAME)}/'
try:
    os.mkdir(LOG_DIR[:-1])
except:
    print('folder exists, make sure this call is from inference.py')




