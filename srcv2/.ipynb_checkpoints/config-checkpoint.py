
import os
i = 0

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
INPUT PATHS
'''
input_path = ['/mnt/gfs/gv1/project_sonar_data/seti/', '/content/drive/MyDrive/SETI/input/', '/kaggle/input/256258normed/']
DATA_PATH = input_path[i]


ORIG_IMAGE_SIZE = (256,273) # (freq, time): aligning with cv2, not to confuse with np.array shape
RESIZED_IMAGE_SIZE = (256, 258)
NORM_ORIG_IMAGE_SIZE = (256,273)

IMAGE_SIZE = NORM_ORIG_IMAGE_SIZE
ORIG_IMAGE = False
IMAGE_TYPE = 'norm'

norm_image_path = [f'/mnt/gfs/gv1/project_sonar_data/seti/normalized_images_seti/{IMAGE_SIZE[0]}{IMAGE_SIZE[1]}/',
                         f'/content/drive/MyDrive/SETI/normalized_images_seti/{IMAGE_SIZE[0]}{IMAGE_SIZE[1]}/',
                        f'/kaggle/working/256258normed/']

resize_image_path = [f'/mnt/gfs/gv1/project_sonar_data/seti/resized_images_seti/{IMAGE_SIZE[0]}{IMAGE_SIZE[1]}/',
                         f'/content/drive/MyDrive/SETI/resized_images/{IMAGE_SIZE[0]}{IMAGE_SIZE[1]}/',
                        f'/kaggle/working/256258normed/']

NORM_IMAGE_PATH = norm_image_path[i]
RESIZED_IMAGE_PATH = resize_image_path[i]


SAVE_IMAGE = True


    
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''    
        
        
        
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
FOLDS = 1
EPOCHS = 150
BATCH_SIZE = 32
TARGET_SIZE = 1
NET = 'SeResNet'
MODEL_NAME = 'legacy_seresnet18'
CHANNELS = 1
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
T_0 = EPOCHS//10
T_MAX = 7


'''
AUGMENTATION PARAMETERS
'''
OVERSAMPLE = 0
MIXUP = False
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


SAVED_MODEL_NAME = f'{NET}_{MODEL_NAME}_bs{BATCH_SIZE}_Chl0_{IMAGE_SIZE[0]}{IMAGE_SIZE[1]}_mixup{MIXUP}_aug{AUG}_ups{OVERSAMPLE}_scd{SCHEDULER}_dropout{DROPOUT}_InvOrigNorm_epoch{EPOCHS}'

log_path = ['/home/asajw/SETI/output/', '/content/SETI/output/', '/kaggle/working/']

LOG_DIR = f'{os.path.join(log_path[i], SAVED_MODEL_NAME)}/'
try:
    os.mkdir(LOG_DIR[:-1])
except:
    print('folder exists, make sure this call is from inference.py')





