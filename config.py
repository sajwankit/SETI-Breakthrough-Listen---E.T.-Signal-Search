import datetime
import pytz
tz_NY = pytz.timezone('Asia/Kolkata')
datetime_NY = datetime.datetime.now(tz_NY)
DATETIME = datetime_NY.strftime('%m%d')

i = 0
input_path = ['/mnt/gfs/gv1/project_sonar_data/seti/', '/content/drive/MyDrive/SETI/input/']
DATA_PATH = input_path[i]


IMAGE_SIZE = (128,128)
resize_image_path = [f'/home/asajw/resized_images_seti/{IMAGE_SIZE[0]}{IMAGE_SIZE[1]}/',
                     f'/content/drive/MyDrive/SETI/resized_images/{IMAGE_SIZE[0]}{IMAGE_SIZE[1]}/']
RESIZED_IMAGE_PATH = resize_image_path[i]



DEVICE = 'cuda'
EPOCHS = 7
BATCH_SIZE = 32
TARGET_SIZE = 1
FOLDS = 4

MODEL_NAME = 'nfnet_l0'
CHANNELS = 1

LEARNING_RATE = 1e-4
FACTOR = 0.1
PATIENCE = 2
EPS = 1e-8


SEED = 42

out_path = ['/home/asajw/seti_models/', '/content/drive/MyDrive/SETI/output/']
OUTPUT_PATH = out_path[i]

log_path = ['/home/asajw/SETI/output/', '/content/SETI/output/']
LOG_DIR = log_path[i]
