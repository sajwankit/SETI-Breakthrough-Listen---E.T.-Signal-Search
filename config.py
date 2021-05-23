import datetime as dt
DATETIME = dt.date.today().strftime('%m%d_%H%M')

i = 0
input_path = ['/mnt/gfs/gv1/project_sonar_data/seti/', '/content/drive/MyDrive/SETI/input/']
DATA_PATH = input_path[i]
DEVICE = 'cuda'
EPOCHS = 7
BATCH_SIZE = 32
TARGET_SIZE = 1
FOLDS = 4

MODEL_NAME = 'nfnet_l0'
CHANNELS = 6

LEARNING_RATE = 5e-4
FACTOR = 0.1
PATIENCE = 2
EPS = 1e-8


SEED = 42

out_path = ['/home/asajw/seti_models/', '/content/drive/MyDrive/SETI/output/']
OUTPUT_PATH = out_path[i]

log_path = ['/home/asajw/SETI/output/', '/content/SETI/output/']
LOG_DIR = log_path[i]
