from torch.optim import lr_scheduler

DATA_PATH = '/content/drive/MyDrive/SETI/input/'
DEVICE = 'cuda'
EPOCHS = 3
BATCH_SIZE = 32
TARGET_SIZE = 1

MODEL_NAME = 'nfnet_l0'
CHANNELS = 6

LEARNING_RATE = 5e-4
FACTOR = 0.1
PATIENCE = 2
EPS = 1e-8
config.SCHEDULER = {'ReduceLROnPlateau': ReduceLROnPlateau(optimizer, mode='min',
                                                            factor=config.FACTOR, patience=config.PATIENCE,
                                                            verbose=True, eps=config.EPS)}


LOG_DIR = '/content/SETI/'
SEED = 42

OUTPUT_PATH = '/content/drive/MyDrive/SETI/output/'
