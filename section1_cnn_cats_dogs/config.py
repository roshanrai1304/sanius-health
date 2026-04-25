import os
import torch

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cifar10")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

IMAGE_SIZE = 32
BATCH_SIZE = 64
NUM_WORKERS = 0
LEARNING_RATE = 0.001
NUM_EPOCHS_BASELINE = 50
NUM_EPOCHS_IMPROVED = 50
WEIGHT_DECAY = 1e-4
PATIENCE = 7
RANDOM_SEED = 42
TRAIN_RATIO = 0.8

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
