import torch
# Hyperparameters for setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 100
VALIDATION_SPLIT = 0.2
PATIENCE = 7
LAMBDA_REG = 0.003
# Pruning
REG_TYPE = "L1"
MODE = "weight-wise"
COMPRESS_RATIO = 0.2
K=100 #No. of batches
EPSILON = 0.0001
DELTA = 0.000001

CURVE_PATH = "results/curves.csv"