import torch
# Hyperparameters for setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.2
PATIENCE = 10
LAMBDA_REG = 0.003

# Pruning
REG_TYPE = "L1"
MODE = "weigth-wise"
COMPRESS_RATIO = 0.3
UPDATE_INTERVAL=4 #No. of batches
EPSILON = 0.0001