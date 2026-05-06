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
WEIGHT_PRUNING_THRESHOLD = 1e-4 # should be a function of EPSILON
#KERNEL_PRUNING_THRESHOLD = 0.00001* kernel_size # should be a function of EPSILON
#CHANNEL_PRUNING_THRESHOLD = 0.00001 *channel_size# should be a function of EPSILON


UPDATE_PER_EPOCH = 5

EPSILON = 1e-5 # when updating penalties
DELTA = 0.000001 # when calculating (why necessary under sqrt)

CURVE_PATH = "results/"

# warm-up and recovery phase lengths
A = 100 # number of epochs until convergence (with reg)
B = 3 # number of epochs
C = 2# number of epochs