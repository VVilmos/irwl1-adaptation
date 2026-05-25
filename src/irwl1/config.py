import torch
# Hyperparameters for setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 100
VALIDATION_SPLIT = 0.2
PATIENCE = 7
LAMBDA_REG = 0.0001
WEIGHT_DECAY = 1e-4

# Pruning
REG_TYPE = "L1"
MODE = "weight-wise"
WEIGHT_PRUNING_THRESHOLD = 1e-3 # should be a function of EPSILON
KERNEL_PRUNING_THRESHOLD = 1e-03 / 9 
CHANNEL_PRUNING_THRESHOLD = 1e-03 / 9 / 16


UPDATE_PER_EPOCH = 5

EPSILON_START = 1e8
EPSILON_END = 1e-6
EPSILON_DECAY_STEPS = 50
EPSILON_SIGMOID_CENTER = 0.6
EPSILON_SIGMOID_STEEPNESS = 10
EPSILON_SPIKE_INDICATOR = 1e1
EPSILON_EARLY_STOP_FLOOR = 1e-6
THRESHOLDED_SPARSITY_MIN_DELTA = 0.5
EPSILON = EPSILON_START # when updating penalties
DELTA = 0.000001 # when calculating (why necessary under sqrt)

CURVE_PATH = "results/"

# warm-up and recovery phase lengths
NUM_REG_EPOCHS = 100 # number of epochs until convergence (with reg)
NUM_RECOVERY_EPOCHS = 10 # number of epochs
NUM_PRETRAIN_EPOCHS = 10# number of epochs


# admin
MODEL = "ResNet20"
WANDB_MODE = "online"