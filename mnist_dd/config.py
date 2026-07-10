# D = size of input (number of pixels in one MNIST image)
D = 784
# K = number of classes (number of possible digits)
K = 10
# N_TRAIN = number of training points (from Belkin)
N_TRAIN = 4000
# random seed for choosing training points
SEED = 0
# learning rate (not specified by Belkin)
LR = 0.05
# momentum (from Belkin)
MOMENTUM = 0.95
# maximum number of epochs before stopping (from Belkin)
MAX_EPOCHS = 6000
# how many epochs between learning rate decays (from Belkin)
DECAY_INTERVAL = 500
# decay factor for learning rate (from Belkin)
GAMMA = 0.9