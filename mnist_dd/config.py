import torch.nn as nn
# D = size of input (number of pixels in one MNIST image)
D = 784
# K = number of classes (number of possible digits)
K = 10
# N_TRAIN = number of training points (Belkin = 4000)
N_TRAIN = 4000
# random seed for choosing training points
SEED = 0
# learning rate (not specified by Belkin)
LR = 0.005
# momentum (Belkin = 0.95)
MOMENTUM = 0.95
# maximum number of epochs before stopping (Belkin = 6000)
MAX_EPOCHS = 6000
# how many epochs between learning rate decays (Belkin = 500)
DECAY_INTERVAL = 500
# decay factor for learning rate (Belkin = 0.9 for 10% decay)
GAMMA = 0.9
# these are guessed from the graph
# H_VALS = [4.,    5.,    8.,   13.,   18.,   23.,   25.,   28.,   30., 33.,   35.,   38.,   40.,   43.,   45.,   48.,   50.,   54., 58.,   63.,   75.,  101.,  201.,  314., 1006.]
H_VALS = [4, 6, 9, 13, 15, 25, 30, 34, 38, 41, 44, 47, 48, 49, 50, 51, 57, 63, 88, 107, 252, 314, 1006]
# H_VALS = [40, 43, 45, 48, 50, 54]
H_VALS=[4, 6]
# variance for new weights in models that are resuing weights from past (not specified by Belkin)
VAR = 0.01
# number of trials to average for each H value (Belkin = 5)
N_TRIALS = 3
# random seed list so the train set will change for each of the 5 iterations
SEEDS = range(N_TRIALS)
# how often (in epochs) to check whether we've reached 0 training error for Belkin stopping criterion
EARLY_STOP_CHECK_INTERVAL = 50
# loss function to use (Belkin = nn.MSELoss())
LOSS_FUNC = nn.MSELoss()
# mini-batch size for training, None => full batch (Belkin doesn't specify)
BATCH_SIZE = 50
# booleans for config
# always decay the learning rate even in overparameterized models (Belkin = False)
ALWAYS_DECAY = False
# always stop if train classification error hits 0 (Belkin = False)
ALWAYS_STOP = False
# reuse weights in underparameterized models (Belkin = True)
REUSE_WEIGHTS_UNDERPARAM = True
# reuse weights in overparameterized models (Belkin = False)
REUSE_WEIGHTS_OVERPARAM = False
# Label for file name
CONFIG_LABEL = f"belkin_batchsize{BATCH_SIZE}_lr{LR}_test"
# Plot title
PLOT_TITLE = f"LR = {LR}, Batch size = {BATCH_SIZE}, Belkin Setup"
# True = zoomed-in plot (skip first 3 H values, tick marks starting at 14)
# False = full plot (all H values, tick marks starting at 3)
ZOOMED_PLOT = False