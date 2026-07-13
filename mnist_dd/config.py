import torch.nn as nn
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
# these are guessed from the graph
H_VALS = [4.,    5.,    8.,   13.,   18.,   23.,   25.,   28.,   30.,
         33.,   35.,   38.,   40.,   43.,   45.,   48.,   50.,   54.,
         58.,   63.,   75.,  101.,  201.,  314., 1006.]
# variance for new weights in models that are resuing weights from past (not specified by Belkin)
VAR = 0.01
# number of trials to average for each H value (from Belkin)
N_TRIALS = 5
# random seed list so the train set will change for each of the 5 iterations
SEEDS = range(N_TRIALS)
# how often (in epochs) to check whether we've reached 0 training error for Belkin stopping criterion
EARLY_STOP_CHECK_INTERVAL = 25
# loss function to use (Belkin = nn.MSELoss())
LOSS_FUNC = nn.MSELoss()
# booleans for config
# always decay the learning rate even in overparameterized models (Belkin = False)
ALWAYS_DECAY = False
# always stop if train classification error hits 0 (Belkin = False)
ALWAYS_STOP = False
# reuse weights in underparameterized models (Belkin = True)
REUSE_WEIGHTS_UNDERPARAM = True
# reuse weights in overparameterized models (Belkin = False)
REUSE_WEIGHTS_OVERPARAM = True
# Label for file name
CONFIG_LABEL = "mse_always_reuse_always_decay_always_stop"
# Plot title
PLOT_TITLE = "MSE Loss, Always Reuse Weights, Conditional LR Decay and Stopping"
# True = zoomed-in plot (skip first 3 H values, tick marks starting at 14)
# False = full plot (all H values, tick marks starting at 3)
ZOOMED_PLOT = True