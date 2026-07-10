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
# STD for new weights in models that are resuing weights from past (not specified by Belkin)
STD = 0.1
# number of trials to average for each H value (from Belkin)
N_TRIALS = 5
# random seed list so the train set will change for each of the 5 iterations
SEEDS = range(N_TRIALS)