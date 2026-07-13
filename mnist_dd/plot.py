from matplotlib import pyplot as plt
import pandas as pd
from utils import num_params
from config import N_TRAIN, K, CONFIG_LABEL, LOSS_FUNC, PLOT_TITLE, ZOOMED_PLOT
from matplotlib.ticker import ScalarFormatter
import torch.nn as nn

sweep_df = pd.read_csv(f'./results/{CONFIG_LABEL}.csv')
sweep_df["params"] = num_params(sweep_df["H"])

if ZOOMED_PLOT:
    plot_df = sweep_df[3:]
    ticks = [14, 40, 100, 300, 800]
else:
    plot_df = sweep_df
    ticks = [3, 10, 40, 100, 300, 800]

plt.clf()
plt.subplot(211)
plt.title(f"{PLOT_TITLE}")

plt.semilogx(plot_df['params']/1e3, plot_df['test_zeroone']*100, marker='D', ms=4, label="Test", color="tab:blue")
plt.semilogx(plot_df['params']/1e3, plot_df['train_zeroone']*100, label="Train", color="tab:orange")
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.ticklabel_format(style='plain', axis='x')
plt.xticks(ticks)
plt.axvline([N_TRAIN*K/1e3], color='black', linestyle='--', alpha=0.5)
plt.ylabel("Zero-one loss (%)")
plt.legend()
plt.tight_layout()
plt.subplot(212)
if isinstance(LOSS_FUNC, nn.MSELoss):
    plt.semilogx(plot_df['params']/1e3, plot_df['test_MSE'], marker='D', ms=4, label="Test", color="tab:blue")
    plt.semilogx(plot_df['params']/1e3, plot_df['train_MSE'], label="Train", color="tab:orange")
    plt.ylabel("Squared loss")
elif isinstance(LOSS_FUNC, nn.CrossEntropyLoss):
    plt.semilogx(plot_df['params']/1e3, plot_df['test_CE'], marker='D', ms=4, label="Test", color="tab:blue")
    plt.semilogx(plot_df['params']/1e3, plot_df['train_CE'], label="Train", color="tab:orange")
    plt.ylabel("Cross-Entropy loss")
plt.xlabel(r"Number of parameters/weights ($\times10^3$)")
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.ticklabel_format(style='plain', axis='x')
plt.xticks(ticks, [str(t) for t in ticks])
plt.axvline([N_TRAIN*K/1e3], color='black', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(f'./plots/{CONFIG_LABEL}.jpg')
