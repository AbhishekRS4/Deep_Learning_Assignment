import numpy as np
import matplotlib.pyplot as plt

def save_plot_losses(losses, file_name="losses_scratch.png", fontsize=24):
    fig = plt.figure(figsize=(9, 9))
    plt.plot(np.arange(1, len(losses["train"])+1), losses["train"], label="mean train loss")
    plt.plot(np.arange(1, len(losses["valid"])+1), losses["valid"], label="mean validation loss")
    plt.legend(fontsize=fontsize)
    plt.grid()
    plt.xlabel("Num epochs", fontsize=fontsize)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title("Visualization of the loss vs num of epochs", fontsize=fontsize)
    fig.savefig(file_name)
    return
