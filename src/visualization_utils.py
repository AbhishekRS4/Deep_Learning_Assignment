import numpy as np
import matplotlib.pyplot as plt

def save_plot_losses(losses, file_name="losses_scratch.png"):
    fig = plt.figure(figsize=(9, 9))
    plt.plot(np.arange(1, len(losses["train"])+1), losses["train"], label="mean train loss")
    plt.plot(np.arange(1, len(losses["valid"])+1), losses["valid"], label="mean validation loss")
    plt.legend(fontsize=24)
    plt.grid()
    plt.xlabel("Num epochs", fontsize=24)
    plt.ylabel("Loss", fontsize=24)
    plt.title("Visualization of the loss vs num of epochs", fontsize=24)
    fig.savefig(file_name)
    return
