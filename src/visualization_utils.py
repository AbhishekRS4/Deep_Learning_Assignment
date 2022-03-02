import numpy as np
import matplotlib.pyplot as plt

def save_plot_losses(losses, file_name="losses_scratch.png"):
    fig = plt.figure(figsize=(9, 9))
    plt.plot(np.arange(1, len(losses["train"])+1), losses["train"], label="mean train loss")
    plt.plot(np.arange(1, len(losses["valid"])+1), losses["valid"], label="mean validation loss")
    plt.legend()
    plt.grid()
    plt.xlabel("Num epochs")
    plt.ylabel("Loss")
    plt.title("Visualization of the loss vs num of epochs")
    fig.savefig(file_name)
    return
