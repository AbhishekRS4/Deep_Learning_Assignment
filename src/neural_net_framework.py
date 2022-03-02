import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import argparse
import numpy as np

from dataset import load_gq_data, split_data
from visualization_utils import save_plot_losses
from metrics import compute_test_metrics, compute_accuracy

class GQDataset(Dataset):
    def __init__(self, dataset_x, dataset_y, transform=None):
        self.transform = None
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y

    def __len__(self):
        return len(self.dataset_x)

    def __getitem__(self, idx):
        x = self.dataset_x[idx]
        y = self.dataset_y[idx]

        return x, y

class MultiLayerNeuralNet(nn.Module):
    def __init__(self, num_hidden_layers=2, num_neurons_hidden=10, num_neurons_input=2, num_neurons_output=1):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_hidden = num_neurons_hidden
        self.num_neurons_input = num_neurons_input
        self.num_neurons_output = num_neurons_output

        self.input_layer = nn.Sequential(
            (nn.Linear(self.num_neurons_input, self.num_neurons_hidden)),
            (nn.ReLU(inplace=True)),
        )

        self.hidden_layer = nn.Sequential(
            (nn.Linear(self.num_neurons_hidden, self.num_neurons_hidden)),
            (nn.ReLU(inplace=True)),
        )

        self.output_layer = nn.Sequential(
            (nn.Linear(self.num_neurons_hidden, self.num_neurons_output)),
            (nn.Sigmoid()),
        )

    def forward(self, x):
        # input + 1 hidden layer
        hidden_layer_output = self.input_layer(x)

        # hidden layers
        for i in range(self.num_hidden_layers):
            hidden_layer_output = self.hidden_layer(hidden_layer_output)

        # output layer
        predicted_probs = self.output_layer(hidden_layer_output)

        return predicted_probs

def train_loop(model, optimizer, dataset_loader, bce_loss, device):
    model.train()
    num_batches = len(dataset_loader)
    train_loss = 0

    for batch_x, batch_y in dataset_loader:
        batch_x = batch_x.to(device, dtype=torch.float)
        batch_y = batch_y.to(device, dtype=torch.float)
        optimizer.zero_grad()
        predicted_probs = model(batch_x)
        loss = bce_loss(predicted_probs, batch_y.unsqueeze(1))

        # Backpropagation
        loss.backward()
        optimizer.step()

        train_loss += loss
    train_loss /= num_batches
    return train_loss

def validation_loop(model, dataset_loader, bce_loss, device):
    model.eval()
    num_batches = len(dataset_loader)
    valid_loss = 0
    pred_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dataset_loader:
            batch_x = batch_x.to(device, dtype=torch.float)
            batch_y = batch_y.to(device, dtype=torch.float)
            predicted_probs = model(batch_x)
            predicted_label = predicted_probs > 0.5

            loss = bce_loss(predicted_probs, batch_y.unsqueeze(1))
            acc = compute_accuracy(
                predicted_label.clone().detach().cpu().numpy(),
                batch_y.unsqueeze(1).clone().detach().cpu().numpy()
            )
            pred_labels.append(predicted_label)

            valid_loss += loss

    valid_loss /= num_batches
    return valid_loss, np.array(pred_labels)

def test_loop(model, dataset_loader, bce_loss, device):
    return validation_loop(model, dataset_loader, bce_loss, device)

def start_model_training(FLAGS):
    device = torch.device("cpu")

    mlnn_model = MultiLayerNeuralNet(num_hidden_layers=FLAGS.num_hidden_layers)
    mlnn_model.to(device)
    sgd_optimizer = torch.optim.SGD(mlnn_model.parameters(), FLAGS.learning_rate)
    bce_loss = nn.BCELoss(reduction="mean")

    gq_data = load_gq_data()
    #print(gq_data[0].shape)
    #print(gq_data[1].shape)

    train_x, test_x, train_y, test_y = split_data(gq_data[0], gq_data[1], test_size=0.2)
    train_x, valid_x, train_y, valid_y = split_data(train_x, train_y, test_size=0.1)

    train_dataset = GQDataset(train_x, train_y,
        transforms.Compose([transforms.ToTensor()])
    )
    train_dataset_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True)

    valid_dataset = GQDataset(valid_x, valid_y,
        transforms.Compose([transforms.ToTensor()])
    )
    valid_dataset_loader = DataLoader(valid_dataset, batch_size=FLAGS.batch_size, shuffle=False)

    test_dataset = GQDataset(test_x, test_y,
        transforms.Compose([transforms.ToTensor()])
    )
    test_dataset_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("Multi layer neural network training is starting")
    losses = {"train":None, "valid":None}
    train_losses = []
    valid_losses = []

    for epoch in range(1, FLAGS.num_epochs+1):
        train_loss = train_loop(mlnn_model, sgd_optimizer, train_dataset_loader, bce_loss, device)
        valid_loss, valid_pred_labels = validation_loop(mlnn_model, valid_dataset_loader, bce_loss, device)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_acc = compute_accuracy(valid_y, valid_pred_labels)
        print(f"Epoch: {epoch} / {FLAGS.num_epochs}, train_loss: {train_loss:.5f}, valid_loss: {valid_loss:.5f}, valid_acc: {valid_acc:.5f}")

    losses["train"] = train_losses
    losses["valid"] = valid_losses
    if FLAGS.save_plot:
        save_plot_losses(losses, file_name="losses_pytorch.png")

    test_loss, test_pred_labels = test_loop(mlnn_model, test_dataset_loader, bce_loss, device)
    test_acc, test_cm, test_f1 = compute_test_metrics(test_y, test_pred_labels)
    print("\n---------------")
    print("Test metrics")
    print("---------------")
    print(f"test_loss: {test_loss:.6f}, test_accuracy_score: {test_acc:.4f}, test_f1_score: {test_f1:.4f}")
    print(f"test confusion matrix")
    print(test_cm)
    print("\nModel summary")
    print(summary(mlnn_model, (1, 2)))
    print("Multi layer neural network training completed")
    return

def main():
    learning_rate = 3e-3
    num_epochs = 100
    batch_size = 1
    num_hidden_layers = 2
    save_plot = 0

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--learning_rate", default=learning_rate,
        type=float, help="learning rate")
    parser.add_argument("--num_epochs", default=num_epochs,
        type=int, help="number of epochs to train")
    parser.add_argument("--batch_size", default=batch_size,
        type=int, help="number of batchs in a batch")
    parser.add_argument("--num_hidden_layers", default=num_hidden_layers,
        type=int, help="number of hidden layers in the model")
    parser.add_argument("--save_plot", default=save_plot,
        type=int, choices=[0, 1], help="flag to save plot of losses")

    FLAGS, unparsed = parser.parse_known_args()
    start_model_training(FLAGS)

main()
