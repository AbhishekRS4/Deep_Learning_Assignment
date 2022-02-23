import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import argparse
import numpy as np

from metrics import compute_accuracy
from dataset import load_gq_data, split_data

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
    def __init__(self, num_hidden_layers=1, num_neurons_hidden=10, num_neurons_input=2, num_neurons_output=1):
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
        hidden_layer_output = self.input_layer(x)

        for i in range(self.num_hidden_layers):
            hidden_layer_output = self.hidden_layer(hidden_layer_output)

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
    valid_acc = 0

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

            valid_loss += loss
            valid_acc += acc

    valid_loss /= num_batches
    valid_acc /= num_batches
    return valid_loss, valid_acc

def test_loop(model, dataset_loader, bce_loss, device):
    return validation_loop(model, dataset_loader, bce_loss, device)

def start_model_training(FLAGS):
    device = torch.device("cpu")

    mlnn_model = MultiLayerNeuralNet(num_hidden_layers=FLAGS.num_hidden_layers)
    mlnn_model.to(device)
    sgd_optimizer = torch.optim.SGD(mlnn_model.parameters(), FLAGS.learning_rate)
    bce_loss = nn.BCELoss(reduction="mean")

    gq_data = load_gq_data()
    print(gq_data[0].shape)
    print(gq_data[1].shape)

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
    for epoch in range(1, FLAGS.num_epochs+1):
        train_loss = train_loop(mlnn_model, sgd_optimizer, train_dataset_loader, bce_loss, device)
        valid_loss, valid_acc = validation_loop(mlnn_model, valid_dataset_loader, bce_loss, device)
        print(f"Epoch: {epoch} / {FLAGS.num_epochs}, train_loss: {train_loss:.5f}, valid_loss: {valid_loss:.5f}, valid_acc: {valid_acc:.5f}")

    test_loss, test_acc = test_loop(mlnn_model, test_dataset_loader, bce_loss, device)
    print(f"test_loss: {test_loss:.6f}, test_acc: {test_acc:.4f}")
    print("Model summary")
    print(summary(mlnn_model, (1, 2)))
    print("Multi layer neural network training completed")
    return

def main():
    learning_rate = 1e-3
    num_epochs = 100
    batch_size = 1
    num_hidden_layers = 2

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

    FLAGS, unparsed = parser.parse_known_args()
    start_model_training(FLAGS)

main()
