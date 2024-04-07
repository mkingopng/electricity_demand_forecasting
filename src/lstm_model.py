"""
- scaling is important. do you scale both training and target?
- look ahead is the forecast horizon
- look back: how many time steps to look back
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


class DemandDataset(Dataset):
    def __init__(self, sequences, labels, sequence_length=100):
        self.sequences = sequences
        self.labels = labels
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        end = min(index + self.sequence_length, len(self.sequences) - 1)
        sequence = self.sequences[index:end]
        label = self.labels[end]
        return torch.tensor(sequence, dtype=torch.float), torch.tensor(label, dtype=torch.float)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # hyperparameters
    input_size = 1  # the number of input features in dataset
    hidden_layer_size = 50
    output_size = 1
    learning_rate = 0.001
    batch_size = 1
    epochs = 10

    # dataset & dataLoader
    train_dataset = DemandDataset(
        dataset_sequences,
        dataset_labels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # model
    model = LSTMModel(input_size, hidden_layer_size, output_size)
    model = model.to(device)

    # loss
    loss_function = nn.L1Loss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    for epoch in range(epochs):
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            y_pred = model(sequences)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            print(f'epoch: {epoch:3} loss: {single_loss.item():10.8f}')
