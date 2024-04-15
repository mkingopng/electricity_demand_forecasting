"""

"""
import torch
import numpy as np
from tqdm import tqdm
from lstm_model import LSTMModel, EarlyStopping
from lstm_dataset import DemandDataset
from lstm_config import LstmCFG

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb


def train_model(model, train_loader, device, optimizer, loss_function, lr_scheduler, CFG):
    model.train()
    total_train_loss = 0
    num_train_batches = 0

    for sequences, labels in tqdm(train_loader, desc="Training Epoch"):
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        y_pred = model(sequences)
        loss = loss_function(y_pred, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        num_train_batches += 1

    lr_scheduler.step()

    avg_train_loss = total_train_loss / num_train_batches
    if CFG.logging:
        wandb.log({"training loss": avg_train_loss})

    return avg_train_loss


def test_model(model, test_loader, device, loss_function):
    model.eval()
    total_test_loss = 0
    num_test_batches = 0

    with torch.no_grad():
        for sequences, labels in tqdm(test_loader, desc="Testing Epoch"):
            sequences, labels = sequences.to(device), labels.to(device)
            y_pred = model(sequences)
            total_test_loss += loss_function(y_pred, labels).item()
            num_test_batches += 1

    avg_test_loss = total_test_loss / num_test_batches
    return avg_test_loss
