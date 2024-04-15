"""

"""
import torch
from torch.utils.data import Dataset
from lstm_config import LstmCFG


class DemandDataset(Dataset):
    def __init__(self, df, label_col, sequence_length=LstmCFG.seq_length):
        """
        Initializes the dataset with the dataframe, label column, and sequence
        length
        :param df: The dataframe containing the dataset
        :param label_col: The target variable
        :param sequence_length:
        :return:
        """
        self.df = df
        self.label_col = label_col
        self.sequence_length = sequence_length

    def __len__(self):
        """
        returns the total number of samples that can be generated from the
        dataframe
        :return: the total number of samples
        """
        return len(self.df) - self.sequence_length

    def __getitem__(self, index):
        """
        generates a sample from the dataset at the specified index
        :param index: the index of the sample to generate
        :return: (tuple) a tuple containing the sequence tensor and the label
        tensor
        """
        sequence = self.df.iloc[index:index + self.sequence_length].drop(self.label_col, axis=1)
        label = self.df.iloc[index + self.sequence_length][self.label_col]
        sequence_tensor = torch.tensor(sequence.values, dtype=torch.float)
        label_tensor = torch.tensor([label], dtype=torch.float)
        return sequence_tensor, label_tensor