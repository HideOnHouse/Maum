import torch
import os
from torch.utils.data import Dataset, DataLoader


class VoiceDataset(Dataset):
    """
    Generate voice dataset via window size
    """

    def __init__(self, window_size, stride=1):
        self.window_size = window_size
        self.stride = stride

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


class MaumClassifier(torch.nn.Module):
    def __init__(self):
        super(MaumClassifier, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d()
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(-1, 4)
        )

    def forward(self):
        pass


def main():
    pass


if __name__ == '__main__':
    main()
