import os
import csv

import torch
import torchvision


class MaumRnnModel(torch.nn.Module):
    def __init__(self):
        super(MaumRnnModel, self).__init__()
        
        self.layer1 = torch.nn.Sequential(

        )

        self.layer2 = torch.nn.Sequential(

        )

        self.layer3 = torch.nn.Sequential(

        )

        self.layer4 = torch.nn.Sequential(

        )

        self.layer5 = torch.nn.Sequential(

        )

        self.layer6 = torch.nn.Sequential(

        )

        self.classifier = torch.nn.Sequential(

        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        out = self.classifier(x)
        return out


def main():
    pass


if __name__ == '__main__':
    main()
