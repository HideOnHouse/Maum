import os
import csv

import torch
import torchvision
from torch.utils.data import dataset

from tqdm import tqdm


class CustomDataset(dataset):
    def __init__(self):
        super(CustomDataset, self).__init__()
        self.x = []
        self.t = []
        with open("data_vector.csv" 'r') as f:
            data = f.readlines()
            for row in data:
                self.x.append(row[0])
                self.t.append(row[1])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.t[item]


class MaumRnnModel(torch.nn.RNN):
    def __init__(self):
        super(MaumRnnModel, self).__init__(input_size=7, hidden_size=8)
        pass


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # hyper parameter
    learning_rate = 0.001
    training_epoch = 20
    batch_size = 13

    model = MaumRnnModel().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("Learning Start")
    train_set = CustomDataset()
    train_data = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
    model.train()
    for epoch in range(training_epoch):
        print("Epoch %s start" % (epoch + 1))
        avg_cost = 0
        for data in tqdm(train_data):
            features, labels = data
            features = features.to(device)
            labels = labels.to(device)
            prediction = model(features)

            optimizer.zero_grad()
            cost = criterion(prediction, labels)
            cost.backward()
            optimizer.step()

            avg_cost += cost / len(train_data)
        print('cost :', float(avg_cost))
    print("Leaning Finish")

    torch.save(model, r'Project_directory\result\model.pth')
    model = torch.load(r'Project_directory\result\model.pth')

    print("Test Start")
    test_set = CustomDataset()
    test_data = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=True)

    model.eval()
    with torch.no_grad():
        for data in tqdm(test_data):
            features, labels = data
            features = features.to(device)
            labels = labels.to(device)
            prediction = model(features)
            print(labels == prediction)


if __name__ == '__main__':
    main()
