from datetime import datetime
import sys
import argparse
import logging
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_dataset


def __stdout_log():
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)

    fh = logging.FileHandler('logs/training.log')
    fh.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    fh.setFormatter(formatter)
    log.addHandler(handler)
    log.addHandler(fh)
    return log


logger = __stdout_log()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train PyTorch model to predict extrema')
    parser.add_argument('-N',
                        help='N, length of one time series',
                        type=int,
                        nargs='?',
                        default=1024)
    parser.add_argument('-M',
                        help='M, multiplier. N*M = length of the whole time series',
                        type=int,
                        nargs='?',
                        default=1000)
    parser.add_argument('-T',
                        help='T, multiplier for extremum constraint. The bigger T the less extrema will be found',
                        type=float,
                        nargs='?',
                        default=1.1)
    parser.add_argument('-k',
                        help='k, extremum constraint. k + 1 = minimum points between two extrema',
                        type=int,
                        nargs='?',
                        default=3)
    parser.add_argument('--seed',
                        '-s',
                        help='seed for reproductivity',
                        type=int)
    parser.add_argument('--gpu',
                        help='Use GPU to train the model',
                        dest='gpu',
                        action='store_true')
    parser.add_argument('--no-gpu',
                        help='Use CPU to train the model',
                        dest='gpu',
                        action='store_false')
    parser.set_defaults(gpu=False)
    parser.add_argument('--epochs',
                        '-e',
                        help='number of epochs in training',
                        type=int,
                        default=10)
    parser.add_argument('--learning-rate',
                        '-lr',
                        help='learning rate for optimizer',
                        type=float,
                        default=0.001)
    parser.add_argument('--logging',
                        help='log training process',
                        dest='logging',
                        action='store_true')
    parser.add_argument('--no-logging',
                        help='Does not log training process',
                        dest='logging',
                        action='store_false')
    parser.set_defaults(logging=True)
    return parser.parse_args()


class extremNet(nn.Module):
    def __init__(self, device):
        super(extremNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 2, 5).to(device=device)
        self.conv2 = nn.Conv1d(2, 4, 5).to(device=device)
        self.pool = nn.MaxPool1d(2).to(device=device)
        self.fc1 = nn.Linear(1012, 1024).to(device=device)
        self.fc2 = nn.Linear(1024, 1024).to(device=device)
        self.bn = nn.BatchNorm1d(1024).to(device=device)

    def forward(self, x):
        x = x - F.pad(x, (1, 0))[:, :-1]
        x = x.view(-1, 1, 1024)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 1012)
        x = torch.relu(self.fc1(x))
        x = self.bn(x)
        x = self.fc2(x)
        return x


def model_train(model, criterion, optimizer, n_epochs, device, trainloader, testloader, logging_loss):
    for epoch in range(n_epochs):
        model.train()
        start_time = datetime.now()
        for i, data in enumerate(trainloader, 0):
            inputs, labels, _ = data
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)
            optimizer.zero_grad()

            outputs = model(inputs)
            labels = labels.type_as(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if logging_loss:
            model.eval()
            running_loss_train = evaluate(model, trainloader, device, criterion)
            running_loss_test = evaluate(model, testloader, device, criterion)

            log_message = '{} epoch, loss_train: {:.4f}, loss_test: {:.4f},   time: {}'.format(
                epoch + 1,
                running_loss_train / len(trainloader),
                running_loss_test / len(testloader),
                datetime.now() - start_time)
            logger.info(log_message)

    model.eval()
    return model


def evaluate(model, dataloader, device, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            inputs, labels, _ = data
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(inputs)
            labels = labels.type_as(outputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss


if __name__ == '__main__':

    args = parse_arguments()
    N = args.N
    M = args.M
    T = args.T
    k = args.k
    seed_val = args.seed
    is_gpu = args.gpu
    n_epochs = args.epochs
    learning_rate = args.learning_rate
    logging_loss = args.logging

    if is_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dataset = pytorch_dataset.SeriesDataset(N, M, T, k, seed_val)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size

    trainset, testset = random_split(dataset, [train_size, test_size])
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
    testloader = DataLoader(testset, batch_size=16, shuffle=True)

    model = extremNet(device).double()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model_train(model, criterion, optimizer, n_epochs, device, trainloader, testloader, logging_loss)
    logger.info('Model trained')
    torch.save(model.state_dict(), 'model/extremNet.pth')
