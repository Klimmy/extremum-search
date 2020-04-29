from datetime import datetime
import os
import sys
import argparse
import logging
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import dataset


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train PyTorch model_weights to predict extrema')
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
                        default=5.2)
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
                        help='Use GPU to train the model_weights',
                        dest='gpu',
                        action='store_true')
    parser.add_argument('--no-gpu',
                        help='Use CPU to train the model_weights',
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


class ExtremNet(nn.Module):
    """
    Neural net architecture to find extrema in time series
    Consists of two convolutional layers with ReLU and pooling and two linear layers with Batch Normalization between
    """
    def __init__(self, n_classes, device):
        """
        :param device: CPU or GPU, either torch.device('cuda') or torch.device('cpu')
        """
        super(ExtremNet, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv1d(1, 4, 10).to(device=device)
        self.conv2 = nn.Conv1d(4, 16, 10).to(device=device)
        self.conv3 = nn.Conv1d(16, 32, 10).to(device=device)
        self.pool = nn.MaxPool1d(2).to(device=device)
        self.fc1 = nn.Linear((self.n_classes - 64) * 4, self.n_classes).to(device=device)
        self.fc2 = nn.Linear(self.n_classes, self.n_classes).to(device=device)
        self.bn = nn.BatchNorm1d(self.n_classes).to(device=device)

    def forward(self, x):
        x = x - F.pad(x, (1, 0))[:, :-1]  # transform time series to 1st difference
        x = x.view(-1, 1, self.n_classes)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, (self.n_classes - 64) * 4)
        x = torch.relu(self.fc1(x))
        x = self.bn(x)
        x = self.fc2(x)
        return x


class Model:
    """
    Neural net model to find extrema in time series with training, evaluating, saving, predicting capabilities
    """
    def __init__(self, n_classes, learning_rate=0.001, is_gpu=False):
        self.n_classes = n_classes
        if is_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.logger = None
        self.learning_rate = learning_rate
        self.model = ExtremNet(self.n_classes, self.device).double()
        self.criterion = nn.BCEWithLogitsLoss()  # As we are solving multi-label task this loss function is appropriate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self, n_epochs, trainloader, testloader=None, logging_loss=False):
        self.logger = self.__stdout_log(os.path.dirname(__file__) + '/logs/training.log')
        self.logger.info('{} will be used'.format(self.device))
        for epoch in range(n_epochs):
            self.model.train()
            start_time = datetime.now()
            for i, data in enumerate(trainloader, 0):
                inputs, labels, _ = data
                inputs = inputs.to(device=self.device)
                labels = labels.to(device=self.device)
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                labels = labels.type_as(outputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            if logging_loss:
                self.model.eval()
                running_loss_train = self.evaluate(trainloader)
                if testloader is None:
                    self.logger.info('Test data set was not chosen. Only train loss will be calculated')
                    running_loss_test = 0
                else:
                    running_loss_test = self.evaluate(testloader)

                log_message = '{} epoch, loss_train: {:.4f}, loss_test: {:.4f}, time: {}'.format(
                    epoch + 1,
                    running_loss_train / len(trainloader),
                    running_loss_test / len(testloader),
                    datetime.now() - start_time)
                self.logger.info(log_message)

        self.model.eval()
        self.logger.info('Model trained')

    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                inputs, labels, _ = data
                inputs = inputs.to(device=self.device)
                labels = labels.to(device=self.device)
                outputs = self.model(inputs)
                labels = labels.type_as(outputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
        return running_loss

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model = ExtremNet(self.n_classes, self.device).double()
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def predict_class(self, inputs, threshold):
        if not torch.is_tensor(inputs):
            inputs = torch.from_numpy(inputs).view(1, self.n_classes)
        inputs = inputs.to(device=self.device)
        YMin = self._predict(inputs, is_proba=False, threshold=threshold).detach().numpy()[0]
        YMax = self._predict(-inputs, is_proba=False, threshold=threshold).detach().numpy()[0]
        return YMin, YMax

    def predict_proba(self, inputs):
        if not torch.is_tensor(inputs):
            inputs = torch.from_numpy(inputs).view(1, self.n_classes)
        inputs = inputs.to(device=self.device)
        YMin = self._predict(inputs, is_proba=True).detach().numpy()[0]
        YMax = self._predict(-inputs, is_proba=True).detach().numpy()[0]
        return YMin, YMax

    def _predict(self, inputs, is_proba=True, threshold=0.5):
        predictions = self.model(inputs)
        predictions = torch.sigmoid(predictions).to(device=self.device)
        if not is_proba:
            predictions = torch.where(predictions > threshold,
                                      torch.ones(inputs.size()[1]),
                                      torch.zeros(inputs.size()[1])
                                      )
        return predictions

    def __stdout_log(self, path):
        log = logging.getLogger()
        log.setLevel(logging.DEBUG)

        fh = logging.FileHandler(path)
        fh.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        fh.setFormatter(formatter)
        log.addHandler(handler)
        log.addHandler(fh)
        return log


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

    dataset = dataset.SeriesDataset(N, M, T, k, seed_val)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size

    trainset, testset = random_split(dataset, [train_size, test_size])
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
    testloader = DataLoader(testset, batch_size=16, shuffle=True)

    extrem_model = Model(N, learning_rate, is_gpu)
    extrem_model.train(n_epochs, trainloader, testloader, logging_loss)
    extrem_model.save(os.path.dirname(__file__) + '/model_weights/extremNet.pth')
