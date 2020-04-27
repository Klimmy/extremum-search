import os
import argparse
import sys
import logging
from sklearn.metrics import confusion_matrix
import numpy as np
from torch.utils.data import DataLoader

import model
import dataset


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate model_weights with confusion matrix')
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
    parser.add_argument('--threshold',
                        help='Threshold for predictor',
                        type=float,
                        nargs='?',
                        default=0.5)

    return parser.parse_args()

def show_results(text, actual, predicted, logger):
    conf_matr = confusion_matrix(actual, predicted)
    accuracy = np.sum(conf_matr.diagonal()) / np.sum(conf_matr)
    precision = conf_matr[1, 1] / np.sum(conf_matr[:, 1])
    recall = conf_matr[1, 1] / np.sum(conf_matr[1, :])
    log_text = '\nConfusion matrix for {} \n{}\nAccuracy: {:.2f} \nPrecision: {:.2f} \nRecall: {:.2f}'.format(text,
                                                                                                              conf_matr,
                                                                                                              accuracy,
                                                                                                              precision,
                                                                                                              recall)
    logger.info(log_text)


def __stdout_log(path):
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
    threshold = args.threshold
    logger = __stdout_log(os.path.dirname(__file__) + '/logs/evaluation.log')

    dataset = dataset.SeriesDataset(N, M, T, k, seed_val)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    extrem_model = model.Model(N, is_gpu=is_gpu)
    extrem_model.load(os.path.dirname(__file__) + '/model_weights/extremNet.pth')
    predictions_min_all, predictions_max_all = np.array([]), np.array([])
    labels_min_all, labels_max_all = np.array([]), np.array([])
    for i, data in enumerate(dataloader, 0):
        if i > 1000:
            break
        inputs, labels_min, labels_max = data
        predictions_min, predictions_max = extrem_model.predict_class(inputs, threshold=threshold)
        predictions_min_all = np.append(predictions_min_all, predictions_min)
        predictions_max_all = np.append(predictions_max_all, predictions_max)
        labels_min_all = np.append(labels_min_all, labels_min)
        labels_max_all = np.append(labels_max_all, labels_max)

    show_results('YMin', labels_min_all, predictions_min_all, logger)
    show_results('YMax', labels_max_all, predictions_max_all, logger)
    show_results('All', np.bitwise_or(labels_min_all.astype(int), labels_max_all.astype(int)), np.bitwise_or(predictions_min_all.astype(int), predictions_max_all.astype(int)), logger)
