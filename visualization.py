#!/usr/bin/env python3
import argparse
import os
from matplotlib import pyplot as plt
import data_generator
plt.style.use('ggplot')

import model


def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot synthetic time series with extrema')
    parser.add_argument('--quantity',
                        '-q',
                        help='Quantity of random subseries to plot',
                        type=int,
                        nargs='?',
                        default=4)
    parser.add_argument('--mode',
                        '-m',
                        help='Mode how plots will be drawn. "together" on one figure; "separate" on separate',
                        type=str,
                        nargs='?',
                        default='together')
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
                        help='seed to generate time series',
                        type=int)
    parser.add_argument('--prediction',
                        help='add predictions to plots',
                        dest='prediction',
                        action='store_true')
    parser.add_argument('--no-prediction',
                        help='do not add predictions to plots',
                        dest='prediction',
                        action='store_false')
    parser.set_defaults(prediction=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    quantity = args.quantity
    mode = args.mode
    N = args.N
    M = args.M
    T = args.T
    k = args.k
    seed_generate = args.seed
    make_prediction = args.prediction


    data = data_generator.SyntheticSeries(N, M, k)
    data.generate(seed=seed_generate)
    data.calculate_extrema(T)
    if make_prediction:
        extrem_model = model.Model(N)
        extrem_model.load(os.path.dirname(__file__) + '/model_weights/extremNet.pth')
    else:
        extrem_model = None

    data.plot_random_subseries(quantity=quantity, mode=mode, prediction_model=extrem_model)
