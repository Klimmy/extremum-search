#!/usr/bin/env python3
import numpy as np
import random
import argparse
# import bisect
from matplotlib import pyplot as plt

plt.style.use('ggplot')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate synthetic time series and find extrema with constraints')
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
    return parser.parse_args()


class SyntheticSeries:
    """
    Generates a synthetic time series X of predefined length N * M
    Calculates all extrema (YMin, YMax) of this time series with the following constraints:
        - There should be more than k points between two extrema
        - Absolute value difference between two consecutive extrema should be more than D = T * np.std(Х[:-k] – Х[k:])
        - X shouldn't have any additional extrema with these constraints between already found extrema
    """

    def __init__(self, N, M, k=3):
        """
        :param N: Length of one time series
        :param M: Multiplier, such that N*M = length of the whole time series
        :param k: First constraint (see main description)
        """
        self.N = N
        self.M = M
        self.k = k
        self.X = np.empty((self.N * self.M))
        self.length = (self.N * self.M) - 1
        self.D = None
        self.YMin = np.array([0 for i in range(self.length + 1)])
        self.YMax = np.array([0 for i in range(self.length + 1)])

    def generate(self, seed=None):
        """
        Generates time series X
        :param seed: integer to reproduction
        """
        self.X[0] = 1
        if seed is not None:
            random.seed(seed)
        for i in range(1, self.N * self.M):
            self.X[i] = self.X[i - 1] + random.uniform(-1, 1)

    def calculate_extrema(self, T):
        """
        Calculates extrema with described constraints for generated time series X
        :param T: Multiplier for the second extremum constraint. The bigger T the less extrema will be found
        """
        self.YMin = np.array([0 for i in range(self.length + 1)])
        self.YMax = np.array([0 for i in range(self.length + 1)])
        self.D = T * np.std(self.X[:-self.k] - self.X[self.k:])

        next_cand_idx = self._first_ext()  # find first extremum (max or min) and return next candidate index
        if next_cand_idx < self.length - self.k and self.YMax.nonzero()[0]:
            next_cand_idx = self._next_min_ext(next_cand_idx)
        max_turn = True
        while next_cand_idx < self.length - self.k:  # iterate through time series and find max/min one by one
            if max_turn:
                next_cand_idx = self._next_max_ext(next_cand_idx)
                max_turn = False
            else:
                next_cand_idx = self._next_min_ext(next_cand_idx)
                max_turn = True

    def _first_ext(self):
        """
        Find first appropriate extremum and update YMin/YMax accordingly
        :return (int): next extremum candidate index
        """
        idx = 0
        min_idx = 0
        max_idx = 0
        while idx < self.length - self.k \
                and (abs(self.X[idx + 1] - self.X[max_idx]) < self.D or idx <= max_idx + self.k) \
                and (abs(self.X[idx + 1] - self.X[min_idx]) < self.D or idx <= min_idx + self.k):
            idx += 1
            if self.X[min_idx] > self.X[idx]:
                min_idx = idx
            if self.X[max_idx] < self.X[idx]:
                max_idx = idx
        idx += 1
        if idx < self.length - self.k and self.X[idx] > self.X[0]:
            self.YMin[min_idx] = 1
        elif idx < self.length - self.k and self.X[idx] < self.X[0]:
            self.YMax[max_idx] = 1
        return idx

    def _next_min_ext(self, idx):
        """
        Find next appropriate minimum and update YMin accordingly
        :return (int): next maximum candidate index
        """
        min_idx = idx
        while idx < self.length - self.k \
                and (self.X[idx + 1] < self.X[min_idx]
                     or idx <= min_idx + self.k
                     or abs(self.X[idx + 1] - self.X[min_idx]) < self.D):
            idx += 1
            if self.X[min_idx] > self.X[idx]:
                min_idx = idx
        self.YMin[min_idx] = 1
        return idx + 1

    def _next_max_ext(self, idx):
        """
        Find next appropriate maximum and update YMax accordingly
        :return (int): next minimum candidate index
        """
        max_idx = idx
        while idx < self.length - self.k \
                and (self.X[idx + 1] > self.X[max_idx]
                     or idx <= max_idx + self.k
                     or abs(self.X[idx + 1] - self.X[max_idx]) < self.D):
            idx += 1
            if self.X[max_idx] < self.X[idx]:
                max_idx = idx
        self.YMax[max_idx] = 1
        return idx + 1

    def get_extrema_quantity(self):
        return sum(self.YMin) + sum(self.YMax)

    def calculate_optimal_T(self, period=100, t_start=2, t_end=10, verbose=False):
        """
        Calculates optimal T for desired period
        :param period: Desired average period between two extrema
        :param t_start: Start of the evaluating range for T
        :param t_end: End of the evaluating range for T
        :param verbose: Show the plot of loss for different T
        :return: float, optimal T between 0.0 and 3.0
        """
        needed = (self.N * self.M) // period
        iterations = (t_end - t_start) * 10
        loss = np.empty(iterations)
        for i in range(iterations):
            T = t_start + (i / 10)
            self.calculate_extrema(T)
            loss[i] = ((self.get_extrema_quantity() - needed) / needed) ** 2
        if verbose:
            plt.figure(figsize=(12, 8), dpi=80)
            plt.plot(np.array(range(iterations)) / 10 + t_start, loss)
            plt.show()
        return loss.argmin() / 10 + t_start

    def plot_random_subseries(self, quantity=4, mode='together', prediction_model=None):
        """
        Draw several plots (equals to quantity) of generated subseries from series X
        :param quantity: Amount of subseries will be fetched
        :param mode: Mode how plots will be drawn. "together" on one figure; "separate" on separate
        :param seed: integer to reproduction
        """
        if self.X.size == 0:
            raise Exception('You need to generate data first. Check "generate" function')
        if quantity <= 0 or not isinstance(quantity, int):
            raise Exception('quantity have to be positive integer')
        if mode == 'together':
            fig = plt.figure(num='Time series X with random subseries', constrained_layout=True, figsize=(12, 8), dpi=80)
            grid = fig.add_gridspec((quantity + 1) // 2 + 1, 2)
            ax = fig.add_subplot(grid[0, :], title='All time series X. Length {}'.format(int(self.N * self.M)))
            ax.plot(self.X, color='g')
            for position in range(quantity):
                idx = random.randint(0, self.N * (self.M - 1))
                sub_X = self.X[idx:idx + self.N]
                sub_YMin = self.YMin[idx:idx + self.N].nonzero()[0]
                sub_YMax = self.YMax[idx:idx + self.N].nonzero()[0]
                ax = fig.add_subplot(grid[position // 2 + 1, position % 2])
                ax.plot(range(idx, idx + self.N),sub_X, color='g', zorder=0)
                ax.scatter(sub_YMax + idx, sub_X[sub_YMax], color='r', label='Max', s=50, zorder=1)
                ax.scatter(sub_YMin + idx, sub_X[sub_YMin], color='b', label='Min', s=50, zorder=1)
                if position == 0:
                    ax.legend(loc="best", fontsize=15)
            plt.show()
        elif mode == 'separate':
            for position in range(quantity):

                fig = plt.figure(num='Random subseries of X with extrema', figsize=(12, 8), dpi=80)
                if prediction_model is not None:
                    ax_grid = 111
                else:
                    ax_grid = 111
                ax1 = fig.add_subplot(ax_grid)
                idx = random.randint(0, self.N * (self.M - 1))
                sub_X = self.X[idx:idx + self.N]
                sub_YMin = self.YMin[idx:idx + self.N].nonzero()[0]
                sub_YMax = self.YMax[idx:idx + self.N].nonzero()[0]
                ax1.plot(range(idx, idx + self.N), sub_X, color='g', zorder=0)
                ax1.scatter(sub_YMax + idx, sub_X[sub_YMax], color='r', label='Max', s=60, zorder=1)
                ax1.scatter(sub_YMin + idx, sub_X[sub_YMin], color='b', label='Min', s=60, zorder=1)
                if prediction_model is not None:
                    ax2 = ax1.twinx()
                    sub_YMin_prediction, sub_YMax_prediction = prediction_model.predict_proba(sub_X)
                    ax2.plot(range(idx, idx + self.N), sub_YMax_prediction, color='r')
                    ax2.plot(range(idx, idx + self.N), sub_YMin_prediction, color='b')
                ax1.legend(loc="best", fontsize=15)
                plt.show()
        else:
            raise Exception('Unrecognized mode. Currently supported either "together" or "separate"')


if __name__ == '__main__':
    args = parse_arguments()
    N = args.N
    M = args.M
    T = args.T
    k = args.k
    seed_val = args.seed

    data = SyntheticSeries(N, M, k)
    data.generate(seed=seed_val)
    data.calculate_extrema(T)

    np.set_printoptions(threshold=100000)
    print('Time Series X: \n {}'.format(data.X))
    print('YMin: \n {}'.format(data.YMin))
    print('YMax: \n {}'.format(data.YMax))
