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
                        type=int,
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
    return parser.parse_args()


class syntheticSeries:
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
        self.D = None
        self.YMin = None
        self.YMax = None

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
        self.D = T * np.std(self.X[:-self.k] - self.X[self.k:])
        extrema = self._get_extrema()
        extrema_indexes = extrema.nonzero()[0]
        extrema_indexes = self._apply_idx_constraint(extrema_indexes)
        extrema_indexes = self._apply_value_constraint(extrema_indexes)

        # remove all redundant (after applying constraints) extrema using mask
        mask = np.zeros((self.N * self.M))
        mask.flat[extrema_indexes] = 1
        extrema = mask * extrema

        self.YMin = (extrema * (extrema == 1) * 1).astype(int)
        self.YMax = (extrema * (extrema == -1) * -1).astype(int)

    def get_extrema_quantity(self):
        return sum(self.YMin) + sum(self.YMax)

    def calculate_optimal_T(self, period=10, verbose=False):
        """
        Calculates optimal T for desired period
        :param period: Desired average period between two extrema
        :param verbose: Show the plot of loss for different T
        :return: float, optimal T between 0.0 and 3.0
        """
        needed = (self.N * self.M) // period
        iterations = 300
        loss = np.empty((iterations))
        for i in range(iterations):
            T = i / 100
            self.calculate_extrema(T)
            loss[i] = ((self.get_extrema_quantity() - needed) / needed) ** 2
        if verbose:
            plt.figure(figsize=(12, 8), dpi=80)
            plt.plot(np.array(range(iterations)) / 100, loss)
            plt.show()
        return loss.argmin() / 100

    def plot_random_subseries(self, quantity=4, mode='together'):
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
                ax.plot(range(idx, idx + self.N),sub_X, color='g')
                ax.scatter(sub_YMax + idx, sub_X[sub_YMax], color='r', label='Max', s=15)
                ax.scatter(sub_YMin + idx, sub_X[sub_YMin], color='b', label='Min', s=15)
                if position == 0:
                    ax.legend(loc="best", fontsize=15)
            plt.show()
        elif mode == 'separate':
            for position in range(quantity):
                plt.figure(num='Random subseries of X with extrema', figsize=(12, 8), dpi=80)
                idx = random.randint(0, self.N * (self.M - 1))
                sub_X = self.X[idx:idx + self.N]
                sub_YMin = self.YMin[idx:idx + self.N].nonzero()[0]
                sub_YMax = self.YMax[idx:idx + self.N].nonzero()[0]
                plt.plot(range(idx, idx + self.N), sub_X, color='g')
                plt.scatter(sub_YMax + idx, sub_X[sub_YMax], color='r', label='Max', s=20)
                plt.scatter(sub_YMin + idx, sub_X[sub_YMin], color='b', label='Min', s=20)
                plt.legend(loc="best", fontsize=15)
                plt.show()
        else:
            raise Exception('Unrecognized mode. Currently supported either "together" or "separate"')

    def _get_extrema(self):
        """
        :return: numpy array, all extrema found in time series X
        """
        extrema = np.diff(self.X)  # calculates 1st difference for time series, which is equal to 1st derivative
        extrema = np.sign(extrema)  # we need only sign of the derivative. Once sign is changed there is an extremum.
        extrema = np.ediff1d(extrema, to_begin=0, to_end=0)  # ediff1d used to find change points and to add zeros
        extrema = np.sign(extrema)  # finally -1 means Maxima, 1 means Minima
        return extrema

    def _apply_idx_constraint(self, extrema_indexes):
        """
        Applies first constraint (k) for extrema
        :param extrema_indexes: array of extrema indexes
        :return: filtered extrema
        """
        filtered = []
        for idx in extrema_indexes:
            if not filtered or idx - filtered[-1] > self.k:
                filtered.append(idx)
        return np.array(filtered)

        # this is used in case we want to apply 2nd constraint of all points in X (not only on a consecutive one)

        # def _apply_value_constraint(self, extrema_indexes):
        #     """
        #     Applies second constraint (D) for extrema
        #     :param extrema_indexes: array of extrema indexes
        #     :return: filtered extrema
        #     """
        #     intervals = []
        #     extrema_indexes_filtered = []
        #     for idx, val in enumerate(self.X[extrema_indexes]):
        #         interval_idx = bisect.bisect_left(intervals, (val,)) - 1
        #         if not intervals or interval_idx < 0 or val > intervals[interval_idx][1]:
        #             intervals.insert(interval_idx + 1, (val - self.D, val + self.D))  #  add interval
        #             extrema_indexes_filtered.append(extrema_indexes[idx])  #  add idx
        #     return extrema_indexes_filtered

    def _apply_value_constraint(self, extrema_indexes):
        """
        Applies second constraint (D) for extrema
        :param extrema_indexes: array of extrema indexes
        :return: filtered extrema
        """
        extrema_indexes_filtered = []
        prev_val = None
        for idx, val in enumerate(self.X[extrema_indexes]):
            if prev_val is None or val < prev_val - self.D or val > prev_val + self.D:
                prev_val = val
                extrema_indexes_filtered.append(extrema_indexes[idx])
        return np.array(extrema_indexes_filtered)


if __name__ == '__main__':
    args = parse_arguments()
    N = args.N
    M = args.M
    T = args.T
    k = args.k
    seed_val = args.seed
    data = syntheticSeries(N, M, k)
    data.generate(seed=seed_val)
    data.calculate_extrema(T)

    np.set_printoptions(threshold=100000)
    print('Time Series X: \n {}'.format(data.X))
    print('YMin: \n {}'.format(data.YMin))
    print('YMax: \n {}'.format(data.YMax))
