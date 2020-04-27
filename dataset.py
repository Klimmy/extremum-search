
from torch.utils.data import Dataset, DataLoader
import data_generator
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Creates PyTorch Dataset from synthetic time series and extrema as a target')
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

class SeriesDataset(Dataset):
    """
    Creates PyTorch Dataset from synthetic time series and extrema as a target
    """
    def __init__(self, N, M, T=5.7, k=3, seed=None):
        """
        Creates time series X of length N*M and extrema with constraints
        :param N: Length of one time series
        :param M: Multiplier, such that N*M = length of the whole time series
        :param k: Extremum constraint. k + 1 = minimum points between two extrema
        :param T: Multiplier for an extremum constraint. The bigger T the less extrema will be found
        :param seed: seed for time series generator
        """
        self.data = data_generator.SyntheticSeries(N, M, k)
        self.data.generate(seed)
        self.data.calculate_extrema(T)

    def __len__(self):
        return self.data.N * (self.data.M - 1) + 1

    def __getitem__(self, idx):
        sub_X = self.data.X[idx:idx + self.data.N]
        sub_YMin = self.data.YMin[idx:idx + self.data.N]
        sub_YMax = self.data.YMax[idx:idx + self.data.N]
        return sub_X, sub_YMin, sub_YMax



if __name__ == '__main__':
    import timeit

    args = parse_arguments()
    N = args.N
    M = args.M
    T = args.T
    k = args.k
    seed_val = args.seed

    dataset = SeriesDataset(N, M, T, k, seed_val)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    print('Made 100 calls for {:.2f} sec, with N={}, M={}'.format(
        timeit.timeit('next(iter(dataloader))',
                      "from __main__ import dataloader",
                      number=100),
        N,
        M))
