#!/usr/bin/env python3
import argparse
from matplotlib import pyplot as plt
import data_generator
plt.style.use('ggplot')



def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculates optimal T for desired period')
    parser.add_argument('--period',
                        '-p',
                        help='Desired average period between two extrema',
                        type=int,
                        nargs='?',
                        default=100)
    parser.add_argument('--start',
                        help='Start of the evaluating range for T',
                        type=int,
                        nargs='?',
                        default=2)
    parser.add_argument('--end',
                        help='End of the evaluating range for T',
                        type=int,
                        nargs='?',
                        default=10)
    parser.add_argument('--verbose',
                        '-v',
                        help='Show the plot of loss for different T',
                        dest='verbose',
                        action='store_true')
    parser.add_argument('--no-verbose',
                        '-no-v',
                        help='Don"t Show the plot of loss for different T',
                        dest='verbose',
                        action='store_false')
    parser.set_defaults(verbose=True)
    parser.add_argument('-N',
                        help='N, length of one time series',
                        type=int,
                        nargs='?',
                        default=1024)
    parser.add_argument('-M',
                        help='M, multiplier. N*M = length of the whole time series',
                        type=int,
                        nargs='?',
                        default=10)
    parser.add_argument('-k',
                        help='k, extremum constraint. k + 1 = minimum points between two extrema',
                        type=int,
                        nargs='?',
                        default=3)
    parser.add_argument('--seed',
                        '-s',
                        help='seed to generate time series',
                        type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    t_start = args.start
    t_end = args.end
    period = args.period
    verbose = args.verbose
    N = args.N
    M = args.M
    k = args.k
    seed_generate = args.seed

    data = data_generator.SyntheticSeries(N, M, k)
    data.generate(seed=seed_generate)
    optimal_T = data.calculate_optimal_T(t_start=t_start, t_end=t_end, period=period, verbose=verbose)
    print('Optimal T for chosen period is {}'.format(optimal_T))
