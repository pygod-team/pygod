"""CLI Parameters Handler (under development)"""
import argparse


def main():
    parser = argparse.ArgumentParser(description='PyGOD')
    parser.add_argument('model', type=str,
                        help='detector model')
    parser.add_argument('-d', '--dataset', type=str, default='Cora',
                        help='graph dataset to be evaluated, [Cora, Pubmed, '
                             'Citeseer] ')
    parser.add_argument('--hid_dim', type=int, default=64,
                        help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='number of hidden layers,'
                             'must be greater than 2 (default: 4)')
    parser.add_argument('--epoch', type=int, default=5,
                        help='maximum training epoch')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='balance parameter')
    parser.add_argument('--wd', type=float, default=1e-3,
                        help='weight decay')
    parser.add_argument("--act", type=bool, default=True,
                        help="using activation function or not")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU Index, -1 for using CPU (default: 0)")
    parser.add_argument("--verbose", type=bool, default=True,
                        help="print log information")

    args = parser.parse_args()
    print(args.dataset)
    # TODO: add model example here


if __name__ == "__main__":
    main()
