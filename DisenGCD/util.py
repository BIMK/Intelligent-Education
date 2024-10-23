import argparse
import pickle
import scipy.sparse as sp
import numpy as np
import torch
from DMM_ablation_only2.preprocess import build_graph


class CommonArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(CommonArgParser, self).__init__()
        self.add_argument('--lr', type=float, default=0.0001, help='learning rate')
        self.add_argument('--wd', type=float, default=0.001, help='weight decay')
        self.add_argument('--n_hid', type=int, default=64, help='hidden dimension')
        self.add_argument('--alr', type=float, default=0.0003, help='learning rate for architecture parameters')
        self.add_argument('--steps', type=int, nargs='+', help='number of intermediate states in the meta data')
        self.add_argument('--dataset', type=str, default='JunYi')
        self.add_argument('--gpu', type=int, default=0)
        self.add_argument('--epochs', type=int, default=100, help='number of epochs for supernet training')

        self.add_argument('--ratio', type=float, default=0.8)
        self.add_argument('--k', type=float, default=1, help='sampling')
        self.add_argument('--lam_seq', type=float, default=0.9, help='threshold')
        self.add_argument('--lam_res', type=float, default=0.9, help='threshold')

        self.add_argument('--exer_n', type=int, default=17746,
                          help='The number for exercise.')
        self.add_argument('--knowledge_n', type=int, default=123,
                          help='The number for knowledge concept.')
        self.add_argument('--student_n', type=int, default=4163,
                          help='The number for student.')

        self.add_argument('--dropout', type=float, default=0.3)
        self.add_argument('--no_norm', action='store_true', default=False, help='disable layer norm')
        self.add_argument('--in_nl', action='store_true', default=False, help='non-linearity after projection')
        self.add_argument('--seed', type=int, default=0)


"""
if __name__ == '__main__':
     print(build_map())
     print(len(build_map()))
"""
def construct_local_map(args):
    local_map = {
        'directed_g': build_graph('direct', args.knowledge_n),
        'undirected_g': build_graph('undirect', args.knowledge_n),
        'k_from_e': build_graph('k_from_e', args.knowledge_n + args.exer_n),
        'e_from_k': build_graph('e_from_k', args.knowledge_n + args.exer_n),
    }
    return local_map
