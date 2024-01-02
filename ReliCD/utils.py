

import argparse
import random
import os
import torch
import numpy as np

# ranking loss

def seed_torch(seed=5):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    

def main():
    with open('../data/assist2009/config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

    parser = argparse.ArgumentParser()
    # seed_torch(seed=5)
    parser.add_argument('--dataset', type=str, default = '../data/assist2009', help = 'choose a dataset')
    parser.add_argument('--device', type=str, default = 'cuda:1', help = 'choose cpu or gpu')
    parser.add_argument('--batch_size', type=int, default = 32, help = 'batch_size')
    parser.add_argument('--epoch', type=int, default = 12, help = 'epoch num')
    parser.add_argument('--lr', type=float, default = 2e-3, help = 'learning rate')
    parser.add_argument('--dropout', type=float, default = 0.5, help = 'dropout rate')
    parser.add_argument('--student_n', type=int, default = student_n, help = 'student num')
    parser.add_argument('--exer_n', type=int, default = exer_n, help = 'exer num')
    parser.add_argument('--knowledge_n', type=int, default = knowledge_n, help = 'knowledge num')
    parser.add_argument('--kl_weight', type=float, default=1e-4, help='kl_weight')
    parser.add_argument('--loss_ranking_weight', type=float, default=0.1, help='ranking loss')
    parser.add_argument('--prednet_len1', type=int, default = 512, help = 'input layer ')
    parser.add_argument('--prednet_len2', type=int, default = 256, help = 'hidden layer ')
    parser.add_argument('--p', type=float, default = 0.1, help = 'dropout covariance ')
    parser.add_argument('--eps', type=float, default = 1e-10, help = 'prevents log from being 0')

    params = parser.parse_args()
        
    return params



if __name__ == '__main__':
    main()
    seed_torch()
    


    
