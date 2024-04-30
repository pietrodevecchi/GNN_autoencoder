import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'

import torch

import argparse
import numpy as np

from learn_geom_model import Learner
from utils import str2bool



def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.save_plot and not os.path.exists('images'):
        os.makedirs('images')

    learner = Learner(args)

    if args.train_model: learner.train()
    
    learner.forecast(args.save_plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Neural Networks solver for PDEs')

    # Problem
    parser.add_argument('--example', default='AD', type=str, help='example name')
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')

    parser.add_argument('--train_model', default=True, type=str2bool, help='train or test')
    parser.add_argument('--plot_nodes', default=0, type=int, help='Plot option, set to 0,1,2')


    parser.add_argument('--train_size', default=80, type=int, help='training set size')
    parser.add_argument('--resume_training', default=False, type=str2bool, help='resume training')

    # Net Parameters
    # parser.add_argument('in_node', action='store_const', const=3)
    parser.add_argument('in_node', action='store_const', const=1)
    parser.add_argument('in_edge', action='store_const', const=3)
    parser.add_argument('out_channels', action='store_const', const=1)
    # AAA
    #parser.add_argument('--mlp_layers', default=3, type=int, help='number of hidden layers per MLP')
    parser.add_argument('--mlp_layers', default=3, type=int, help='number of hidden layers per MLP')
    # parser.add_argument('--hidden_channels', default=32, type=int, help='dimension of hidden units')
    # parser.add_argument('--mp_steps', default=12, type=int, help='number of message passing steps')
    # AAA
    #parser.add_argument('--hidden_channels', default=16, type=int, help='dimension of hidden units')
    parser.add_argument('--hidden_channels', default=16, type=int, help='dimension of hidden units')
    # parser.add_argument('--hidden_channels', default=8, type=int, help='dimension of hidden units')
    parser.add_argument('--dropout_p', type=float, default=0.2, help='drop output')


    # parser.add_argument('--mp_steps', default=1, type=int, help='number of message passing steps')
    # AAA
    # parser.add_argument('--MPblock_layers', default=3, type=int, help='number of message passing layers per MPBlock')
    parser.add_argument('--MPblock_layers', default=3, type=int, help='number of message passing layers per MPBlock')


    parser.add_argument('--pool_k', default=0.5, type=float, help='k-pooling factor(s)')
    parser.add_argument('--pooling_depth', default=2, type=int, help='number of pooling layers')

    parser.add_argument('--edge_augment_factor', default=0, type=float, help='augment factor, set to 0 for no augmentation')
    parser.add_argument('--edge_augment_pooling', default=True, type=str2bool, help='augment adjacency matrix after pooling')

    parser.add_argument('--multi-scaling', default=True, type=str2bool, help='use multi-scaling')
    parser.add_argument('--scales', default=[0.1, 0.2], nargs='+', type=float, help='scales for coarsening message passing')
    


    # Training Parameters
    parser.add_argument('--seed', default=10, type=int, help='random seed')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--noise_var', default=1e-6, type=float, help='training noise variance')
    parser.add_argument('--batch_size', default=1, type=int, help='training batch size')
    # parser.add_argument('--epochs', default=1500, type=int, help='training iterations')
    parser.add_argument('--epochs', default=1, type=int, help='training iterations')
    parser.add_argument('--milestones', default=[500, 1000], nargs='+', type=int, help='learning rate scheduler milestones')
    parser.add_argument('--w1', default=1., type=float, help='weight for loss 1')
    parser.add_argument('--w2', default=0., type=int, help='weight for loss 2')

    # Save Parameters
    parser.add_argument('--save_plot', default=True, type=str2bool, help='Save test simulation gif')
    

    args = parser.parse_args()
    if args.example != 'AD' and args.example != 'Stokes':
        raise ValueError('Example name should be AD or Stokes')
    main(args)



