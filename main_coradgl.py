import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import utils
import dataloader

from gnn_wrapper import GNNWrapper, SemiSupGNNWrapper


#
# # fix random seeds for reproducibility
# SEED = 123
# torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(SEED)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--epochs', type=int, default=100000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--cuda_dev', type=int, default=0,
                        help='select specific CUDA device for training')
    parser.add_argument('--n_gpu_use', type=int, default=1,
                        help='select number of CUDA device for training')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='logging training status cadency')
    parser.add_argument('--tensorboard', action='store_true', default=True,
                        help='For logging the model in tensorboard')


    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if not use_cuda:
        args.n_gpu_use = 0

    device = utils.prepare_device(n_gpu_use=args.n_gpu_use, gpu_id=args.cuda_dev)

    # configugations
    cfg = GNNWrapper.Config()
    cfg.use_cuda = use_cuda
    cfg.device = device

    cfg.log_interval = args.log_interval
    cfg.tensorboard = args.tensorboard

    cfg.dataset_path = './data'
    cfg.epochs = args.epochs
    cfg.lrw = args.lr
    cfg.activation = nn.Tanh()
    cfg.state_transition_hidden_dims = [100, 50]
    cfg.output_function_hidden_dims = [50, ]
    cfg.state_dim = 50
    cfg.max_iterations = 50
    cfg.convergence_threshold = 0.00001
    cfg.graph_based = False
    cfg.task_type = "semisupervised"

    cfg.lrw = 0.001

    # model creation
    model = SemiSupGNNWrapper(cfg)
    # dataset creation
    dset = dataloader.get_dgl_cora(aggregation_type="sum", sparse_matrix=True) # generate the dataset
    #dset = dataloader.get_dgl_citation(aggregation_type="sum") # generate the dataset
    #dset = dataloader.get_dgl_karate(aggregation_type="sum")  # generate the dataset

    model(dset)  # dataset initialization into the GNN

    # training code
    for epoch in range(1, args.epochs + 1):
        model.train_step(epoch)

        model.valid_step(epoch)
        model.test_step(epoch)

if __name__ == '__main__':
    main()
