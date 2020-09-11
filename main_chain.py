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

from lpgnn_wrapper import LPGNNWrapper, SemiSupLPGNNWrapper


#
# fix random seeds for reproducibility
SEED = 780104040
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


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
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='logging training status cadency')
    # parser.add_argument('--save-model', action='store_true', default=False,
    #                     help='For Saving the current Model')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='For logging the model in tensorboard')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if not use_cuda:
        args.n_gpu_use = 0

    device = utils.prepare_device(n_gpu_use=args.n_gpu_use, gpu_id=args.cuda_dev)
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # configugations
    cfg = LPGNNWrapper.Config()
    cfg.use_cuda = use_cuda
    cfg.device = device
    # cfg.seed = SEED

    cfg.log_interval = args.log_interval
    cfg.tensorboard = args.tensorboard

    cfg.dataset_path = './data'
    cfg.epochs = args.epochs
    cfg.activation = nn.Tanh()
    cfg.state_transition_hidden_dims = [10, ]
    cfg.output_function_hidden_dims = [3, ]
    # cfg.state_dim = [7, 2]
    cfg.state_dim = [5, ]
    cfg.graph_based = False
    cfg.log_interval = 300
    cfg.lrw = 0.01
    cfg.lrx = 0.03
    cfg.lrλ = 0.01
    cfg.task_type = "semisupervised"
    cfg.layers = len(cfg.state_dim) if type(
        cfg.state_dim) is list else 1  # getting number of LPGNN layers from state_dim list

    # LPGNN
    cfg.eps = 1e-6
    cfg.state_constraint_function = "eps"
    cfg.loss_w = 0.001
    # model creation  - a unique model
    model = SemiSupLPGNNWrapper(cfg)
    # dataset creation
    #dset = dataloader.get_karate(aggregation_type="sum", sparse_matrix=True)  # generate the dataset
    # dset = dataloader.get_twochainsSSE(aggregation_type="sum", percentage=0.1, sparse_matrix=True)  # generate the dataset
    dset = dataloader.get_twochains(num_nodes_per_graph= 1000,
        pct_labels= .2,
        pct_valid= .2,sparse_matrix=True)  # generate the dataset
    model(dset)  # dataset initalization into the GNN

    import time
    start_get = time.time()
    # training code
    for epoch in range(args.epochs):
        model.global_step(epoch, start_get)
        # model.valid_step(epoch)
    # model.test_step()

    # if args.save_model:
    #     torch.save(model.gnn.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
