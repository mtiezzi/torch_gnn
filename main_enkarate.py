import torch
import torch.nn as nn
import argparse
import utils
import dataloader

from gnn_wrapper import GNNWrapper, SemiSupGNNWrapper


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

    # configurations
    cfg = GNNWrapper.Config()
    cfg.use_cuda = use_cuda
    cfg.device = device

    cfg.log_interval = args.log_interval
    cfg.tensorboard = args.tensorboard

    cfg.dataset_path = './data'
    cfg.epochs = args.epochs
    cfg.lrw = args.lr
    cfg.activation = nn.Tanh()
    cfg.state_transition_hidden_dims = [5,]
    cfg.output_function_hidden_dims = [5]
    cfg.state_dim = 2
    cfg.max_iterations = 50
    cfg.convergence_threshold = 0.01
    cfg.graph_based = False
    cfg.log_interval = 10
    cfg.task_type = "semisupervised"

    cfg.lrw = 0.001

    # model creation
    model = SemiSupGNNWrapper(cfg)
    # dataset creation
    E, N, targets, mask_train, mask_test = dataloader.old_load_karate()
    dset = dataloader.from_EN_to_GNN(E, N, targets, aggregation_type="sum", sparse_matrix=True)  # generate the dataset
    dset.idx_train = mask_train
    dset.idx_test = mask_test
    model(dset)  # dataset initalization into the GNN

    # training code
    for epoch in range(1, args.epochs + 1):
        model.train_step(epoch)

        if epoch % 10 == 0:
            model.test_step(epoch)

if __name__ == '__main__':
    main()
