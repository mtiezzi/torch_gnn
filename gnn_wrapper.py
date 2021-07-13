import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from abc import ABCMeta, abstractmethod
from .utils import Accuracy
from torch.utils.tensorboard import SummaryWriter
import torchvision
import wandb
from .pygnn import GNN


class GNNWrapper:
    class Config:
        def __init__(self):
            self.device = None
            self.use_cuda = None
            self.dataset_path = None
            self.log_interval = None
            self.tensorboard = None
            self.task_type = None

            # hyperparams
            self.lrw = None
            self.loss_f = None
            self.epochs = None
            self.convergence_threshold = None
            self.max_iterations = None
            self.n_nodes = None
            self.state_dim = None
            self.label_dim = None
            self.output_dim = None
            self.graph_based = False
            self.activation = torch.nn.Tanh()
            self.state_transition_hidden_dims = None
            self.output_function_hidden_dims = None
            self.task_type = "semisupervised"

            # optional
            # self.loss_w = 1.
            # self.energy_weight = 0.
            # self.l2_weight = 0.

    def __init__(self, config: Config):
        self.config = config

        # to be populated
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.test_loader = None

        if self.config.tensorboard:
            self.writer = SummaryWriter('logs/tensorboard')
        self.first_flag_writer = True

    def __call__(self, dset, state_net=None, out_net=None):
        # handle the dataset info
        self._data_loader(dset)
        self.gnn = GNN(self.config, state_net, out_net).to(self.config.device)
        self._criterion()
        self._optimizer()
        self._accuracy()

    def _data_loader(self, dset):  # handle dataset data and metadata
        self.dset = dset.to(self.config.device)
        self.config.label_dim = self.dset.node_label_dim
        self.config.n_nodes = self.dset.num_nodes
        self.config.output_dim = self.dset.num_classes

    def _optimizer(self):
        # for name, param in self.gnn.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        # exit()
        self.optimizer = optim.Adam(self.gnn.parameters(), lr=self.config.lrw)
        # self.optimizer = optim.SGD(self.gnn.parameters(), lr=self.config.lrw)

    def _criterion(self, ):

        self.criterion = nn.CrossEntropyLoss()

    def _accuracy(self):
        self.TrainAccuracy = Accuracy(type=self.config.task_type)
        self.ValidAccuracy = Accuracy(type=self.config.task_type)
        self.TestAccuracy = Accuracy(type=self.config.task_type)

    def train_step(self, epoch):
        self.gnn.train()
        data = self.dset
        self.optimizer.zero_grad()
        self.TrainAccuracy.reset()
        # output computation
        output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels)
        # loss computation - semisupervised
        loss = self.criterion(output, data.targets)

        loss.backward()

        self.optimizer.step()

        # # updating accuracy
        # batch_acc = self.TrainAccuracy.update((output, target), batch_compute=True)
        with torch.no_grad():  # Accuracy computation
            # accuracy_train = torch.mean(
            #     (torch.argmax(output[data.idx_train], dim=-1) == data.targets[data.idx_train]).float())
            self.TrainAccuracy.update(output, data.targets)
            accuracy_train = self.TrainAccuracy.compute()

            if epoch % self.config.log_interval == 0:
                print(
                    'Train Epoch: {} \t Mean Loss: {:.6f}\tAccuracy Full Batch: {:.6f} \t  Best Accuracy : {:.6f}  \t Iterations: {}'.format(
                        epoch, loss, accuracy_train, self.TrainAccuracy.get_best(), iterations))

                if self.config.tensorboard:
                    self.writer.add_scalar('Training Accuracy',
                                           accuracy_train,
                                           epoch)
                    self.writer.add_scalar('Training Loss',
                                           loss,
                                           epoch)
                    self.writer.add_scalar('Training Iterations',
                                           iterations,
                                           epoch)

                    for name, param in self.gnn.named_parameters():
                        self.writer.add_histogram(name, param, epoch)
        # self.TrainAccuracy.reset()

    def predict(self, edges, agg_matrix, node_labels):
        return self.gnn(edges, agg_matrix, node_labels)

    def test_step(self, epoch):
        ####  TEST
        self.gnn.eval()
        data = self.dset
        self.TestAccuracy.reset()
        with torch.no_grad():
            output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels)
            test_loss = self.criterion(output, data.targets)

            self.TestAccuracy.update(output, data.targets)
            acc_test = self.TestAccuracy.compute()
            # acc_test = torch.mean(
            #     (torch.argmax(output[data.idx_test], dim=-1) == data.targets[data.idx_test]).float())

            if epoch % self.config.log_interval == 0:
                print('Test set: Average loss: {:.4f}, Accuracy:  ({:.4f}%) , Best Accuracy:  ({:.4f}%)'.format(
                    test_loss, acc_test, self.TestAccuracy.get_best()))

                if self.config.tensorboard:
                    self.writer.add_scalar('Test Accuracy',
                                           acc_test,
                                           epoch)
                    self.writer.add_scalar('Test Loss',
                                           test_loss,
                                           epoch)
                    self.writer.add_scalar('Test Iterations',
                                           iterations,
                                           epoch)

    def valid_step(self, epoch):
        ####  TEST
        self.gnn.eval()
        data = self.dset
        self.ValidAccuracy.reset()
        with torch.no_grad():
            output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels)
            test_loss = self.criterion(output, data.targets)

            self.ValidAccuracy.update(output, data.targets)
            acc_valid = self.ValidAccuracy.compute()
            # acc_test = torch.mean(
            #     (torch.argmax(output[data.idx_test], dim=-1) == data.targets[data.idx_test]).float())

            if epoch % self.config.log_interval == 0:
                print('Valid set: Average loss: {:.4f}, Accuracy:  ({:.4f}%) , Best Accuracy:  ({:.4f}%)'.format(
                    test_loss, acc_valid, self.ValidAccuracy.get_best()))

                if self.config.tensorboard:
                    self.writer.add_scalar('Valid Accuracy',
                                           acc_valid,
                                           epoch)
                    self.writer.add_scalar('Valid Loss',
                                           test_loss,
                                           epoch)
                    self.writer.add_scalar('Valid Iterations',
                                           iterations,
                                           epoch)


class SemiSupGNNWrapper(GNNWrapper):
    class Config:
        def __init__(self):
            self.device = None
            self.use_cuda = None
            self.dataset_path = None
            self.log_interval = None
            self.tensorboard = None
            self.task_type = None

            # hyperparams
            self.lrw = None
            self.loss_f = None
            self.epochs = None
            self.convergence_threshold = None
            self.max_iterations = None
            self.n_nodes = None
            self.state_dim = None
            self.label_dim = None
            self.output_dim = None
            self.graph_based = False
            self.activation = torch.nn.Tanh()
            self.state_transition_hidden_dims = None
            self.output_function_hidden_dims = None

            # optional
            # self.loss_w = 1.
            # self.energy_weight = 0.
            # self.l2_weight = 0.

    def __init__(self, config: Config):
        super().__init__(config)

    def _data_loader(self, dset):  # handle dataset data and metadata
        self.dset = dset.to(self.config.device)
        self.config.label_dim = self.dset.node_label_dim
        self.config.n_nodes = self.dset.num_nodes
        self.config.output_dim = self.dset.num_classes

    def _accuracy(self):
        self.TrainAccuracy = Accuracy(type="semisupervised")
        self.ValidAccuracy = Accuracy(type="semisupervised")
        self.TestAccuracy = Accuracy(type="semisupervised")

    def train_step(self, epoch):
        self.gnn.train()
        data = self.dset
        self.optimizer.zero_grad()
        self.TrainAccuracy.reset()
        # output computation
        output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels)
        # loss computation - semisupervised
        loss = self.criterion(output[data.idx_train], data.targets[data.idx_train])

        loss.backward()

        # with torch.no_grad():
        #     for name, param in self.gnn.named_parameters():
        #         if "state_transition_function" in name:
        #             #self.writer.add_histogram("gradient " + name, param.grad, epoch)
        #             param.grad = 0*  param.grad

        self.optimizer.step()

        # # updating accuracy
        # batch_acc = self.TrainAccuracy.update((output, target), batch_compute=True)
        with torch.no_grad():  # Accuracy computation
            # accuracy_train = torch.mean(
            #     (torch.argmax(output[data.idx_train], dim=-1) == data.targets[data.idx_train]).float())
            self.TrainAccuracy.update(output, data.targets, idx=data.idx_train)
            accuracy_train = self.TrainAccuracy.compute()

            if epoch % self.config.log_interval == 0:
                print(
                    'Train Epoch: {} \t Mean Loss: {:.6f}\tAccuracy Full Batch: {:.6f} \t  Best Accuracy : {:.6f}  \t Iterations: {}'.format(
                        epoch, loss, accuracy_train, self.TrainAccuracy.get_best(), iterations))

                if self.config.tensorboard:
                    self.writer.add_scalar('Training Accuracy',
                                           accuracy_train,
                                           epoch)
                    self.writer.add_scalar('Training Loss',
                                           loss,
                                           epoch)
                    self.writer.add_scalar('Training Iterations',
                                           iterations,
                                           epoch)
                    for name, param in self.gnn.named_parameters():
                        self.writer.add_histogram(name, param, epoch)
                        self.writer.add_histogram("gradient " + name, param.grad, epoch)
        # self.TrainAccuracy.reset()
        return output  # used for plotting

    def predict(self, edges, agg_matrix, node_labels):
        return self.gnn(edges, agg_matrix, node_labels)

    def test_step(self, epoch):
        ####  TEST
        self.gnn.eval()
        data = self.dset
        self.TestAccuracy.reset()
        with torch.no_grad():
            output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels)
            test_loss = self.criterion(output[data.idx_test], data.targets[data.idx_test])

            self.TestAccuracy.update(output, data.targets, idx=data.idx_test)
            acc_test = self.TestAccuracy.compute()
            # acc_test = torch.mean(
            #     (torch.argmax(output[data.idx_test], dim=-1) == data.targets[data.idx_test]).float())

            if epoch % self.config.log_interval == 0:
                print('Test set: Average loss: {:.4f}, Accuracy:  ({:.4f}%) , Best Accuracy:  ({:.4f}%)'.format(
                    test_loss, acc_test, self.TestAccuracy.get_best()))

                if self.config.tensorboard:
                    self.writer.add_scalar('Test Accuracy',
                                           acc_test,
                                           epoch)
                    self.writer.add_scalar('Test Loss',
                                           test_loss,
                                           epoch)
                    self.writer.add_scalar('Test Iterations',
                                           iterations,
                                           epoch)
            return output  # used for plotting

    def valid_step(self, epoch):
        ####  TEST
        self.gnn.eval()
        data = self.dset
        self.ValidAccuracy.reset()
        with torch.no_grad():
            output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels)
            test_loss = self.criterion(output[data.idx_valid], data.targets[data.idx_valid])

            self.ValidAccuracy.update(output, data.targets, idx=data.idx_valid)
            acc_valid = self.ValidAccuracy.compute()
            # acc_test = torch.mean(
            #     (torch.argmax(output[data.idx_test], dim=-1) == data.targets[data.idx_test]).float())

            if epoch % self.config.log_interval == 0:
                print('Valid set: Average loss: {:.4f}, Accuracy:  ({:.4f}%) , Best Accuracy:  ({:.4f}%)'.format(
                    test_loss, acc_valid, self.ValidAccuracy.get_best()))

                if self.config.tensorboard:
                    self.writer.add_scalar('Valid Accuracy',
                                           acc_valid,
                                           epoch)
                    self.writer.add_scalar('Valid Loss',
                                           test_loss,
                                           epoch)
                    self.writer.add_scalar('Valid Iterations',
                                           iterations,
                                           epoch)


class RegressionWrapper(GNNWrapper):

    def __call__(self, dset, state_net=None, out_net=None, criterion=None, wandb=True):
        # handle the dataset info
        self._data_loader(dset)
        self.gnn = GNN(self.config, state_net, out_net).to(self.config.device)
        self._criterion(criterion)
        self._optimizer()
        self._accuracy()
        self.wandb = wandb

    def _criterion(self, criterion=None):
        self.criterion = criterion

    def _accuracy(self):
        self.best_valid_loss = None
        self.patience = 5000  # TODO depending on epochs
        self.patience_counter = 0

    def train_step(self, epoch):
        self.gnn.train()
        data = self.dset
        self.optimizer.zero_grad()
        # self.TrainAccuracy.reset()
        # output computation
        output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels)
        # loss computation - semisupervised
        loss = self.criterion(output, data.targets)

        loss.backward()

        # with torch.no_grad():
        #     for name, param in self.gnn.named_parameters():
        #         if "state_transition_function" in name:
        #             #self.writer.add_histogram("gradient " + name, param.grad, epoch)
        #             param.grad = 0*  param.grad

        self.optimizer.step()

        # # updating accuracy
        # batch_acc = self.TrainAccuracy.update((output, target), batch_compute=True)
        with torch.no_grad():

            if epoch % self.config.log_interval == 0:
                # where the magic happens
                if self.wandb:
                    wandb.log({"epoch": epoch, "train_loss": loss, "iterations": iterations}, step=epoch)

                print(
                    f'Train Epoch: {epoch} \t Mean Loss: {loss:.6f}\t Iterations: {iterations}')

                if self.config.tensorboard:

                    self.writer.add_scalar('Training Loss',
                                           loss,
                                           epoch)
                    self.writer.add_scalar('Training Iterations',
                                           iterations,
                                           epoch)
                    for name, param in self.gnn.named_parameters():
                        self.writer.add_histogram(name, param, epoch)
                        self.writer.add_histogram("gradient " + name, param.grad, epoch)
        # self.TrainAccuracy.reset()
        return output  # used for plotting

    def predict(self, edges, agg_matrix, node_labels):
        return self.gnn(edges, agg_matrix, node_labels)

    def test_step(self, epoch):
        ####  TEST
        self.gnn.eval()
        data = self.dset
        # self.TestAccuracy.reset()
        with torch.no_grad():
            output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels)
            test_loss = self.criterion(output, data.targets)

            # self.TestAccuracy.update(output, data.targets, idx=data.idx_test)
            # acc_test = self.TestAccuracy.compute()
            # acc_test = torch.mean(
            #     (torch.argmax(output[data.idx_test], dim=-1) == data.targets[data.idx_test]).float())

            # where the magic happens

            if epoch % self.config.log_interval == 0:
                if self.wandb:
                    wandb.log({"epoch": epoch, "test_loss": test_loss, "iterations": iterations}, step=epoch)
                print(
                    f'Test Epoch: {epoch} \t Mean Loss: {test_loss:.6f}\t Iterations: {iterations}')

                if self.config.tensorboard:
                    self.writer.add_scalar('Test Loss',
                                           test_loss,
                                           epoch)
                    self.writer.add_scalar('Test Iterations',
                                           iterations,
                                           epoch)
        return output  # used for plotting

    def valid_step(self, epoch):
        ####  TEST
        self.gnn.eval()
        data = self.dset
        # self.ValidAccuracy.reset()
        with torch.no_grad():
            output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels)
            test_loss = self.criterion(output, data.targets)

            # self.ValidAccuracy.update(output, data.targets, idx=data.idx_valid)
            # acc_valid = self.ValidAccuracy.compute()
            # acc_test = torch.mean(
            #     (torch.argmax(output[data.idx_test], dim=-1) == data.targets[data.idx_test]).float())
            # where the magic happens

            if epoch % self.config.log_interval == 0:
                if self.wandb:
                    wandb.log({"epoch": epoch, "valid_loss": test_loss, "iterations": iterations}, step=epoch)
                print(
                    f'Valid Epoch: {epoch} \t Mean Loss: {test_loss:.6f}\t Iterations: {iterations}')

                if self.config.tensorboard:
                    self.writer.add_scalar('Valid Loss',
                                           test_loss,
                                           epoch)
                    self.writer.add_scalar('Valid Iterations',
                                           iterations,
                                           epoch)

            if self.best_valid_loss is None:
                # fist time populating test loss
                self.best_valid_loss = test_loss

            if test_loss < self.best_valid_loss:
                self.best_valid_loss = test_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter == self.patience:
                exit()

            return output  # used for plotting
