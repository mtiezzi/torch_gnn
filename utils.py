import torch
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torchvision


def prepare_device(n_gpu_use, gpu_id=None):
    """
    setup specific GPU device if available, move model into configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
              "on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:{}'.format(gpu_id) if n_gpu_use > 0 else 'cpu')
    print("Executing on device: ", device)
    return device


class Metric:
    def __init__(self):
        self.reset()

    @abstractmethod
    def reset(self):
        """
        Resets the metric to it's initial state.

        This is called at the start of each epoch.
        """
        pass

    @abstractmethod
    def update(self, output, target):
        """
        Updates the metric's state using the passed batch output.

        This is called once for each batch.

        Args:
            output: the is the output from the engine's process function.
            target: target to match
        """
        pass

    @abstractmethod
    def compute(self):
        """
        Computes the metric based on it's accumulated state.

        This is called at the end of each epoch.

        Returns:
            Any: the actual quantity of interest.

        Raises:
            NotComputableError: raised when the metric cannot be computed.
        """
        pass


class Accuracy(Metric):

    def __init__(self, is_multilabel=False, type=None):
        self._is_multilabel = is_multilabel
        self._type = type
        # self._num_classes = None
        self._num_correct = None
        self._num_examples = None
        self.best_accuracy = -1
        super(Accuracy, self).__init__()

    def reset(self):
        # self._num_classes = None
        self._num_correct = 0
        self._num_examples = 0
        super(Accuracy, self).reset()

    def update(self, output, target, batch_compute=False, idx=None):
        y_pred = output

        if self._type == "binary":
            correct = torch.eq(y_pred.view(-1).to(target), target.view(-1))
        elif self._type == "multiclass":
            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred))
            if batch_compute:
                batch_dim = correct.shape[0]
                batch_accuracy = torch.sum(correct).item() / batch_dim

        if self._type == "multilabel":
            # if y, y_pred shape is (N, C, ...) -> (N x ..., C)
            num_classes = y_pred.size(1)
            last_dim = y_pred.ndimension()
            y_pred = torch.transpose(y_pred, 1, last_dim - 1).reshape(-1, num_classes)
            target = torch.transpose(target, 1, last_dim - 1).reshape(-1, num_classes)
            correct = torch.all(target == y_pred.type_as(target), dim=-1)
        elif self._type == "semisupervised":
            target = target[idx]
            y_pred = y_pred[idx]
            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred))

        # elif self._type == "semisupervised":
        #     output[data.idx_test], data.targets[data.idx_test]
        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]

        if batch_compute:
            return batch_accuracy

    def get_best(self):
        return self.best_accuracy

    def compute(self):
        if self._num_examples == 0:
            raise Exception('Accuracy must have at least one example before it can be computed.')
        acc = self._num_correct / self._num_examples
        if acc > self.best_accuracy:
            self.best_accuracy = acc
        return self._num_correct / self._num_examples


