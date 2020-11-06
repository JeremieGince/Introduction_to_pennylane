import pennylane.numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import tqdm


class ModelTrainer:

    def __init__(self,
                 model,
                 batch_size,
                 train_split,
                 train_dataset,
                 test_dataset=None,
                 loss_function=nn.CrossEntropyLoss(),
                 optimizer=optim.SGD,
                 optimizer_parameters={"lr": 0.01, "momentum": 0.9, "nesterov": True, "weight_decay": 10 ** -6},
                 scheduler=None,
                 scheduler_parameters={},
                 use_cuda=torch.cuda.is_available()
                 ):
        self.model = model
        self.best_model = []
        self.best_accuracy = 0.0
        self.use_cuda = use_cuda
        if self.use_cuda:
            model.cuda()

        self.history = {
            'epochs': [],
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': []
        }

        # Dataset
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.test_dataset = test_dataset
        self.train_loader, self.val_loader = self.get_data_loaders(self.train_dataset,
                                                                   train_split=train_split,
                                                                   shuffle=True)
        if test_dataset is not None:
            self.test_loader = self.get_data_loaders_test(self.test_dataset)

        # Loss function
        self.criterion = loss_function

        # Optimizer
        self.optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            **optimizer_parameters)

        # Scheduler
        self.scheduler = None
        if scheduler:
            self.scheduler = scheduler(self.optimizer, **scheduler_parameters)

        # Time
        self.time_train = []
        self.time_test = []

    def __repr__(self):
        return str(self.criterion) + str(self.optimizer) + str(self.scheduler) + str(self.history)

    def do_epoch(self):
        if self.scheduler:
            self.scheduler.step()

        for j, (inputs, targets) in enumerate(self.train_loader):
            if self.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            inputs = Variable(inputs)
            targets = Variable(targets).long()

            # Use model.zero_grad() instead of optimizer.zero_grad()
            # Otherwise, variables that are not optimized won't be cleared
            self.model.zero_grad()
            output = self.model(inputs)

            loss = self.criterion(output, targets)
            loss.backward()
            self.optimizer.step()

    def train(self, n_epoch, test_val_freq=1, test_train_freq=1):
        if self.history['epochs']:
            current_epoch = self.history['epochs'][-1]
        else:
            current_epoch = 0
        end_epoch = current_epoch + n_epoch

        while current_epoch < end_epoch:
            current_epoch += 1

            self.model.train()

            start = time.time()
            self.do_epoch()
            end = time.time()

            train_str = val_str = ''
            train_acc = val_acc = train_loss = val_loss = None

            if (end_epoch - current_epoch) % test_train_freq == 0:
                train_acc, train_loss = self.validate(self.train_loader)
                if train_acc:
                    train_str = ' - Train acc: {:.2f}'.format(train_acc)
                train_str += ' - Train loss: {:.4f}'.format(train_loss)

            if (end_epoch - current_epoch) % test_val_freq == 0 and self.val_loader:
                val_acc, val_loss = self.validate(self.val_loader)
                if val_acc:
                    val_str = ' - Val acc: {:.2f}'.format(val_acc)
                val_str += ' - Val loss: {:.4f}'.format(val_loss)

            #             print('Epoch {}'.format(current_epoch)
            #                   + train_str
            #                   + val_str
            #                   + ' - Training time: {:.2f}s'.format(end - start))

            self.save_history(current_epoch, train_acc, val_acc, train_loss, val_loss)

            if train_acc > self.best_accuracy:
                self.best_accuracy = train_acc
                self.best_model = self.model

    def test(self):
        if self.test_dataset is None:
            return

        true, proba = self.validate_test(self.test_loader)

        return true, proba

    def validate_test(self, val_loader):
        true = []
        proba = []

        smax = nn.Softmax()
        self.model.eval()

        for j, (inputs, targets) in enumerate(val_loader):

            if self.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            inputs = Variable(inputs, volatile=True)
            targets = Variable(targets, volatile=True).long()
            output = self.model(inputs)

            true.extend(targets.data.cpu().numpy().tolist())
            proba.extend(smax(output).data.cpu().numpy().tolist())

        return true, proba

    def validate(self, val_loader):
        true = []
        pred = []
        val_loss_sum = 0

        self.model.eval()

        for j, (inputs, targets) in enumerate(val_loader):

            if self.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            inputs = Variable(inputs, volatile=True)
            targets = Variable(targets, volatile=True).long()
            output = self.model(inputs)

            #             val_loss_sum += self.criterion(output, targets).data.cpu()[0] * inputs.shape[0]
            val_loss_sum += self.criterion(output, targets).data.cpu() * inputs.shape[0]

            if isinstance(self.criterion, nn.CrossEntropyLoss):
                predictions = output.max(dim=1)[1]
                true.extend(targets.data.cpu().numpy().tolist())
                pred.extend(predictions.data.cpu().numpy().tolist())

        if isinstance(self.criterion, nn.CrossEntropyLoss):
            accuracy = (np.array(true) == np.array(pred)).mean() * 100
        else:
            accuracy = None

        return accuracy, val_loss_sum / len(val_loader.dataset)

    def save_history(self, epoch, train_acc, val_acc, train_loss, val_loss):
        self.history['epochs'].append(epoch)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)

    def get_data_loaders(self, dataset, train_split, shuffle=True):
        num_data = len(dataset)
        indices = np.arange(num_data)

        if shuffle:
            np.random.shuffle(indices)

        if train_split == 1:
            sampler = SubsetRandomSampler(indices)
            loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)
            return loader, None

        else:
            split = int(train_split * num_data)
            train_idx, valid_idx = indices[:split], indices[split:]

            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            train_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler)
            valid_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=valid_sampler)

            return train_loader, valid_loader

    def get_data_loaders_test(self, dataset):
        indices = np.arange(len(dataset))
        sampler = SequentialSampler(indices)
        loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, shuffle=False)
        return loader

    def plot_history(self, saving=False):
        epochs = self.history['epochs']

        fig, axes = plt.subplots(2, 1)

        axes[0].set_title('Train accuracy')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Accuracy')
        axes[0].plot(epochs, self.history['train_acc'], label='Train')
        axes[0].plot(epochs, self.history['val_acc'], label='Validation')
        axes[0].legend()

        axes[1].set_title('Train loss')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')
        axes[1].plot(epochs, self.history['train_loss'], label='Train')
        axes[1].plot(epochs, self.history['val_loss'], label='Validation')
        plt.tight_layout()
        if saving:
            plt.savefig("figures/training_history.png", dpi=300)
        plt.show()
