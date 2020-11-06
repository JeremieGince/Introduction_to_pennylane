import pennylane.numpy as np
import random
import torch
from torch.utils.data import Dataset
from scipy.stats import zscore
from scipy.signal import lfilter
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


class MoonDataset(Dataset):
    def __init__(self):
        # Fixing the dataset and problem
        self.X, self.y = make_moons(n_samples=200, noise=0.1)
        self.y_ = torch.unsqueeze(torch.tensor(self.y), 1)  # used for one-hot encoded labels
        self.y_hot = torch.scatter(torch.zeros((200, 2)), 1, self.y_, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y_hot[idx]

    def show(self):
        c = ["#1f77b4" if y_ == 0 else "#ff7f0e" for y_ in self.y]  # colours for each class
        plt.axis("off")
        plt.scatter(self.X[:, 0], self.X[:, 1], c=c)
        plt.show()



class TimeSeriesDataset(Dataset):

    def __init__(self, simulation, frequency_sampling=2.0, ts_type="calcium", ts_length=20, t_toSkip=10, noise=0.05):
        super(TimeSeriesDataset, self).__init__()
        self.simulation = simulation
        self.W = simulation["brain"].W
        self.spike_times = simulation["spike_times"]
        self.calcium_tau = simulation["calcium_tau"]
        self.frequency_sampling = frequency_sampling
        self.ts_type = ts_type
        self.ts_length = ts_length
        self.t_toSkip = t_toSkip
        self.noise = noise

        self.ts_spikes = self.resample(self.spike_times, self.frequency_sampling)
        self.ts_calcium = self.calcium_filter()

        if self.t_toSkip:
            self.ts_spikes = self.ts_spikes[:, int(self.t_toSkip * self.frequency_sampling):]
            self.ts_calcium = self.ts_calcium[:, int(self.t_toSkip * self.frequency_sampling):]

    def __len__(self):
        return self.ts_spikes.shape[1] - self.ts_length - 1

    def __getitem__(self, idx):

        if self.ts_type == "spikes":
            TS = self.ts_spikes
        elif self.ts_type == "calcium":
            TS = self.ts_calcium

        t_initial = np.random.randint(0, np.shape(TS)[1] - self.ts_length - 1)
        ts = np.transpose(TS[:, t_initial:t_initial + self.ts_length])
        target = TS[:, t_initial + self.ts_length + 1]

        if self.noise:
            ts = ts + np.random.normal(0, self.noise, ts.shape)
            target = target + np.random.normal(0, self.noise, target.shape)

        return [torch.from_numpy(ts), target]

    def resample(self, spike_times, frequency_sampling):
        frequency_simulation = 1000 / self.simulation["dt"]
        n_increment_per_sample = int(frequency_simulation / frequency_sampling)
        ts_spikes = []

        for neuron in range(self.simulation["brain"].N):
            timeSerie = self.simulation["spike_times"][neuron].toarray()[0]
            timeSerie = timeSerie[0:n_increment_per_sample * int(
                len(timeSerie) / n_increment_per_sample)]  # Keeping multiple of n...
            ts_spikes.append(np.sum(timeSerie.reshape(-1, n_increment_per_sample), axis=1))
        return np.array(ts_spikes)

    def calcium_filter(self):
        dt = 1 / self.frequency_sampling
        gamma = 1 - dt / self.calcium_tau
        ts_calcium = lfilter(np.array([1]), [1, -gamma], self.ts_spikes, axis=1)
        ts_calcium = zscore(ts_calcium, axis=1)
        ts_calcium = np.nan_to_num(ts_calcium)
        return ts_calcium