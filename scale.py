import torch
import numpy as np

class StandardScaler():
    """Standardize data by removing the mean and scaling to
    unit variance.  This object can be used as a transform
    in PyTorch data loaders.

    Args:
        mean (FloatTensor): The mean value for each feature in the data.
        scale (FloatTensor): Per-feature relative scaling.
    """

    def __init__(self, mean=None, scale=None):
        if mean is not None:
            mean = torch.FloatTensor(mean)
        if scale is not None:
            scale = torch.FloatTensor(scale)
        self.mean_ = mean
        self.scale_ = scale

    def fit(self, sample):
        """Set the mean and scale values based on the sample data.
        """
        self.mean_ = sample.mean(0, keepdim=True)
        self.scale_ = sample.std(0, unbiased=False, keepdim=True)
        return self

    def __call__(self, sample):
        return (sample - self.mean_)/self.scale_

    def inverse_transform(self, sample):
        """Scale the data back to the original representation
        """
        return sample * self.scale_ + self.mean_


def compute_std_mean(samples, edges=True):
    def std_mean(index):
        population = torch.zeros(size=(len(samples[0][index]) * len(samples), len(samples[0][index][0])))
        for i, sample in enumerate(samples):
            population[i*len(sample[index]):i*len(sample[index])+len(sample[index]), :] = sample[index]

        return torch.mean(population, dim=0), torch.std(population, dim=0)

    mean_nodes, std_nodes = std_mean(index=0)
    if edges:
        mean_edges, std_edges = std_mean(index=2)
        return mean_nodes, std_nodes, mean_edges, std_edges
    else:
        return mean_nodes, std_nodes


def normalize(samples, train_ind, edges=True):
    # Normalize mean = 0 and variance = 1
    training_samples = [samples[i] for i in train_ind]
    if edges:
        mean_nodes, std_nodes, mean_edges, std_edges = compute_std_mean(training_samples)

        return [
            (
                (sample[0] - mean_nodes) / std_nodes,
                sample[1],
                (sample[2] - mean_edges) / std_edges
            ) for i, sample in enumerate(samples)
        ]
    else:
        mean_nodes, std_nodes = compute_std_mean(training_samples, edges=False)
        return [
            (
                (sample[0] - mean_nodes) / std_nodes,
                sample[1]
            ) for i, sample in enumerate(samples)
        ]
