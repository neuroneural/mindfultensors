import torch
import io
import numpy as np
from typing import Sized
from torch.utils.data.sampler import Sampler


def unit_interval_normalize(img):
    """Unit interval preprocessing"""
    img = (img - img.min()) / (img.max() - img.min())
    return img


def qnormalize(img, qmin=0.01, qmax=0.99):
    """Unit interval preprocessing"""
    img = (img - img.quantile(qmin)) / (img.quantile(qmax) - img.quantile(qmin))
    return img


def mtransform(tensor_binary):
    buffer = io.BytesIO(tensor_binary)
    tensor = torch.load(buffer)
    return tensor


def mcollate(results, field=("input", "label")):
    results = results[0]
    # Assuming 'results' is your dictionary containing all the data
    input_tensors = [results[id_][field[0]] for id_ in results.keys()]
    label_tensors = [results[id_][field[1]] for id_ in results.keys()]
    # Stack all input tensors into a single tensor
    stacked_inputs = torch.stack(input_tensors)
    # Stack all label tensors into a single tensor
    stacked_labels = torch.stack(label_tensors)
    return stacked_inputs.unsqueeze(1), stacked_labels.long()


def collate_subcubes(results, coord_generator, samples=4):
    data, labels = mcollate(results)
    num_subjs = labels.shape[0]
    data = data.squeeze(1)

    batch_data = []
    batch_labels = []

    for i in range(num_subjs):
        subcubes, sublabels = subcube_list(
            data[i, :, :, :], labels[i, :, :, :], samples, coord_generator
        )
        batch_data.extend(subcubes)
        batch_labels.extend(sublabels)

    # Converting the list of tensors to a single tensor
    batch_data = torch.stack(batch_data).unsqueeze(1)
    batch_labels = torch.stack(batch_labels)

    return batch_data, batch_labels


def subcube_list(cube, labels, num, coords_generator):
    subcubes = []
    sublabels = []

    for i in range(num):
        coords = coords_generator.get_coordinates()
        subcube = cube[
            coords[0][0] : coords[0][1],
            coords[1][0] : coords[1][1],
            coords[2][0] : coords[2][1],
        ]
        sublabel = labels[
            coords[0][0] : coords[0][1],
            coords[1][0] : coords[1][1],
            coords[2][0] : coords[2][1],
        ]
        subcubes.append(subcube)
        sublabels.append(sublabel)

    return subcubes, sublabels


class DBBatchSampler(Sampler):
    """
    A batch sampler from a random permutation. Used for generating indices for MongoDataset
    """

    data_source: Sized

    def __init__(self, data_source, batch_size=1, seed=None):
        """TODO describe function

        :param data_source: a dataset of Dataset class
        :param batch_size: number of samples in the batch (sample is an MRI split to 8 records)
        :returns: an object of mBatchSampler class

        """
        self.batch_size = batch_size
        self.data_source = data_source
        self.data_size = len(self.data_source)
        self.seed = seed

    def __chunks__(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    def __iter__(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        return self.__chunks__(
            np.random.permutation(self.data_size), self.batch_size
        )

    def __len__(self):
        return (
            self.data_size + self.batch_size - 1
        ) // self.batch_size  # Number of batches
