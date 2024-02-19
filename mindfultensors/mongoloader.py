from typing import Sized
import ipdb

import numpy as np
from pymongo import MongoClient
import torch
import io
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from mindfultensors.gencoords import CoordsGenerator


def unit_interval_normalize(img):
    """Unit interval preprocessing"""
    img = (img - img.min()) / (img.max() - img.min())
    return img


def qnormalize(img, qmin=0.01, qmax=0.99):
    """Unit interval preprocessing"""
    img = (img - img.quantile(qmin)) / (img.quantile(qmax) - img.quantile(qmin))
    return img


class MongoDataset(Dataset):
    """
    A dataset for fetching batches of records from a MongoDB
    """

    def __init__(
        self,
        indices,
        transform,
        collection,
        sample,
        normalize=unit_interval_normalize,
        id="id",
    ):
        """Constructor

        :param indices: a set of indices to be extracted from the collection
        :param transform: a function to be applied to each extracted record
        :param collection: pymongo collection to be used
        :param sample: a pair of fields to be fetched as `input` and `label`, e.g. (`T1`, `label104`)
        :param id: the field to be used as an index. The `indices` are values of this field
        :returns: an object of MongoDataset class

        """

        self.indices = indices
        self.transform = transform
        self.collection = collection
        # self.fields = {_: 1 for _ in self.fields} if fields is not None else {}
        self.fields = {"id": 1, "chunk": 1, "kind": 1, "chunk_id": 1}
        self.sample = sample
        self.normalize = normalize
        self.id = id

    def __len__(self):
        return len(self.indices)

    def make_serial(self, samples_for_id, kind):
        return b"".join(
            [
                sample["chunk"]
                for sample in sorted(
                    (
                        sample
                        for sample in samples_for_id
                        if sample["kind"] == kind
                    ),
                    key=lambda x: x["chunk_id"],
                )
            ]
        )

    def __getitem__(self, batch):
        # Fetch all samples for ids in the batch and where 'kind' is either
        # data or labela s specified by the sample parameter
        samples = list(
            self.collection["bin"].find(
                {
                    self.id: {"$in": [self.indices[_] for _ in batch]},
                    "kind": {"$in": self.sample},
                },
                self.fields,
            )
        )

        results = {}
        for id in batch:
            # Separate samples for this id
            samples_for_id = [
                sample
                for sample in samples
                if sample[self.id] == self.indices[id]
            ]

            # Separate processing for each 'kind'
            data = self.make_serial(samples_for_id, self.sample[0])
            label = self.make_serial(samples_for_id, self.sample[1])

            # Add to results
            results[id] = {
                "input": self.normalize(self.transform(data).float()),
                "label": self.transform(label),
            }

        return results


class MBatchSampler(Sampler):
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


def name2collections(name: str, database):
    collection_bin = database[f"{name}.bin"]
    collection_meta = database[f"{name}.meta"]
    return collection_bin, collection_meta


def create_client(worker_id, dbname, colname, mongohost):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    client = MongoClient("mongodb://" + mongohost + ":27017")
    colbin, colmeta = name2collections(colname, client[dbname])
    dataset.collection = {"bin": colbin, "meta": colmeta}


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
