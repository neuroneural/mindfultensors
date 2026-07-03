import lz4.frame
import torch
import io
import bson
import numpy as np
from typing import Sized
from torch.utils.data.sampler import Sampler

# LZ4 Frame magic bytes: 0x04224D18 (little-endian)
LZ4_MAGIC = b"\x04\x22\x4d\x18"


def unit_interval_normalize(img):
    """Unit interval preprocessing"""
    img = (img - img.min()) / (img.max() - img.min())
    return img


def qnormalize(img, qmin=0.01, qmax=0.99):
    """Unit interval preprocessing"""
    img = (img - img.quantile(qmin)) / (img.quantile(qmax) - img.quantile(qmin))
    return img


def mtransform(tensor_binary):
    # Check if data is LZ4 compressed by looking for magic bytes
    if tensor_binary[:4] == LZ4_MAGIC:
        tensor_binary = lz4.frame.decompress(tensor_binary)

    buffer = io.BytesIO(tensor_binary)
    tensor = torch.load(buffer, weights_only=True)
    return tensor


def tensor2bin(tensor):
    """Serialize a tensor to a binary blob."""
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


def insert_kind(collection, subject_id, kind, value, id_field="id", chunk_size_mb=10):
    """Insert a kind (tensor or scalar) into a unified collection.

    :param collection: pymongo collection to insert into
    :param subject_id: the subject's index value, stored under `id_field`
    :param kind: the kind name, e.g. `smri` or `age`
    :param value: a tensor (chunked and stored as binary) or a scalar (int/float/bool)
    :param id_field: the field name to store subject_id under
    :param chunk_size_mb: chunk size in MB for tensor kinds
    """
    if torch.is_tensor(value):
        _insert_tensor_kind(collection, subject_id, kind, value, id_field, chunk_size_mb)
    else:
        collection.insert_one({
            id_field: subject_id,
            "kind":   kind,
            "dtype":  type(value).__name__,
            "value":  value,
        })


def _insert_tensor_kind(collection, subject_id, kind, tensor, id_field, chunk_size_mb):
    binary           = tensor2bin(tensor)
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    num_chunks       = (len(binary) + chunk_size_bytes - 1) // chunk_size_bytes
    docs = []
    for chunk_id in range(num_chunks):
        start = chunk_id * chunk_size_bytes
        end   = min(start + chunk_size_bytes, len(binary))
        docs.append({
            id_field:   subject_id,
            "kind":     kind,
            "dtype":    "tensor",
            "chunk_id": chunk_id,
            "chunk":    bson.Binary(binary[start:end]),
        })
    collection.insert_many(docs)


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
        return self.__chunks__(np.random.permutation(self.data_size), self.batch_size)

    def __len__(self):
        return (
            self.data_size + self.batch_size - 1
        ) // self.batch_size  # Number of batches
