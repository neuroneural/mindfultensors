import torch
from pymongo import MongoClient
from torch.utils.data import Dataset, get_worker_info
from torch.utils.data.sampler import Sampler
from pymongo.errors import OperationFailure
import time

from .gencoords import CoordsGenerator
from .utils import (
    unit_interval_normalize,
    qnormalize,
    mtransform,
    mcollate,
    collate_subcubes,
    subcube_list,
    DBBatchSampler,
)

DEFAULT_TRANSFORMS = {
    "tensor": mtransform,
    "int":    lambda x: torch.tensor(x, dtype=torch.long),
    "float":  lambda x: torch.tensor(x, dtype=torch.float32),
    "bool":   lambda x: torch.tensor(x, dtype=torch.bool),
}

__all__ = [
    "unit_interval_normalize",
    "qnormalize",
    "mtransform",
    "mcollate",
    "collate_subcubes",
    "subcube_list",
    "MongoDataset",
    "MongoheadDataset",
    "name2collection",
    "create_client",
    "DBBatchSampler",
]


class MongoDataset(Dataset):
    """
    A dataset for fetching batches of records from a single MongoDB collection
    """

    def __init__(
        self,
        indices,
        collection,
        fetch,
        transforms=None,
        normalize=unit_interval_normalize,
        fields=None,
        id="id",
    ):
        """Constructor

        :param indices: a set of indices to be extracted from the collection
        :param collection: pymongo collection to be used
        :param fetch: a tuple of kind names to be fetched, e.g. (`smri`, `gender_encoded`). Supports both chunk kinds (tensors) and scalar kinds (single value docs)
        :param transforms: optional dict mapping dtype strings to transform functions, e.g. `{"int": lambda x: torch.tensor(x, dtype=torch.long)}`. Overrides DEFAULT_TRANSFORMS for the specified dtype.
        :param normalize: a function to be applied to each tensor kind after transform
        :param fields: optional MongoDB projection dict, e.g. `{"id": 1, "kind": 1, "chunk": 1}`. Fetches all fields if not specified.
        :param id: the field to be used as an index. The `indices` are values of this field
        :returns: an object of MongoDataset class

        """
        if not fetch:
            raise ValueError("fetch must be a non-empty tuple of kind names")

        self.indices = indices
        self.collection = collection
        self.fetch = tuple(fetch)
        self.transforms = {**DEFAULT_TRANSFORMS, **(transforms or {})}
        self.normalize = normalize
        self.fields = fields or {}
        self.id = id

        collection.create_index([(id, 1), ("kind", 1)])

    def __len__(self):
        return len(self.indices)

    def make_serial(self, kind_docs):
        ordered = [None] * len(kind_docs)
        for doc in kind_docs:
            ordered[doc["chunk_id"]] = doc["chunk"]
        return b"".join(ordered)

    def __getitem__(self, batch):
        batch_ids = [self.indices[i] for i in batch]
        docs = list(self.collection.find(
            {self.id: {"$in": batch_ids}, "kind": {"$in": list(self.fetch)}},
            self.fields or None,
        ))

        grouped = {}
        for doc in docs:
            key = (doc[self.id], doc["kind"])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(doc)

        results = {}
        for idx in batch:
            subject_id = self.indices[idx]
            results[idx] = {}
            for kind in self.fetch:
                kind_docs = grouped.get((subject_id, kind), [])
                if not kind_docs:
                    continue
                dtype = kind_docs[0]["dtype"]
                t = self.transforms.get(dtype, lambda x: x)
                if dtype == "tensor":
                    binary = self.make_serial(kind_docs)
                    results[idx][kind] = self.normalize(t(binary).float())
                else:
                    results[idx][kind] = t(kind_docs[0]["value"])

        return results


class MongoheadDataset(MongoDataset):
    def __init__(self, *args, keeptrying=True, **kwargs):
        """Constructor

        :param indices: a set of indices to be extracted from the collection
        :param collection: pymongo collection to be used
        :param fetch: a tuple of kind names to be fetched, e.g. (`smri`, `gender_encoded`)
        :param transforms: optional dict mapping dtype strings to transform functions
        :param id: the field to be used as an index. The `indices` are values of this field
        :param keeptrying: whether to keep retrying to fetch a record if the process failed or just report this and fail
        :returns: an object of MongoheadDataset class

        """
        super().__init__(*args, **kwargs)
        self.keeptrying = keeptrying  # Initialize the keeptrying attribute

    def retry_on_eof_error(retry_count, verbose=False):
        def decorator(func):
            def wrapper(self, batch, *args, **kwargs):
                myException = Exception  # Default Exception if not overwritten
                for attempt in range(retry_count):
                    try:
                        return func(self, batch, *args, **kwargs)
                    except (
                        EOFError,
                        OperationFailure,
                        RuntimeError,
                    ) as e:  # Specifically catching EOFError
                        if self.keeptrying:
                            if verbose:
                                print(
                                    f"EOFError caught. Retrying {attempt+1}/{retry_count}"
                                )
                            time.sleep(1)
                            myException = e
                            continue
                        else:
                            raise e
                raise myException("Failed after multiple retries.")

            return wrapper

        return decorator

    @retry_on_eof_error(retry_count=10, verbose=True)
    def __getitem__(self, batch):
        # Directly use the parent class's __getitem__ method
        # The decorator will handle exceptions
        return super().__getitem__(batch)


def name2collection(name, database):
    return database[name]


def create_client(worker_id, dbname, colname, mongohost):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    client = MongoClient("mongodb://" + mongohost + ":27017")
    dataset.collection = name2collection(colname, client[dbname])
