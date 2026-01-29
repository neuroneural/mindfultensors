from pymongo import MongoClient
from torch.utils.data import Dataset, get_worker_info
from torch.utils.data.sampler import Sampler
from torch import tensor, stack
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

__all__ = [
    "unit_interval_normalize",
    "qnormalize",
    "mtransform",
    "mcollate",
    "collate_subcubes",
    "subcube_list",
    "MongoDataset",
    "DBBatchSampler",
]


class MongoDataset(Dataset):
    """
    A dataset for fetching batches of records from a MongoDB
    """

    def __init__(
        self,
        indices,
        transform,
        collection,
        bin_sample = [],
        meta_sample = [],
        normalize=unit_interval_normalize,
        id="id",
    ):
        """Constructor

        :param indices: a set of indices to be extracted from the collection
        :param transform: a function to be applied to each extracted record
        :param collection: pymongo collection to be used
        :param bin_sample: list of fields to be fetched from 'bin' collection (stores chunked data, e.g., 'smri', 'dwi'),
        :param meta_sample: list of fields to be fetched from 'meta' collection (stores meta data, e.g., 'gender_encoded', 'modalities'),
        :param id: the field to be used as an index. The `indices` are values of this field
        :returns: an object of MongoDataset class

        """
        assert len(bin_sample) > 0 or len(meta_sample) > 0, "At least one of bin_sample or meta_sample must be non-empty"

        self.indices = indices
        self.transform = transform
        self.collection = collection
        # self.fields = {_: 1 for _ in self.fields} if fields is not None else {}
        self.fields = {"id": 1, "chunk": 1, "kind": 1, "chunk_id": 1}
        self.bin_sample = bin_sample
        self.meta_sample = meta_sample
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
        # data or label as specified by the sample parameter

        bin_is_empty = len(self.bin_sample) == 0
        meta_is_empty = len(self.meta_sample) == 0


        if not bin_is_empty:
            bin_samples = list(
                self.collection["bin"].find(
                    {
                        self.id: {"$in": [self.indices[_] for _ in batch]},
                        "kind": {"$in": self.bin_sample},
                    },
                    self.fields,
                )
            )
        if not meta_is_empty:
            meta_samples = list(
                self.collection["meta"].find(
                    {
                        self.id: {"$in": [self.indices[_] for _ in batch]},
                    },
                    list(self.meta_sample)+ [self.id],
                )
            )

        results = {}
        for id in batch:
            results[id] = {}

            # Proc bin
            if not bin_is_empty:
                bin_for_id = [
                    sample
                    for sample in bin_samples
                    if sample[self.id] == self.indices[id]
                ]

                for kind in self.bin_sample:
                    data = self.make_serial(bin_for_id, kind)
                    results[id][kind] = self.normalize(self.transform(data).float())

            # Proc meta
            if not meta_is_empty:
                meta_for_id = [
                    sample
                    for sample in meta_samples
                    if sample[self.id] == self.indices[id]
                ]

                assert len(meta_for_id) != 0, f"No meta entries found for id {id}"
                assert len(meta_for_id) < 2, f"More than one meta entry found for id {id}"
                meta_for_id = meta_for_id[0]

                for kind in self.meta_sample:
                    label = meta_for_id[kind]
                    try:
                        label = tensor(label)
                    except Exception as e:
                        # Can't tensor-ize, raise error with details
                        raise ValueError(f"Cannot convert label for kind '{kind}' and id '{id}' to tensor. Value: {label}. Original error: {e}")
                    if label.ndim == 0:
                        label = label.unsqueeze(0)
                        
                    results[id][kind] = label

        return results
    
    def default_collate(self, results):
        """
        Returns a collated batch from results fetched by __getitem__.
        The order of outputs corresponds to the order of fields in self.bin_sample+self.meta_sample.
        Not guaranteed to work for any self.bin_sample or self.meta_sample, know what you're fetching.
        """
        results = results[0] # because something wraps outputs in a list? mystery...

        output = []
        for kind in list(self.bin_sample):
            output.append(stack([results[id_][kind] for id_ in results.keys()]).unsqueeze(1))
        
        for kind in list(self.meta_sample):
            output.append(stack([results[id_][kind] for id_ in results.keys()]).long())
        
        return tuple(output)


class MongoheadDataset(MongoDataset):
    def __init__(self, *args, keeptrying=True, **kwargs):
        """Constructor

        :param indices: a set of indices to be extracted from the collection
        :param transform: a function to be applied to each extracted record
        :param collection: pymongo collection to be used
        :param sample: a pair of fields to be fetched as `input` and `label`, e.g. (`T1`, `label104`)
        :param id: the field to be used as an index. The `indices` are values of this field
        :param keeptrying: whether to keep retrying to fetch a record if the process failed or just report this and fail
        :returns: an object of MongoDataset class

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


def name2collections(name: str, database):
    collection_bin = database[f"{name}.bin"]
    collection_meta = database[f"{name}.meta"]
    return collection_bin, collection_meta


def create_client(worker_id, dbname, colname, mongohost):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    client = MongoClient("mongodb://" + mongohost + ":27017")
    colbin, colmeta = name2collections(colname, client[dbname])
    dataset.collection = {"bin": colbin, "meta": colmeta}
