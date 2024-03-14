from pymongo import MongoClient
from torch.utils.data import Dataset, get_worker_info
from torch.utils.data.sampler import Sampler

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
        sample,
        normalize=unit_interval_normalize,
        id="id",
        persist_on_EOF=False,
    ):
        """Constructor

        :param indices: a set of indices to be extracted from the collection
        :param transform: a function to be applied to each extracted record
        :param collection: pymongo collection to be used
        :param sample: a pair of fields to be fetched as `input` and `label`, e.g. (`T1`, `label104`)
        :param normalize: a function to apply to the input
        :param id: the field to be used as an index. The `indices` are values of this field
        :param persist_on_EOF: whether to keep trying if deserialization fails
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
        self.keeptyring = persist_on_EOF

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

    def retry_on_eof_error(self, retry_count):
        def decorator(func):
            def wrapper(self, batch, *args, **kwargs):
                for _ in range(retry_count):
                    try:
                        return func(self, batch, *args, **kwargs)
                    except EOFError as e:
                        if self.keeptyring:
                            print(
                                f"EOFError caught. Retrying {_+1}/{retry_count}"
                            )
                            continue
                        else:
                            raise e
                raise EOFError("Failed after multiple retries.")

            return wrapper

        return decorator

    @retry_on_eof_error(retry_count=3)  # Retry up to 3 times
    def __getitem__(self, batch):
        # Fetch all samples for ids in the batch and where 'kind' is either
        # data or label as specified by the sample parameter
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
