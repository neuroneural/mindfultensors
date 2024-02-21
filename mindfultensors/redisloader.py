import pickle as pkl
from redis import Redis

from torch.utils.data import Dataset, get_worker_info

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
    "RedisDataset",
    "DBBatchSampler",
]


class RedisDataset(Dataset):
    """
    A dataset for fetching batches of records from a MongoDB
    """

    def __init__(
        self,
        indices,
        transform,
        dbkey,
        normalize=unit_interval_normalize,
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
        self.Redis = None
        self.dbkey = dbkey
        self.normalize = normalize

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, batch):
        # Fetch all samples for ids in the batch and where 'kind' is either
        # data or label as specified by the sample parameter

        results = {}
        for id in batch:
            # Separate samples for this id

            # Separate processing for each 'kind'
            payload = pkl.loads(self.Redis.brpoplpush(self.dbkey, self.dbkey))
            data = payload[0]
            label = payload[1]

            # Add to results
            results[id] = {
                "input": self.normalize(self.transform(data).float()),
                "label": self.transform(label),
            }

        return results


def create_client(worker_id, redishost):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    client = Redis(host=redishost)
    dataset.Redis = client
