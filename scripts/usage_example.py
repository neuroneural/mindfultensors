import torch
from torch.utils.data import DataLoader
from mindfultensors.gencoords import CoordsGenerator
from mindfultensors.mongoloader import (
    create_client,
    collate_subcubes,
    mcollate,
    MBatchSampler,
    MongoDataset,
    MongoClient,
    mtransform,
)

device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

# dspecifying the database location and collection name
LABELNOW="50label"
MONGOHOST = "trendscn018.rs.gsu.edu"
DBNAME = 'MindfulTensors'
COLLECTION = "HCP"
# index field and labels to retrieve
INDEX_ID = "subject"
VIEWFIELDS = ["subdata", LABELNOW, "id", "subject"]
SAMPLES = 16 # subcubes per subject to sample
# percent of the data in a collection to use for validation
validation_percent = 0.1

# specify dimension of the larger volume
volume_shape = [256]*3
# specify dimension of the subvolume
subvolume_shape = [256]*3
coord_generator = CoordsGenerator(volume_shape, subvolume_shape)

# wrapper functions
def createclient(x):
    return create_client(x,
                         dbname=DBNAME,
                         colname=COLLECTION,
                         mongohost=MONGOHOST)

def mycollate_full(x):
    return mcollate(x, labelname=LABELNOW)

def mycollate(x):
    return collate_subcubes(x, coord_generator,
                            labelname=LABELNOW,
                            samples=SAMPLES)

def mytransform(x):
    return mtransform(x, label=LABELNOW)


client = MongoClient("mongodb://" + MONGOHOST + ":27017")
db = client[DBNAME]
posts = db[COLLECTION]
# compute how many unique INDEX_ID values are present in the collection
# these are unique subjects
num_examples = int(posts.find_one(sort=[(INDEX_ID, -1)])[INDEX_ID] + 1)

tdataset = MongoDataset(
    range(int((1 - validation_percent) * num_examples)),
    mytransform,
    None,
    id=INDEX_ID,
    fields=VIEWFIELDS,
    )

# We need a sampler that generates indices instead of trying to split the
# dataset into chunks
# use one subject at a time
tsampler = MBatchSampler(tdataset, batch_size=1)

# the standard pytorch class - ready to be used
tdataloader = DataLoader(
    tdataset,
    sampler=tsampler,
    collate_fn=mycollate_full,
    # if you want the loader to place batch on GPU and at a fixed location
    #pin_memory=True,
    worker_init_fn=createclient,
    num_workers=1, # currently does not work with <1
    )
