import torch
from pymongo import MongoClient
from torch.utils.data import DataLoader

from mindfultensors.mongoloader import (
    MongoDataset,
    create_client,
    name2collection,
)
from mindfultensors.utils import insert_kind, unit_interval_normalize, DBBatchSampler

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------

MONGOHOST  = "localhost"
DBNAME     = "mindfultensors_sim"
COLLECTION = "sim_unified"
INDEX_ID   = "id"
NUM_SUBJECTS = 4
CHUNK_SIZE   = 1  # MB

# -----------------------------------------------------------------------
# Simulated subjects — fixed values for reproducibility
# Each subject has:
#   smri       — 3D tensor (8x8x8), tensor kind
#   falff      — 3D tensor (8x8x8), tensor kind
#   age        — float scalar
#   gender     — int scalar (0=M, 1=F)
#   site       — str scalar
#   is_control — bool scalar
# -----------------------------------------------------------------------

SUBJECTS = [
    {"id": 0, "age": 24.0, "gender": 0, "site": "site_A", "is_control": True},
    {"id": 1, "age": 31.0, "gender": 1, "site": "site_B", "is_control": False},
    {"id": 2, "age": 27.0, "gender": 0, "site": "site_A", "is_control": True},
    {"id": 3, "age": 45.0, "gender": 1, "site": "site_C", "is_control": False},
]

# -----------------------------------------------------------------------
# Populate: simulated data -> unified collection
# -----------------------------------------------------------------------

def populate(db):
    col = db[COLLECTION]
    col.drop()

    for subject in SUBJECTS:
        sid = subject["id"]
        print(f"  inserting subject {sid}")

        torch.manual_seed(sid)
        insert_kind(col, sid, "smri",       torch.rand(8, 8, 8), dtype="f32", id_field=INDEX_ID, chunk_size_mb=CHUNK_SIZE)
        insert_kind(col, sid, "falff",      torch.rand(8, 8, 8), dtype="f32", id_field=INDEX_ID, chunk_size_mb=CHUNK_SIZE)
        insert_kind(col, sid, "age",        subject["age"],        dtype="float", id_field=INDEX_ID)
        insert_kind(col, sid, "gender",     subject["gender"],     dtype="int",   id_field=INDEX_ID)
        insert_kind(col, sid, "site",       subject["site"],       dtype="str",   id_field=INDEX_ID)
        insert_kind(col, sid, "is_control", subject["is_control"], dtype="bool",  id_field=INDEX_ID)

    col.create_index([(INDEX_ID, 1), ("kind", 1)])
    print(f"done. total docs: {col.count_documents({})}")


# -----------------------------------------------------------------------
# Dataset + DataLoader
# -----------------------------------------------------------------------

client = MongoClient("mongodb://" + MONGOHOST + ":27017")
db     = client[DBNAME]
col    = name2collection(COLLECTION, db)


dataset = MongoDataset(
    indices    = [s["id"] for s in SUBJECTS],
    collection = col,
    fetch      = ("smri", "falff", "age", "gender", "is_control"),
    normalize  = unit_interval_normalize,
    id         = INDEX_ID,
)


def collate(results):
    results = results[0]

    print(f"\n--- before collate ---")
    for kind in next(iter(results.values())).keys():
        values = [results[i][kind] for i in results]
        if hasattr(values[0], "shape"):
            print(f"  {kind}: [{', '.join(str(v.shape) for v in values)}]")
        else:
            print(f"  {kind}: {values}")

    smri   = torch.stack([results[i]["smri"]  for i in results]).unsqueeze(1)
    falff  = torch.stack([results[i]["falff"] for i in results]).unsqueeze(1)
    age    = [results[i]["age"]        for i in results]
    gender = [results[i]["gender"]     for i in results]
    ctrl   = [results[i]["is_control"] for i in results]

    return {"smri": smri, "falff": falff, "age": age, "gender": gender, "is_control": ctrl}


def worker_init(worker_id):
    create_client(
        worker_id, dbname=DBNAME, colname=COLLECTION, mongohost=MONGOHOST
    )


dataloader = DataLoader(
    dataset,
    sampler    = DBBatchSampler(dataset, batch_size=2, seed=42),
    collate_fn = collate,
    num_workers = 0,
)


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    populate(db)

    print("\ncollection state before fetching:")
    for doc in col.find({}, {"_id": 0}):
        if "chunk" in doc:
            print(f"  {{id: {doc['id']}, kind: \"{doc['kind']}\", chunk_id: {doc['chunk_id']}, chunk: <Binary {len(doc['chunk'])} bytes>}}")
        else:
            print(f"  {{id: {doc['id']}, kind: \"{doc['kind']}\", value: {repr(doc['value'])}}}")

    print("\nsmoke test — fetching one batch ...")
    for batch in dataloader:
        print(f"\n--- batch ---")
        print(f"  smri       : {batch['smri'].shape}")
        print(f"  falff      : {batch['falff'].shape}")
        print(f"  age        : {batch['age']}")
        print(f"  gender     : {batch['gender']}")
        print(f"  is_control : {batch['is_control']}")
        break
