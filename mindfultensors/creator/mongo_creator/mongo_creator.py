import bson
import io
import numpy as np
import os
import pandas as pd
from typing import Dict, List, Callable, Optional, Any, Tuple
from tqdm import tqdm

import nibabel as nib
from pymongo import MongoClient, ASCENDING
import torch
from torch import Tensor


from mindfultensors.utils import unit_interval_normalize as normalize
from mindfultensors.creator.base_db_creator import BaseDBCreator


class MongoDBCreator(BaseDBCreator):
    """
    MongoDB Creator class
    """

    def __init__(
        self,
        database_name: str,
        collection_name: str,
        host: str,
        port: int,
        preprocessing_functions: Optional[Dict[str, Callable]] = None,
        chunk_size: int = 10,
    ) -> BaseDBCreator:
        """
        Constructor

        Args:
        database_name: str: name of the database
        collection_name: str: name of the collection
        host: str: host name
        port: int: port number
        preprocessing_functions: Optional[Dict[str, Callable]]: dictionary of preprocessing functions
        chunk_size: int: size of the chunk

        Returns:
        BaseDBCreator: an object of MongoDBCreator class
        """
        super().__init__()
        self._database_name = database_name
        self._collection_name = collection_name
        self._host = host
        self._port = port
        self._url = f"mongodb://{self._host}:{self._port}"
        self._preprocessing_functions = preprocessing_functions
        self._chunk_size = chunk_size
        self._client = None

    def connect(self) -> None:
        """
        Connects to the database
        """
        self._client = MongoClient(self._url)
        self._database = self._client[self._database_name]
        self._collection_bin = self._database[f"{self._collection_name}.bin"]
        self._collection_meta = self._database[f"{self._collection_name}.meta"]

    def write(
        self,
        data: pd.DataFrame,
        input_columns: List[str],
        label_columns: List[str],
        meta_columns: List[str],
        label_description: Optional[Dict[str, str]] = None,
        *args, **kwargs
    ) -> None:
        """
        Writes the data

        Args:
        data: pd.DataFrame: data
        input_columns: List[str]: list of input columns
        label_columns: List[str]: list of label columns
        meta_columns: List[str]: list of meta columns
        label_description: Optional[Dict[str, str]: dictionary of label description

        Returns:
        None
        """
        self._insert_samples(
            data, 
            input_columns,
            label_columns,
            meta_columns,
            label_description=label_description
        )
        index_name_meta = self._collection_meta.create_index([("id", ASCENDING)])
        index_name_bin = self._collection_bin.create_index([("id", ASCENDING)])

    def clean(self) -> None:
        """
        Cleans the database

        Args:
        None

        Returns:
        None
        """
        self._collection_bin.drop()
        self._collection_meta.drop()

    def close(self) -> None:
        """
        Closes the database connection

        Args:
        None

        Returns:
        None
        """
        self._client.close()

    def _insert_samples(
        self,
        data: pd.DataFrame,
        input_columns: List[str],
        label_columns: List[str],
        meta_columns: List[str],
        label_description: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Insert samples

        Args:
        data: pd.DataFrame: data
        label_description: Optional[Dict[str, str]]: dictionary of label description

        Returns:
        None
        """
        selected_columns = input_columns + label_columns + meta_columns
        for index in tqdm(data.index):
            meta_data = {"id": index, "labels": {}}
            for column in selected_columns:
                shape = None
                value = data[column].iloc[index]
                if column in meta_columns:
                    meta_data[column] = str(value)
                else:
                    shape = self._insert_data(column, value, index)
                    if "shape" not in meta_data:
                        meta_data["shape"] = shape
                    else:
                        assert meta_data["shape"] == shape
                    if column in label_columns:
                        if column in label_description:
                            meta_data["labels"][
                                column] = label_description[column]
                        else:
                            meta_data["labels"][
                                column] = "Label is not described"
            self._collection_meta.insert_one(meta_data)

    def _insert_data(
        self,
        column: str,
        filename: str,
        index: int
    ) -> Tuple[int]:
        """
        Insert data

        Args:
        column: str: column
        filename: str: filename
        index: int: index

        Returns:
        shape: Tuple[int]: shape
        """
        tensor_data = self._data_filename_2_tensor(filename)
        shape = tensor_data.shape
        if column in self._preprocessing_functions:
            tensor_data = self._preprocessing_functions[column](tensor_data)
        tensor_data = self._tensor_2_bin(tensor_data)
        # write data
        for chunk in self._chunk_binobj(tensor_data, index, column, self._chunk_size):
            self._collection_bin.insert_one(chunk)
        return shape

    @staticmethod
    def _data_filename_2_tensor(filename: str) -> Tensor:
        assert os.path.exists(filename)
        return torch.from_numpy(np.asanyarray(nib.load(filename).dataobj))

    @staticmethod
    def _tensor_2_bin(tensor: Tensor) -> bytes:
        """
        Convert tensor to binary

        Args:
        tensor: Tensor: tensor

        Returns:
        tensor_binary: binary
        """
        tensor_1d = tensor.to(torch.uint8)
        # Serialize tensor and get binary
        buffer = io.BytesIO()
        torch.save(tensor_1d, buffer)
        tensor_binary = buffer.getvalue()
        return tensor_binary

    @staticmethod
    def _chunk_binobj(
        tensor_compressed: Tensor,
        id: int,
        kind: str,
        chunksize: int
    ) -> Dict[str, Any]:
        """
        Chunk the binary object

        Args:
        tensor_compressed: Tensor: compressed tensor
        id: int: id
        kind: str: kind
        chunksize: int: chunk size

        Returns:
        Dict[str, Any]: dictionary of chunk
        """
        # Convert chunksize from megabytes to bytes
        chunksize_bytes = chunksize * 1024 * 1024

        # Calculate the number of chunks
        num_chunks = len(tensor_compressed) // chunksize_bytes
        if len(tensor_compressed) % chunksize_bytes != 0:
            num_chunks += 1

        # Yield chunks
        for i in range(num_chunks):
            start = i * chunksize_bytes
            end = min((i + 1) * chunksize_bytes, len(tensor_compressed))
            chunk = tensor_compressed[start:end]
            yield {
                "id": id,
                "chunk_id": i,
                "kind": kind,
                "chunk": bson.Binary(chunk),
            }


if __name__ == "__main__":

    # Example usage
    database_name = "mydatabase"
    collection_name = "mycollection"
    mongo_host = "10.245.12.58"
    mongo_port = "27017"
    metadata = pd.DataFrame(
        columns=['t1', 't2', 'subject_id', 'age', 'gender']
    )
    metadata.loc[metadata.shape[0]] = [
        './test_data/Template-T1-U8-RALPFH-BR.nii.gz',
        './test_data/Template-T2-U8-RALPFH-BR.nii.gz',
        1, 25, 'M'
    ]
    creator = MongoDBCreator(
        database_name=database_name,
        collection_name=collection_name,
        host=mongo_host,
        port=mongo_port,
        preprocessing_functions={
            't1': lambda x: normalize(x) * 256,
            't2': lambda x: normalize(x) * 256,
        },
        chunk_size=10,
    )
    creator.connect()
    creator.write(
        data=metadata,
        input_columns=['t1'],
        label_columns=['t2'],
        meta_columns=['subject_id', 'age', 'gender'],
        label_description={"mask": "Lesion mask"}
    )
    creator.clean()
    creator.close()
