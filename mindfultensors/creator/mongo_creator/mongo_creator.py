import pandas as pd
from typing import Dict, List, Callable, Optional, Any, Tuple

from pymongo import MongoClient, ASCENDING


from mindfultensors.utils import unit_interval_normalize as normalize
from mindfultensors.creator.base_db_creator import BaseDBCreator

from utils import insert_samples


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
        insert_samples(
            data, 
            input_columns,
            label_columns,
            meta_columns,
            self._collection_bin,
            self._collection_meta,
            label_description=label_description,
            chunk_size=self._chunk_size,
            preprocessing_functions=self._preprocessing_functions
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
