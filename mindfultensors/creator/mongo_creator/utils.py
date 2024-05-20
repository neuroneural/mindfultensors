import bson
import io
import numpy as np
import nibabel as nib
import os
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Callable, Optional, Any, Tuple

from pymongo.collection import Collection

import torch
from torch import Tensor


def tensor_2_bin(tensor: Tensor) -> bytes:
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


def chunk_binobj(
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


def nifti_filename_2_tensor(filename: str) -> Tensor:
    """
    Convert NIFTI filename to tensor

    Args:
    filename: str: filename of NIFTI file

    Returns:
    Tensor: tensor
    """
    assert os.path.exists(filename)
    assert filename.endswith(".nii") or filename.endswith(".nii.gz")
    return torch.from_numpy(np.asanyarray(nib.load(filename).get_fdata()))


def insert_data(
    column: str,
    filename: str,
    index: int,
    collection_bin: Collection,
    chunk_size: int = 10,
    preprocessing_functions: Optional[Dict[str, Callable]] = None,
) -> Tuple[int]:
    """
    Insert data

    Args:
    column: str: column
    filename: str: filename
    index: int: index
    collection_bin: Collection: collection bin
    chunk_size: int: chunk size
    preprocessing_functions: Optional[Dict[str, Callable]]: dictionary of preprocessing functions

    Returns:
    shape: Tuple[int]: shape
    """
    tensor_data = nifti_filename_2_tensor(filename)
    shape = tensor_data.shape
    if preprocessing_functions and column in preprocessing_functions:
        tensor_data = preprocessing_functions[column](tensor_data)
    tensor_data = tensor_2_bin(tensor_data)
    # write data
    for chunk in chunk_binobj(tensor_data, index, column, chunk_size):
        collection_bin.insert_one(chunk)
    return shape


def insert_samples(
    data: pd.DataFrame,
    input_columns: List[str],
    label_columns: List[str],
    meta_columns: List[str],
    collection_bin: Collection,
    collection_meta: Collection,
    label_description: Optional[Dict[str, str]] = None,
    chunk_size: int = 10,
    preprocessing_functions: Optional[Dict[str, Callable]] = None,
) -> None:
    """
    Insert samples

    Args:
    data: pd.DataFrame: data
    input_columns: List[str]: list of input columns
    label_columns: List[str]: list of label columns
    meta_columns: List[str]: list of meta columns
    collection_bin: Collection: collection bin
    collection_meta: Collection: collection meta
    label_description: Optional[Dict[str, str]]: dictionary of label description
    chunk_size: int: chunk size
    preprocessing_functions: Optional[Dict[str, Callable]]: dictionary of preprocessing functions

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
                shape = insert_data(
                    column, value, index,
                    collection_bin, chunk_size, preprocessing_functions=preprocessing_functions
                )
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
        collection_meta.insert_one(meta_data)
