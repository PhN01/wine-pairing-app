import os
import pickle
from copy import deepcopy
from functools import partial
from io import BytesIO
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import requests
from cryptography.fernet import Fernet
from scipy import spatial

from app import constants as cons

# ----------------------------------------------------------------------
# Embeddings
# ----------------------------------------------------------------------


def request_embedding(sentence: str) -> List[float]:
    """Load an embedding vector for a string passed as input

    Args:
        sentence (str): Input sentence

    Returns:
        List[float]: Embedding vector
    """
    headers = {
        "Content-type": "application/json",
        "Authorization": f"Bearer {cons.API_TOKEN}",
    }
    params = {"sentences": [sentence]}
    res = requests.post(url=cons.NLP_API_URL, headers=headers, json=params)

    embedding = res.json()[0]["embedding"]
    return embedding


def cosine_sim(
    x1: Union[np.array, List[float]], x2: Union[np.array, List[float]]
) -> float:
    """Compute the cosine similarity between two input vectors.

    Args:
        x1 (Union[np.array, List[float]]): Input vector 1 to be compared with vector 2
        x2 (Union[np.array, List[float]]): Input vector 2 to be compared with vector 1

    Returns:
        float: Cosine similarity
    """
    return 1 - spatial.distance.cosine(x1, x2)


def top_n_closest_items_idx(
    target_embedding: np.array, ref_embeddings: np.array, top_n: int = 3
) -> np.array:
    """Compare a target embedding and list of reference embeddings
    and a list indices of the top n most similar items from the
    reference list.

    Args:
        target_embedding (np.array): Target embedding to compare all
            reference embeddings to
        ref_embeddings (np.array): List of reference embeddings
        top_n (int, optional): Number of most similar results to return. Defaults to 3.

    Returns:
        np.array: 2-dimensional numpy array. The values in the i-th row of the array
        represent the index of the i-th most similar item and its cosine similarity
        to the target embedding.
    """
    func = partial(cosine_sim, x2=target_embedding)
    sim = np.array(list(map(func, ref_embeddings)))
    top_n_idx = sim.argsort()[-top_n:][::-1]
    return np.column_stack((top_n_idx, sim[top_n_idx]))


def get_best_wines(
    sentence: str,
    ref_embeddings: np.array,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Wrapper that takes an input sentence (-> target embedding), an array of reference
    embeddings and a wine dataframe as input. The rows of the reference embeddings and the
    dataframe correspond to wine types. The reference embeddings were computed based on
    known food-wine pairings for each wine type. Row indices in the reference embedding array
    correspond to row indices of data in the dataframe.
    Computes a target embedding based on the input sentence and returns the rows of the dataframe
    that correspond to the 3 embeddings from the reference array that are most similar to the input
    sentence embedding.

    Args:
        sentence (str): Input sentence for which the target embedding is computed
        ref_embeddings (np.array): Array of reference embeddings that correspond to known food-wine
            pairings. Each represents a wine type
        df (pd.DataFrame): Wine dataframe for which row indices correspond to row indices in
            the `ref_embeddings` array

    Returns:
        pd.DataFrame: Wine data of top 3 wines for which food-wine pairing embedding is most similar
            to the input sentence embedding
    """
    target_embedding = request_embedding(sentence)
    closest = top_n_closest_items_idx(np.array(target_embedding), ref_embeddings)
    res_df = deepcopy(df.iloc[closest[:, 0]])
    res_df["similarity"] = closest[:, 1].round(2)
    return res_df


# ----------------------------------------------------------------------
# GIS
# ----------------------------------------------------------------------


def get_polygon_rectangle_area(coordinates: List[float]) -> float:
    """Fits a rectangle (exactly N/S and E/W aligned) around a polygon
    and computes the area of the rectangle.

    Args:
        coordinates (List[float]): Polygon coordinates

    Returns:
        float: Area of rectangle around polygon
    """
    coordinates = np.array(coordinates)
    lon = [co[0] for co in coordinates]
    lat = [co[1] for co in coordinates]
    return (np.max(lon) - np.min(lon)) * (np.max(lat) - np.min(lat))


def get_largest_outer_ring_polygon(coordinates: list) -> List[float]:
    """From a list of polygons, return the one with the largest outer rectangle area

    Args:
        coordinates (list): List of polygons

    Returns:
        List[float]: Largest with the largest outer rectangle
    """
    coordinates = np.array(coordinates)

    largest_polygon_size = 0
    largest_polygon = None
    for i_coordinates in coordinates:
        i_coordinates = np.array(i_coordinates)
        while not (len(i_coordinates.shape) == 2 and i_coordinates.shape[-1] == 2):
            i_coordinates = np.array(i_coordinates[0])
        size = get_polygon_rectangle_area(i_coordinates)
        if largest_polygon_size < size:
            largest_polygon = i_coordinates.tolist()
            largest_polygon_size = size
    return largest_polygon


# ----------------------------------------------------------------------
# Data handling
# ----------------------------------------------------------------------


def get_crypt() -> Fernet:
    """Get Fernet encryption engine with local key"""
    return Fernet(cons.FILE_KEY)


def read_file(filepath: str) -> pd.DataFrame:
    """Helper function for reading csv and pickle files

    Args:
        filepath (str): Path of file to be read

    Returns:
        pd.DataFrame: Data loaded from file
    """
    ext = filepath.split(".")[-1]
    if ext == "csv":
        return pd.read_csv(filepath)
    elif ext == "pkl":
        return pd.read_pickle(filepath)
    else:
        raise ValueError(f"Unknown file type {filepath}")


def encrypt_data(data: Any) -> str:
    """Pickle and encrypt a data object

    Args:
        data (Any): Data to be encrypted

    Returns:
        str: Encrypted file string
    """
    fernet = get_crypt()
    encoded = fernet.encrypt(pickle.dumps(data))
    return encoded


def decrypt_data(encoded: str) -> Any:
    """Decrypt a file string

    Args:
        encoded (str): Encrypted file string

    Returns:
        Any: Decrypted data
    """
    fernet = get_crypt()
    decoded = fernet.decrypt(encoded)
    dec_bytes = BytesIO(decoded)
    return pickle.load(dec_bytes)


def load_file_data() -> Dict[str, Any]:
    """Helper function to load the encrypted data used by this application

    Returns:
        Any: Dictionary mapping to decrypted data
    """
    with open(os.path.join(cons.DATA_PATH, cons.ENCRYPTED_DATA_FILE), "rb") as f:
        return decrypt_data(f.read())
