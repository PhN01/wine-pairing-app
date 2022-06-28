import os
import pickle
from copy import deepcopy
from functools import partial
from io import BytesIO
from typing import Any, List, Union

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
    headers = {
        "Content-type": "application/json",
        "Authorization": f"Bearer {cons.API_TOKEN}",
    }
    params = {"sentences": [sentence]}
    res = requests.post(url=cons.NLP_API_URL, headers=headers, json=params)
    # if res.status_code != 200:
    #     raise RuntimeError("Embedding request failed.")
    print(res)
    embedding = res.json()[0]["embedding"]
    return embedding


def cosine_sim(x1: Union[np.array, List[float]], x2: Union[np.array, List[float]]):
    return 1 - spatial.distance.cosine(x1, x2)


def top_n_closest_items_idx(
    target_embedding: np.array, ref_embeddings: np.array, top_n: int = 3
) -> np.array:
    func = partial(cosine_sim, x2=target_embedding)
    sim = np.array(list(map(func, ref_embeddings)))
    top_n_idx = sim.argsort()[-top_n:][::-1]
    return np.column_stack((top_n_idx, sim[top_n_idx]))


def get_best_wines(
    sentence: str,
    ref_embeddings: np.array,
    df: pd.DataFrame,
) -> pd.DataFrame:
    target_embedding = request_embedding(sentence)
    closest = top_n_closest_items_idx(np.array(target_embedding), ref_embeddings)
    res_df = deepcopy(df.iloc[closest[:, 0]])
    res_df["similarity"] = closest[:, 1].round(2)
    return res_df


# ----------------------------------------------------------------------
# GIS
# ----------------------------------------------------------------------


def get_polygon_square_area(coordinates: list) -> float:
    coordinates = np.array(coordinates)
    lon = [co[0] for co in coordinates]
    lat = [co[1] for co in coordinates]
    return (np.max(lon) - np.min(lon)) * (np.max(lat) - np.min(lat))


def change_lon_lat(coordinates: list) -> list:
    coordinates = np.array(coordinates)
    s = coordinates.shape
    assert len(s) >= 2 and s[-1] == 2
    if len(s) == 2:
        return coordinates[:, ::-1].tolist()
    elif len(s) == 3:
        return coordinates[:, :, ::-1].tolist()
    elif len(s) == 4:
        return coordinates[:, :, :, ::-1].tolist()


def change_lon_lat_any(coordinates: list) -> list:
    coordinates = np.array(coordinates)
    s = coordinates.shape
    if len(s) >= 2 and s[-1] == 2:
        return change_lon_lat(coordinates.tolist())
    else:
        co_out = []
        for i, i_co in enumerate(coordinates):
            i_co = np.array(i_co)
            s = i_co.shape
            if len(s) >= 2 and s[-1] == 2:
                co_out.append(change_lon_lat(i_co.tolist()))
            else:
                co_out.append([])
                for j, j_co in enumerate(i_co):
                    j_co = np.array(j_co)
                    s = j_co.shape
                    if len(s) >= 2 and s[-1] == 2:
                        co_out[i].append(change_lon_lat(j_co.tolist()))
                    else:
                        raise ValueError("Unparsable shape of input coordinates.")
        return co_out


def get_outer_ring_polygons(coordinates: list) -> list:
    coordinates = np.array(coordinates)

    out = []
    for i_coordinates in coordinates:
        i_coordinates = np.array(i_coordinates)
        while not (len(i_coordinates.shape) == 2 and i_coordinates.shape[-1] == 2):
            i_coordinates = np.array(i_coordinates[0])
        out.append(i_coordinates.tolist())
    return out


def get_largest_outer_ring_polygon(coordinates: list) -> list:
    coordinates = np.array(coordinates)

    largest_polygon_size = 0
    largest_polygon = None
    for i_coordinates in coordinates:
        i_coordinates = np.array(i_coordinates)
        while not (len(i_coordinates.shape) == 2 and i_coordinates.shape[-1] == 2):
            i_coordinates = np.array(i_coordinates[0])
        size = get_polygon_square_area(i_coordinates)
        if largest_polygon_size < size:
            largest_polygon = i_coordinates.tolist()
            largest_polygon_size = size
    return largest_polygon


# ----------------------------------------------------------------------
# Data handling
# ----------------------------------------------------------------------


def get_crypt() -> Fernet:
    return Fernet(cons.FILE_KEY)


def read_file(file: str) -> pd.DataFrame:
    ftype = file.split(".")[-1]
    if ftype == "csv":
        return pd.read_csv(file)
    elif ftype == "pkl":
        return pd.read_pickle(file)
    else:
        raise ValueError(f"Unknown file type {file}")


def encrypt_data(data: Any) -> str:
    fernet = get_crypt()
    encoded = fernet.encrypt(pickle.dumps(data))
    return encoded


def decrypt_data(encoded: str) -> Any:
    fernet = get_crypt()
    decoded = fernet.decrypt(encoded)
    dec_bytes = BytesIO(decoded)
    return pickle.load(dec_bytes)


def load_file_data() -> Any:
    with open(os.path.join(cons.DATA_PATH, cons.ENCRYPTED_DATA_FILE), "rb") as f:
        return decrypt_data(f.read())
