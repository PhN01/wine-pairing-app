from copy import deepcopy
from functools import partial
from typing import List, Union

import numpy as np
import pandas as pd
import requests
from scipy import spatial

from app import constants as cons


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
