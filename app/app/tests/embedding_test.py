import logging

import app.constants as cons
import numpy as np
from app.utils.embedding import (
    cosine_sim,
    get_best_wines,
    request_embedding,
    top_n_closest_items_idx,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def test_request_single_embedding(string_inputs_list):
    res = request_embedding(string_inputs_list[0])
    assert len(res) == cons.HIDDEM_DIM
    assert not any(np.isnan(res))


def test_cosine_sim():
    a = [-1] * 10
    b = [1] * 10
    assert cosine_sim(a, a) == cosine_sim(b, b) == 1
    assert cosine_sim(a, b) == -1


def test_cosine_sim_arr(rand_arr_vector_list):
    res = []
    for _ in range(10):
        a, b = [
            rand_arr_vector_list[i]
            for i in np.random.choice(len(rand_arr_vector_list), size=2, replace=False)
        ]
        res.append(cosine_sim(a, b))
    assert all([-1 <= x <= 1 for x in res])
    assert not any(np.isnan(res))


def test_cosine_sim_list(rand_arr_vector_list):
    res = []
    for _ in range(10):
        a, b = [
            rand_arr_vector_list[i].tolist()
            for i in np.random.choice(len(rand_arr_vector_list), size=2, replace=False)
        ]
        res.append(cosine_sim(a, b))
    assert all([-1 <= x <= 1 for x in res])
    assert not any(np.isnan(res))


def test_top_n_closest_items_idx(rand_arr_vector_list):
    res = []
    for i in range(10):
        target = rand_arr_vector_list[i]
        res.append(
            top_n_closest_items_idx(target, np.array(rand_arr_vector_list), top_n=i)
        )
    assert [len(x) == i for i, x in enumerate(res)]
    assert np.isnan(np.concatenate(res, axis=0)).sum() == 0


def test_get_best_wines(
    string_inputs_list,
    rand_arr_vector_list,
    example_wine_df,
):
    for input in string_inputs_list:
        df = get_best_wines(input, np.array(rand_arr_vector_list), example_wine_df)
        assert len(df) == 3
