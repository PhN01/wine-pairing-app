"""
conftest.py
"""
from typing import List

import app.constants as cons
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def string_inputs_list() -> List[str]:
    return [
        "Example sentence 1",
        "Another example sentence",
        "3 is always better than 2",
    ]


@pytest.fixture
def rand_arr_vector_list() -> List[List[float]]:
    return [np.random.randn(cons.HIDDEM_DIM) for _ in range(10)]


@pytest.fixture
def example_wine_df() -> pd.DataFrame:
    data = []
    for i in range(10):
        data.append(
            {
                "wine_name": f"name_{i}",
                "taste_0": f"taste_0_{i}",
                "taste_1": f"taste_1_{i}",
                "profile_0": f"profile_0_{i}",
                "profile_1": f"profile_1_{i}",
            }
        )
    return pd.DataFrame(data)
