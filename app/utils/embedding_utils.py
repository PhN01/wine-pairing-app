from typing import List, Union
from functools import partial
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from scipy import spatial
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling


def mean_pooling(
    model_output: BaseModelOutputWithPooling, attention_mask: torch.Tensor
):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def embed_sentences(
    sentences: List[str], tokenizer: PreTrainedTokenizer, model: PreTrainedModel
) -> np.array:
    encoded_sentences = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        model_output = model(**encoded_sentences)
    embeddings = mean_pooling(model_output, encoded_sentences["attention_mask"])
    return embeddings.numpy()


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
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    df: pd.DataFrame,
) -> pd.DataFrame:
    target_embedding = embed_sentences([sentence], tokenizer, model)
    closest = top_n_closest_items_idx(target_embedding, ref_embeddings)
    res_df = deepcopy(df.iloc[closest[:, 0]])
    res_df["similarity"] = closest[:, 1].round(2)
    return res_df
