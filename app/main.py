import os
from typing import Tuple

import pandas as pd
import streamlit as st
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoModel, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

import app.constants.constants as cons
from app.utils.embedding_utils import get_best_wines

st.set_page_config(page_title="Wine recommendations", layout="wide")


@st.experimental_memo
def load_data():
    df = pd.read_pickle(os.path.join(cons.DATA_PATH, cons.INPUT_FILE))
    return df


@st.experimental_singleton
def load_model() -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    tokenizer = AutoTokenizer.from_pretrained(cons.MODEL_NAME)
    model = AutoModel.from_pretrained(cons.MODEL_NAME)
    return tokenizer, model


st.title("Wine recommendation engine")

variety_df = load_data()
tokenizer, model = load_model()

# -------------------------------------------------------------------
# Page layout
# -------------------------------------------------------------------


def create_wine_layout():
    layout_dict = {}
    cols = st.columns([2, 1, 1, 1, 1, 1, 1, 2])
    for i, col in enumerate(cols):
        with col:
            layout_dict[i] = st.empty()
    return layout_dict


def create_wine_list_layout(n_rows: int) -> dict:
    rows = {i: st.container() for i in range(n_rows)}
    layout_dict = {}
    for i, r in rows.items():
        with r:
            layout_dict[i] = create_wine_layout()
    return layout_dict

# -------------------------------------------------------------------
# Rendering helpers
# -------------------------------------------------------------------


def profile_value_to_md(name: str, val: int) -> str:
    return f"{name}<br> __{val}__/5"


def wine_falvors_to_md(wine_data: pd.Series):
    return f"__Tastes like:__ {wine_data['flavor_0']}, {wine_data['flavor_1']} and {wine_data['flavor_2']}."


def render_wine(wine_data: pd.Series, layout_dict: dict) -> None:
    layout_dict[0].markdown(f"### {wine_data['name']}")
    layout_dict[1].metric("Confidence", wine_data["similarity"])
    for i, item in enumerate(["Sweetness", "Body", "Tannins", "Acidity", "Alcohol"]):
        layout_dict[i + 2].markdown(
            profile_value_to_md(item, wine_data[f"{item.lower()}_value"]),
            unsafe_allow_html=True,
        )
    layout_dict[7].markdown(wine_falvors_to_md(wine_data))


def clear_wine_list(layout_dict: dict) -> None:
    for el in layout_dict.values():
        el.empty()

# -------------------------------------------------------------------
# Render site
# -------------------------------------------------------------------


c1, c2, c3 = [st.container(), st.container(), st.container()]
with c1:
    text = st.text_area("What are you eating?")
with c3:
    wines_layout = create_wine_list_layout(cons.NUM_WINES)
with c2:
    col1, col2, col3 = st.columns([3, 1, 7])
    with col1:
        if st.button("Find wines..."):
            df = get_best_wines(
                sentence=text,
                ref_embeddings=variety_df[cons.MODEL_NAME],
                tokenizer=tokenizer,
                model=model,
                df=variety_df,
            )
            for i in range(3):
                render_wine(df.iloc[i], wines_layout[i])
    with col2:
        if st.button("Clear"):
            for i in range(3):
                clear_wine_list(wines_layout[i])
