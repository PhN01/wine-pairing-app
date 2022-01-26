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


st.title("Wine recommendation engine")

variety_df = load_data()
tokenizer, model = load_model()


def plotly_plot(df_row: pd.Series):
    scores = {
        "sweetness": df_row["sweetness_label"],
        "body": df_row["body_label"],
        "tannins": df_row["tannins_label"],
        "acidity": df_row["acidity_label"],
        "alcohol": df_row["alcohol_label"],
    }
    fig = go.Figure(
        data=go.Scatterpolar(
            r=list(scores.values()), theta=list(scores.keys()), fill="toself",
        )
    )
    fig.update_layout(polar=dict(radialaxis=dict(visible=True),), showlegend=False)
    return fig


c1, c2, c3 = [st.container(), st.container(), st.container()]
with c1:
    text = st.text_area("What are you eating?")
with c3:
    # wines = st.empty()
    c31, c32, c33 = [st.container(), st.container(), st.container()]
    with c31:
        c31_c1, c31_c2 = st.columns([2, 8])
        with c31_c1:
            top1_metric = st.empty()
        with c31_c2:
            top1_name = st.empty()
            top1_plot = st.empty()
    with c32:
        c32_c1, c32_c2 = st.columns([2, 8])
        with c32_c1:
            top2_metric = st.empty()
        with c32_c2:
            top2_name = st.empty()
            top2_plot = st.empty()
    with c33:
        c33_c1, c33_c2 = st.columns([2, 8])
        with c33_c1:
            top3_metric = st.empty()
        with c33_c2:
            top3_name = st.empty()
            top3_plot = st.empty()

top_n_mapping = {
    0: {"metric": top1_metric, "plot": top1_plot, "name": top1_name},
    1: {"metric": top2_metric, "plot": top2_plot, "name": top2_name},
    2: {"metric": top3_metric, "plot": top3_plot, "name": top3_name},
}

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
                top_n_mapping[i]["metric"].metric(
                    "Confidence", df.iloc[i]["similarity"]
                )
                top_n_mapping[i]["name"].markdown(f"### {df.iloc[i]['name']}")
                top_n_mapping[i]["plot"].plotly_chart(
                    plotly_plot(df.iloc[i]), use_cotainer_width=True
                )
            # wines.dataframe(df)
    with col2:
        if st.button("Clear"):
            for i in range(3):
                top_n_mapping[i]["name"].empty()
                top_n_mapping[i]["metric"].empty()
                top_n_mapping[i]["plot"].empty()
