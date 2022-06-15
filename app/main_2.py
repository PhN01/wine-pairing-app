import os
from typing import Dict, Tuple

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
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


def set_page_container_style():
    st.markdown(
        """
        <style>
               .css-18e3th9 {
                    padding-top: 1.5rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """,
        unsafe_allow_html=True,
    )


def create_wine_layout():
    layout_dict = {}
    cols = st.columns([2, 1, 1, 1, 1, 1, 1, 2])
    for i, col in enumerate(cols):
        with col:
            layout_dict[i] = st.empty()
    return layout_dict


def create_wine_layout() -> dict:
    return st.empty()


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
        s.pressed_first_button = False
        el.empty()


# -------------------------------------------------------------------
# Render site
# -------------------------------------------------------------------
def update_wine_detail(col_detail):
    global variety_df
    with col_detail:
        wine_name = st.session_state.wine_sel
        if wine_name is None:
            col_detail.empty()
        else:
            col_1, col_2, col_3 = st.columns([2, 2, 6])
            wine_data = variety_df.loc[variety_df.name==wine_name].squeeze()
            with col_1:
                st.markdown("## Profile")
                st.markdown(taste_profile_to_md(wine_data))
            with col_2:
                st.markdown("## Tastes like...")
                st.markdown(flavors_to_md(wine_data))


def select_wines(col_nav):
    df = get_best_wines(
        sentence=text,
        ref_embeddings=variety_df[cons.MODEL_NAME],
        tokenizer=tokenizer,
        model=model,
        df=variety_df,
    )
    wine_sel = st.selectbox(
        "Select a recommended wine",
        key="wine_sel",
        options=df.name.head(3),
        on_change=update_wine_detail,
        kwargs={"col_detail": col_detail},
    )


def wines_to_md(wine_names):
    res = ["**Recommended wines:**"]
    for wine in wine_names:
        res.append(f"* {wine}")
    return "\n".join(res)


def profile_to_md(wine: pd.Series) -> str:
    mapping = {
        "sweetness_label": "Sweetness",
        "body_label": "Body",
        "tannins_label": "Tannins",
        "acidity_label": "Acidity",
        "alcohol_label": "Alcohol",
    }
    taste = (
        wine.loc[mapping.keys()]
        .rename(wine["name"])
        .rename(index=lambda x: mapping[x])
        .squeeze()
    )
    return taste.to_markdown(index=True, tablefmt="grid")

def flavors_to_md(wine: pd.Series) -> str:
    flavor_cols = [f"flavor_{i}" for i in range(5)]

    res = []
    for flavor in wine.loc[flavor_cols].tolist():
        res.append(f"* {flavor}")
    return "\n".join(res)
    


def main():
    set_page_container_style()
    s = st.session_state
    if not s:
        s.pressed_first_button = False
        s.clear_hit = False

    c1, c2, c3 = [st.container(), st.empty(), st.empty()]
    with c1:
        st.markdown("We want to help you figure out, which wine works best with your meal plans. Tell us what you plant to eat:")
        text = st.text_area("")
    with c3.container():
        nav_col, content_col = st.columns([2, 8])
    with c2.container():
        col_1, col_2, col_3, col_4, col_5 = st.columns([1, 1.5, 8.5])
        with col_1:
            if st.button("Find wines") or s.pressed_first_button:
                s.pressed_first_button = True
                df = get_best_wines(
                    sentence=text,
                    ref_embeddings=variety_df[cons.MODEL_NAME],
                    tokenizer=tokenizer,
                    model=model,
                    df=variety_df,
                )
                wines_list = df.name.head(3).tolist()
                with nav_col:
                    st.markdown(wines_to_md(wines_list))
                    wine_sel = st.selectbox(
                        "Select a wine to learn more...",
                        key="wine_sel",
                        options=[None]+wines_list,
                        on_change=update_wine_detail,
                        kwargs={"col_detail": content_col},
                        format_func=lambda x: x if not x is None else "--"
                    )
        with col_2:
            if st.button("Clear outputs"):
                c3.empty()


main()
