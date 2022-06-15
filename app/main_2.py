import os
from typing import Dict, Tuple, List

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


class App:
    layout: Dict[str, DeltaGenerator]

    def __init__(self) -> None:
        self.layout = self.set_layout()

    @staticmethod
    def set_layout() -> Dict[str, DeltaGenerator]:
        layout = {}
        c1, c2, c3 = [st.container(), st.empty(), st.empty()]
        layout["main_input"] = c1
        layout["main_buttons"] = c2
        layout["main_content"] = c3

        with layout["main_buttons"].container():
            find_col, clear_col, _ = st.columns([1, 1.5, 8.5])
            layout["find_button"] = find_col
            layout["clear_button"] = clear_col

        with layout["main_content"].container():
            nav_col, cont1_col, cont2_col, cont3_col = st.columns([2, 2, 2, 4])
            layout["recommendations"] = nav_col
            layout["profile"] = cont1_col
            layout["flavors"] = cont2_col
            layout["regions"] = cont3_col
        
        return layout

    def render_layout(self) -> None:
        self._set_page_container_style()

        s = st.session_state
        if not s:
            s.pressed_first_button = False
            
        with self.layout["main_input"]:
            st.markdown("We want to help you figure out, which wine works best with your meal plans. Tell us what you plant to eat:")
            text = st.text_area("")
        with self.layout["main_buttons"].container():
            with self.layout["find_button"]:
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
                    with self.layout["recommendations"]:
                        st.markdown(self._wines_to_md(wines_list))
                        st.selectbox(
                            "Select a wine to learn more...",
                            key="wine_sel",
                            options=[None] + wines_list,
                            on_change=self._update_wine_detail,
                            format_func=lambda x: x if not x is None else "--"
                        )
            with self.layout["clear_button"]:
                if st.button("Clear outputs"):
                    self.layout["main_content"].empty()

    @staticmethod
    def _wines_to_md(wine_names: List[str]) -> str:
        res = ["**Recommended wines:**"]
        for wine in wine_names:
            res.append(f"* {wine}")
        return "\n".join(res)

    @staticmethod
    def _profile_to_md(wine: pd.Series) -> str:
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
        return taste.to_markdown(index=True)

    @staticmethod
    def _flavors_to_md(wine: pd.Series) -> str:
        flavor_cols = [f"flavor_{i}" for i in range(5)]

        res = []
        for flavor in wine.loc[flavor_cols].tolist():
            res.append(f"* {flavor}")
        return "\n".join(res)

    def _update_wine_detail(self):
        global variety_df
        wine_name = st.session_state.wine_sel
        if wine_name is None:
            self.layout["profile"].empty()
            self.layout["flavors"].empty()
            self.layout["regions"].empty()
        else:
            wine_data = variety_df.loc[variety_df.name == wine_name].squeeze()
            with self.layout["profile"]:
                st.markdown("## Profile")
                st.markdown(self._profile_to_md(wine_data))
            with self.layout["flavors"]:
                st.markdown("## Tastes like...")
                st.markdown(self._flavors_to_md(wine_data))

    @staticmethod
    def _set_page_container_style() -> None:
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

app = App()

app.render_layout()
