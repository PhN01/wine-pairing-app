from typing import Dict, List

import pandas as pd
import pydeck as pdk
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from app import constants as cons
from app.utils import get_best_wines, get_largest_outer_ring_polygon, load_file_data

st.set_page_config(page_title="Wine recommendations", layout="wide")

# ----------------------------------------------------------------------
# Data loaders
# ----------------------------------------------------------------------


@st.experimental_memo
def load_files():
    data = load_file_data()

    # grape data
    variety_df = data[cons.INPUT_FILE]

    # grape production breakdown
    country_bd_df = data[cons.COUNTRY_FILE]
    country_cols = list(country_bd_df.columns)[2:]
    country_bd_df["country_list"] = country_bd_df.apply(
        lambda row: list(row[country_cols].loc[~row[country_cols].isnull()].index),
        axis=1,
    )

    return variety_df, country_bd_df


@st.experimental_memo
def load_polygons():
    data = pd.read_json(cons.POLYGON_FILE)
    df = pd.DataFrame()
    df["country"] = data.features.apply(
        lambda row: cons.COUNTRY_MAPPING.get(
            row["properties"]["admin"], row["properties"]["admin"]
        )
    )
    df["coordinates"] = data.features.apply(
        lambda row: get_largest_outer_ring_polygon(row["geometry"]["coordinates"])
    )
    return df


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------


class App:
    layout: Dict[str, DeltaGenerator]

    def __init__(self) -> None:
        self.layout = self.set_layout()

    @staticmethod
    def set_layout() -> Dict[str, DeltaGenerator]:
        layout = {}
        c1, c2, c3 = [st.container(), st.container(), st.empty()]
        layout["header"] = c1
        layout["main_input"] = c2
        layout["main_content"] = c3

        with layout["main_input"]:
            text, buttons = st.columns([8, 2])
            layout["input_text"] = text
            layout["input_buttons"] = buttons

        with layout["main_content"].container():
            nav_col, space_col, cont1_col, cont2_col, cont3_col = st.columns(
                [2.5, 0.2, 2, 2, 4]
            )
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

        with self.layout["header"]:
            st.markdown(
                "We want to help you figure out, which wine works best with your meal plans. Tell us what you plan to eat:"
            )
        with self.layout["input_text"]:
            text = st.text_area("")
        with self.layout["input_buttons"]:
            # TODO: currently no option to align column contents, which
            # results in buttons not being aligned with text
            # streamlit team is working on it
            st.text("")
            st.text("")
            if st.button("Find wines") or s.pressed_first_button:
                s.pressed_first_button = True
                df = get_best_wines(
                    sentence=text,
                    ref_embeddings=variety_df[cons.MODEL_NAME],
                    df=variety_df,
                )
                wines_list = df.name.head(3).tolist()
                with self.layout["recommendations"]:
                    st.subheader("Recommended Wines:")
                    st.markdown(self._wines_to_md(wines_list))
                    st.selectbox(
                        "Select a wine to learn more...",
                        key="wine_sel",
                        options=[None] + wines_list,
                        on_change=self._update_wine_detail,
                        format_func=lambda x: x if x is not None else "--",
                    )
            if st.button("Clear outputs"):
                self.layout["main_content"].empty()

    @staticmethod
    def _wines_to_md(wine_names: List[str]) -> str:
        res = []
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
        profile = wine.loc[mapping.keys()].rename(index=lambda x: mapping[x]).squeeze()
        res = [f"**{k}:** {v}" for k, v in profile.items()]
        return "<br>".join(res)

    @staticmethod
    def _flavors_to_md(wine: pd.Series) -> str:
        flavor_cols = [f"flavor_{i}" for i in range(5)]

        res = []
        for flavor in wine.loc[flavor_cols].tolist():
            res.append(f"* {flavor}")
        return "\n".join(res)

    @staticmethod
    def _render_polygon_map(wine: pd.Series) -> None:
        global country_bd_df
        global polygon_df

        country_bd = country_bd_df.loc[country_bd_df.name == wine["name"]].squeeze()
        countries = country_bd["country_list"]
        polygons = polygon_df.loc[polygon_df.country.isin(countries)]
        polygons["prod_share"] = polygons.country.map(country_bd)

        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=pdk.ViewState(
                    latitude=19,
                    longitude=21,
                    zoom=0,
                    pitch=0,
                    # height="100%"
                ),
                # views=[pdk.View(type="MapView", controller=True, height="50%")],
                tooltip={
                    "html": "<b>Country:</b> {country} <br /><b>Share of global production:</b> {prod_share}%"
                },
                layers=[
                    pdk.Layer(
                        "PolygonLayer",
                        polygons,
                        id="geojson",
                        opacity=0.8,
                        stroked=False,
                        get_polygon="coordinates",
                        filled=True,
                        extruded=False,
                        wireframe=True,
                        get_fill_color=[80, 176, 207],
                        get_line_color=[80, 176, 207],
                        auto_highlight=True,
                        pickable=True,
                    ),
                ],
            )
        )

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
                st.subheader("Profile")
                st.markdown(self._profile_to_md(wine_data), unsafe_allow_html=True)
            with self.layout["flavors"]:
                st.subheader("Tastes like...")
                st.markdown(self._flavors_to_md(wine_data))
            with self.layout["regions"]:
                self.layout["regions"].empty()
                st.subheader("Where this wine is grown")
                self._render_polygon_map(wine_data)

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


# ----------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------

st.title("Wine pairing app")

variety_df, country_bd_df = load_files()
polygon_df = load_polygons()

app = App()
app.render_layout()
