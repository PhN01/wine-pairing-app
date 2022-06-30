import os

from dotenv import load_dotenv

load_dotenv()


# data
DATA_PATH = "data"
INPUT_FILE = "20220124_wine_recommendation_input.pkl"
POLYGON_FILE = "https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_110m_admin_0_countries.geojson"
COUNTRY_FILE = "20220615_grape_regions.csv"
ENCRYPTED_DATA_FILE = "data.pkl"

# nlp api
NLP_API_URL = os.getenv("NLP_API_URL")
API_TOKEN = os.getenv("NLP_API_TOKEN")
MODEL_NAME = "nreimers/albert-small-v2"

# encryption key
FILE_KEY = os.getenv("FILE_KEY")

# content
NUM_WINES = 3
COUNTRY_MAPPING = {"United States of America": "USA"}
