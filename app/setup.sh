#!/bin/sh

echo "[global]
developmentMode = false

[server]
port = $PORT
enableCORS = false
enableXsrfProtection = false
headless = true

[theme]
base = 'light'

[mapbox]
token = '${MAPBOX_TOKEN}'
" > .streamlit/config.toml
