#!/bin/sh

export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-5401}
export MAPBOX_TOKEN=${MAPBOX_TOKEN:-""}

# If there's a setup.sh script in the /app directory or other path specified, run it before starting
PRE_START_PATH=${PRE_START_PATH:-/app/setup.sh}
echo "Checking for script in $PRE_START_PATH"
if [ -f $PRE_START_PATH ] ; then
    echo "Running script $PRE_START_PATH"
    . "$PRE_START_PATH"
else
    echo "There is no script $PRE_START_PATH"
fi

# run streamlit app
exec streamlit run app/main.py
