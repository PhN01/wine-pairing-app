# Wine pairing app
## Summary

This app proposes wine pairings based on the description of a meal. The meal description can be provided in the form of keywords, a single sentence description or even an entire recipe.

On top of the wine pairings, the app allows you to display additional information about a selected wine like its profile, what it tastes like and where it is produced.

The wine dataset was scraped and curated from different sources. For the food-wine pairings the a numeric representation of the provided meal description is computed and gets compared to the numerical representations of predefined food-wine pairings of each wine in the dataset. The app returns the three wines with food pairings most similar to the provided meal description. For more information see `How it works` section.

## How it works
The main analytical task of the app is to compute the numerical representation of the text input. For this purpose, I have deployed an NLP model from `huggingface` using FastAPI (not public). The API takes the text as input and returns a vector representation (embedding) extracted from the embedding layer of the NLP model.

To assess which wine works best with the provided food description, I compare the text embedding with the numerical representations of known, suitable food pairings for each wine using the cosine distance.

## Local development and testing
### Dependency management
This app uses `poetry` for dependency management. If you don't have `poetry` installed yet, install it by running `pip install poetry` in your terminal. Next, you can install the virtual environment by running in the repository's root directory
``` bash
poetry install
```
Once finished, you can activate the virtual environment by running
``` bash
source `poetry env info --path`/bin/activate
```
Finally, to deactivate the virtual environment, run `deactivate`. For more information about `poetry` see https://python-poetry.org/.

### Running the app
To run the steamlit app locally, `cd` into the first-level `app` directory and execute the following line while the virtual environment is activated
``` bash
streamlit run app/main.py
```
You should then be able to see this app in your browser under http://localhost:5401.

### Changing the code
I use pre-commit hooks to enforce good code quality before changes can be commited to the repository. For any code changes. If you don't have the pre-commit python package installed, run `pip install pre-commit`. To activate the pre-commit hooks specified in `.pre-commit-config.yaml`, run
``` bash
pre-commit install
```
Among other aspects and sanity checks, the preconfigured pre-commit hooks ensure that the code follows `flake8` standards. For more background on pre-commit hooks see https://pre-commit.com/.

## Deployment
The app can also be run as a docker container. Details of the setup are specified in the `Dockerfile`. Based on the docker file, the app can be deployed both locally and remotely. This documentation assumes that you have Docker Desktop installed and some familiarity with Docker. For more information see https://www.docker.com/.

### Local Deployment
For local deployment, we first need to build an image based on the `Dockerfile`. For this, we can run the following line in the repo root directory
``` bash
docker build -t wine-paring-app-local .
```
This creates an image with the tag `wine-pairing-app-local` in Docker Desktop. By default, the streamlit app runs on port 5401 of the container, hence when starting the container, we must expose the corresponding port:
``` bash
docker run -p 5401:5401 wine-pairing-app-local
```
The app will then be available under http://localhost:5401 in your browser.

### Remote Deployment
I chose to deploy my app to Heroku (https://www.heroku.com). To simplify building and deploying to Heroku, the necessary commands are provided in a Makefile. To be able to use the makefile, you need to make sure that your Heroku API key must be set as an environment variable called `HEROKU_API_KEY§`. After logging in to Heroku in your browser, you can find your Heroku API key under `Account settings` > `API Key` > `Reveal`.

To deploy the app to Heroku, we need first to run
``` bash
make build-app-heroku
```
which builds the app image. Please note that if you are not running the command on an `linux/amd64` platform the build process may take longer than the build for local deployment.

Next, push the image to the Heroku container repo by running
``` bash
make push-app-heroku
```

Finally, release the image with
``` bash
make release-heroku
```
