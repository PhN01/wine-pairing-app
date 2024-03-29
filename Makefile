HEROKU_APP_NAME=wine-pairing-app-42
COMMIT_ID=$(shell git rev-parse HEAD)


heroku-login:
	HEROKU_API_KEY=${HEROKU_API_KEY} heroku auth:token

heroku-container-login:
	HEROKU_API_KEY=${HEROKU_API_KEY} heroku container:login

build-app-heroku: heroku-container-login
	docker build --no-cache -t registry.heroku.com/$(HEROKU_APP_NAME)/web . --platform linux/amd64

push-app-heroku: heroku-container-login
	docker push registry.heroku.com/$(HEROKU_APP_NAME)/web

release-heroku: heroku-container-login
	heroku container:release web --app $(HEROKU_APP_NAME)

heroku-logs: heroku-container-login
	heroku logs --tail --app $(HEROKU_APP_NAME)

.PHONY: heroku-login heroku-container-login build-app-heroku push-app-heroku
