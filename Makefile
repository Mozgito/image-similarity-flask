COMPOSE_APP=docker compose -f docker-compose-app.yml
COMPOSE_TOOLS=docker compose -f docker-compose-tools.yml

build:
	@echo "===> Building app"
	@$(COMPOSE_APP) build

rebuild:
	@echo "===> Rebuilding app"
	@docker rmi image-similarity-flask-app
	@$(COMPOSE_APP) build --no-cache

start:
	@echo "===> Running app container"
	@$(COMPOSE_APP) up -d

stop:
	@echo "===> Stopping app container"
	@$(COMPOSE_APP) stop

rm:
	@echo "===> Deleting app container"
	@$(COMPOSE_APP) rm -sf

cont:
	@docker exec -it similarity-flask bash

logs:
	@echo "===> Turn on app logs"
	@$(COMPOSE_APP) logs -f

build-tools:
	@echo "===> Building tools"
	@$(COMPOSE_TOOLS) build

calc-predictions:
	@echo "===> Calculating predictions"
	@$(COMPOSE_TOOLS) run --rm tools python calc_predictions.py
