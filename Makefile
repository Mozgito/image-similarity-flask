COMPOSE=docker compose -f docker-compose.yml

build:
	@echo "===> Building app"
	@$(COMPOSE) build

rebuild:
	@echo "===> Rebuilding app"
	@docker rmi image-similarity-flask-app
	@$(COMPOSE) build --no-cache

start:
	@echo "===> Running app"
	@$(COMPOSE) up -d

stop:
	@echo "===> Running app"
	@$(COMPOSE) stop

rm:
	@echo "===> Deleting container"
	@$(COMPOSE) rm -sf

cont:
	@docker exec -it similarity-flask bash

logs:
	@echo "===> Turn on logs"
	@$(COMPOSE) logs -f
