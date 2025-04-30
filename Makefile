include .env
export $(shell sed 's/=.*//' .env)

IMAGE_NAME=$(DOCKER_USERNAME)/$(APP_NAME)

.PHONY: build run logs test

build:
	docker stop $(IMAGE_NAME) || true
	docker rm $(IMAGE_NAME) || true
	docker build --build-arg PORT=$(APP_PORT) --build-arg ENVIRONMENT=$(ENVIRONMENT) -t $(IMAGE_NAME) .

run:
	docker run -d \
		--name $(APP_NAME) \
		-v $(shell pwd)/app:/code/app \
		-v $(shell pwd)/tests:/code/tests \
		-v $(shell pwd)/.env:/code/.env \
		-v $(shell pwd)/pytest.ini:/code/pytest.ini \
		-p $(APP_PORT):$(CONTAINER_PORT) \
		-e ENVIRONMENT=$(ENVIRONMENT) \
		$(IMAGE_NAME)

logs:
	docker logs -f $(APP_NAME)

test:
	docker exec $(APP_NAME) pytest -v
