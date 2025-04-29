## Setup

1. Create a `.env` by cloning the `.env.example`.
2. Run `make build` to build the Docker image
3. Run `make run` to start the app

## Testing

To run the tests, run `make test`. The tests will be run in a Docker container.

## Docker

To build the Docker image, use `make build`. To start the app in a Docker container, use `make run`.

## Logging

To check application logs, run `make logs`.

## Code Structure

The app is structured as follows:

* `app` - the app code
	+ `dto` - data transfer objects (e.g. request and response bodies)
	+ `provider` - providers for the app (e.g. database, LLM)
	+ `router` - the API router
	+ `service` - services for the app (e.g. inference service)
	+ `utils` - utility functions
* `tests` - the tests
	+ `mocks` - mock objects for the tests
	+ `service` - tests for the services
	+ `utils` - tests for the utility functions
* `Makefile` - the Makefile for building and running the app
* `requirements.txt` - the dependencies for the app
* `README.md` - this file
