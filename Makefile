run-docker-compose:
	uv sync --all-packages
	docker compose up --build