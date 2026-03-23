PROJECT_NAME := magic-card-finder

.PHONY: help \
        build up down logs ps reset-db \
        backend-shell frontend-shell \
        test-backend test-frontend test \
        lint-backend lint-frontend lint \
        fmt-backend fmt-frontend fmt \
        clean

help:
	@echo "Make targets for $(PROJECT_NAME):"
	@echo "  build           Build all Docker images"
	@echo "  up              Start stack with docker compose"
	@echo "  down            Stop stack and remove containers"
	@echo "  reset-db        Drop and recreate the magic_cards database in the running db container"
	@echo "  logs            Tail logs for all services"
	@echo "  ps              Show docker compose services status"
	@echo "  backend-shell   Open a shell inside the backend container"
	@echo "  frontend-shell  Open a shell inside the frontend container"
	@echo "  test-backend    Run backend tests (local python -m pytest)"
	@echo "  test-detector   Run detector service tests (local pytest)"
	@echo "  test-ocr        Run OCR service tests (local pytest)"
	@echo "  test-frontend   Run frontend tests (local npm / vitest)"
	@echo "  test            Run all tests"
	@echo "  lint-backend    Lint backend (if tooling installed)"
	@echo "  lint-frontend   Lint frontend (if tooling installed)"
	@echo "  lint            Run all linters"
	@echo "  fmt-backend     Format backend code (if tooling installed)"
	@echo "  fmt-frontend    Format frontend code (if tooling installed)"
	@echo "  fmt             Format backend and frontend"
	@echo "  clean           Remove build artifacts (node_modules, pycache)"

build:
	docker compose build --no-cache

start:
	docker compose -f docker-compose.dev.yml up --build

stop:
	docker compose down --remove-orphans

start-prod:
	docker compose -f docker-compose.prod.yml up --build

reset-db:
	docker compose exec db psql -U postgres -d postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'magic_cards';" || true
	docker compose exec db psql -U postgres -d postgres -c "DROP DATABASE IF EXISTS magic_cards;"
	docker compose exec db psql -U postgres -d postgres -c "CREATE DATABASE magic_cards;"

logs:
	docker compose logs -f

ps:
	docker compose ps

backend-shell:
	docker compose exec backend /bin/bash

frontend-shell:
	docker compose exec frontend /bin/bash

test-backend:
	@cd backend && \
	if [ ! -d ".venv" ]; then \
	  echo "Creating backend virtualenv in backend/.venv"; \
	  python -m venv .venv; \
	  . .venv/bin/activate && pip install --upgrade pip && pip install ".[dev]"; \
	fi && \
	. .venv/bin/activate && python -m pytest

test-detector:
	@cd detector_service && \
	if [ ! -d ".venv" ]; then \
	  echo "Creating detector_service virtualenv"; \
	  python -m venv .venv; \
	  . .venv/bin/activate && pip install --upgrade pip && pip install ".[dev]"; \
	fi && \
	. .venv/bin/activate && python -m pytest

test-ocr:
	@cd ocr_service && \
	if [ ! -d ".venv" ]; then \
	  echo "Creating ocr_service virtualenv"; \
	  python -m venv .venv; \
	  . .venv/bin/activate && pip install --upgrade pip && pip install ".[dev]"; \
	fi && \
	. .venv/bin/activate && python -m pytest

test-frontend:
	cd frontend && npm run test:ci

test: test-backend test-detector test-ocr test-frontend

lint-backend:
	@if command -v ruff >/dev/null 2>&1; then \
	  cd backend && ruff check . ; \
	else \
	  echo "ruff not installed; skipping backend lint."; \
	fi

lint-frontend:
	@if [ -f frontend/package.json ]; then \
	  cd frontend && npm run lint || echo "frontend lint script failed or not defined."; \
	else \
	  echo "No frontend/package.json; skipping frontend lint."; \
	fi

lint: lint-backend lint-frontend

fmt-backend:
	@if command -v black >/dev/null 2>&1; then \
	  cd backend && black . ; \
	else \
	  echo "black not installed; skipping backend formatting."; \
	fi

fmt-frontend:
	@if [ -f frontend/package.json ]; then \
	  cd frontend && npm run format || echo "frontend format script failed or not defined."; \
	else \
	  echo "No frontend/package.json; skipping frontend formatting."; \
	fi

fmt: fmt-backend fmt-frontend

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	if [ -d frontend/node_modules ]; then rm -rf frontend/node_modules; fi

