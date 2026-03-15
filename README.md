Magic Card Finder
==================

This project scans images or marketplace listings containing Magic: The Gathering cards, identifies each card, and aggregates pricing information from multiple data sources.

## Tech Stack (Concrete Choices)

- **Backend**: Python + **FastAPI** (async, automatic OpenAPI docs), run with Uvicorn.
- **Image processing**: OpenCV for card outline detection, perspective correction, and cropping.
- **Vision model**: **Llama 3.2 Vision** served via **Ollama** running on the **Windows 11 host** by default, configurable via environment variable `OLLAMA_HOST`. You can later switch to an MTG-specific open-source model behind the same interface.
- **External APIs**:
  - **Scryfall**: canonical card data and baseline prices.
  - **Open TCG API (tcgtracking.com)**: optional additional pricing source.
  - **Cardmarket**: trend and condition-based prices (requires API key provided via env).
- **Database**: **Postgres** (via Docker) for caching card metadata, prices, and analysis runs. SQLite could be used for quick, local experiments but is not wired by default.
- **Frontend**: **React + TypeScript** using **MUI** component library for a modern UI.
- **Container orchestration**: `docker-compose` to run frontend, backend, and Postgres. Ollama is expected on the host and reached via `OLLAMA_HOST` (e.g. `http://host.docker.internal:11434` on Windows).

## High-Level Flow

1. The user submits one or more listing URLs and/or uploads images via the React frontend.
2. The FastAPI backend:
   - Fetches images from URLs or accepts uploaded files.
   - Uses OpenCV to detect card-shaped regions and produce normalized crops.
   - Sends each crop to Llama 3.2 Vision via Ollama for card name/expansion recognition.
   - Resolves these into canonical card records via Scryfall.
   - Aggregates pricing from Scryfall, Open TCG API, and Cardmarket.
3. The backend persists card and price data in Postgres and returns a report to the frontend.
4. The frontend displays per-card results, source images, and pricing/trend information, and can offer a downloadable report.

In addition, you can:

- Paste **card names directly** (comma- or newline-separated) to get a report without running the vision pipeline.
- **Download CSV** reports for any analysis.

## Frontend Features

- **Tabs**:
  - **Listing URLs**: paste one or more marketplace URLs (one per line).
  - **Upload Images**: select one or more local images containing cards.
  - **Card Names**: paste one or more card names, separated by commas or newlines.
- **Results view**:
  - Shows each detected/resolved card with name, set/collector number, and per-source prices.
  - Provides a **Download CSV** button for the current analysis.

## Running the Stack

From the project root:

```bash
export OLLAMA_HOST="http://host.docker.internal:11434"  # Windows host Ollama
docker-compose up --build
```

If the Postgres container fails on first run (e.g. after upgrading to Postgres 18), remove the old data volume and start fresh: `docker compose down -v && docker-compose up --build`.

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`

**Hot reload** is enabled: the backend app directory and frontend source are mounted into the containers, so code changes are picked up automatically (uvicorn `--reload` for the API, Vite HMR for the frontend).

## Testing

- **Backend tests** (pytest, via Docker/venv with pytest installed):

  ```bash
  cd backend
  python -m pytest
  ```

- **Frontend tests** (Vitest + React Testing Library):

  ```bash
  cd frontend
  npm test
  ```

## Configuring optional price sources (project-spec §4)

Beyond Scryfall (always used), you can enable extra price sources via environment variables.

### Open TCG API (tcgtracking.com)

- **No API key.** Free, daily updates, no sign-up required.
- Optional env (for the backend):
  - `OPEN_TCG_API_ENABLED=1` — turn on Open TCG pricing (when implemented). Omit or set to `0` to disable.
  - `OPEN_TCG_API_BASE_URL` — optional override; default is `https://tcgtracking.com/tcgapi/v1`.

### Cardmarket.com API

- Set `CARDMARKET_API_ENABLED=1` to enable (when the client is implemented). Default is `0`.
- **Requires a Cardmarket developer app** (OAuth 1.0). Create an app in your Cardmarket account; access may require professional seller approval.
- Set these in `.env` or in the `backend` service in `docker-compose` (do **not** commit real values to git):
  - `CARDMARKET_APP_TOKEN` — App token from your Cardmarket app.
  - `CARDMARKET_APP_SECRET` — App secret.
  - `CARDMARKET_ACCESS_TOKEN` — Access token (from OAuth flow / dedicated app).
  - `CARDMARKET_ACCESS_TOKEN_SECRET` — Access token secret.
- If `CARDMARKET_API_ENABLED` is not set to `1` or any of the credentials are missing, the backend will skip Cardmarket.

### Summary

| Source      | Config needed                          | Required |
|------------|-----------------------------------------|----------|
| Scryfall   | None                                    | Yes      |
| Open TCG   | `OPEN_TCG_API_ENABLED=1` (optional URL) | No       |
| Cardmarket | `CARDMARKET_API_ENABLED=1` + four OAuth env vars | No       |

## Development Notes

- Make sure **Docker Desktop** (or equivalent) is running on Windows 11.
- Ensure **Ollama** is installed and the Llama 3.2 Vision model is pulled on the host. Set `OLLAMA_HOST` for the backend to reach it from inside Docker (e.g. via `host.docker.internal`).
- Copy `.env.dist` to `.env` and set any required values: `cp .env.dist .env`. Do **not** commit `.env` to git; use it for secrets (Cardmarket tokens, etc.) and local overrides.

