# Production

## How to run backend

From the project root:

```bash
uvicorn production.backend.api.main:app --reload --port 8000
```

From inside `production/backend/`:

```bash
uvicorn api.main:app --reload --port 8000
```

## How to run frontend

From `production/frontend/`:

```bash
npm install   # first time only
npm run dev
```

Then open `http://localhost:5173`.

> The Vite dev server proxies `/api` requests to `http://localhost:8000`, so the backend must also be running.
