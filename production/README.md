# Production

## How to run backend

From the project root:

```bash
uvicorn production.backend.api.main:app --reload --port 8000
```

## How to run frontend

The frontend is a static site. Open `production/frontend/index.html` directly in a browser, or serve it with:

```bash
cd production/frontend
python -m http.server 3000
```

Then open `http://localhost:3000`.

> The frontend expects the backend running at `http://localhost:8000`.
