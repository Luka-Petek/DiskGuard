# DiskGuard Frontend

React 19 + Vite dashboard for the DiskGuard hard-drive current-state assessment pipeline.

## Current behaviour

- Uploads a `smartctl -j` JSON file to `POST /api/predict/combined`.
- Displays the combined AHI score, Healthy/Warning/Critical verdict, component model outputs, drive information and analysis history.
- Parses basic drive information and SMART attributes in the browser for display.
- Bundles sample JSON files from the repository `DiskJson/` directory at build time. Files containing `_results` or `sweep` in their names are excluded from the sample list.

The frontend does not run the models itself. The FastAPI backend must be available for analysis.

## Local development

```bash
npm install
npm run dev
```

The API base URL defaults to `http://localhost:8000`. Override it when needed:

```bash
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

On Windows PowerShell:

```powershell
$env:VITE_API_BASE_URL="http://localhost:8000"
npm run dev
```

## Checks

```bash
npm run lint
npm run build
```

## Docker

From the repository root:

```bash
docker compose up -d --build
```

The composed services are exposed at:

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`
- TensorBoard: `http://localhost:6006`

The production frontend image is built with Node and served by Nginx.
