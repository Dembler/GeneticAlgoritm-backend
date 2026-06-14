from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.di import get_settings

settings = get_settings()
base_dir = Path(__file__).resolve().parent
frontend_dist_dir = base_dir / "frontend_dist"
frontend_assets_dir = frontend_dist_dir / "assets"

app = FastAPI(title=settings.app_name)

cors_allowed_origins = settings.parsed_cors_allowed_origins
if cors_allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_allowed_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(router)

if frontend_assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_assets_dir)), name="assets")


def _serve_frontend_file(path: str | None = None):
    if not frontend_dist_dir.exists():
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Frontend build not found. Run `npm install` and `npm run build` in the `frontend` directory.",
            },
        )

    if path:
        candidate = frontend_dist_dir / path
        if candidate.is_file():
            return FileResponse(candidate)

    index_file = frontend_dist_dir / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)

    return JSONResponse(
        status_code=503,
        content={
            "detail": "Frontend index.html not found. Build the Vue app before opening the UI.",
        },
    )


@app.get("/", include_in_schema=False)
async def serve_frontend_root():
    return _serve_frontend_file()


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_frontend(full_path: str):
    if full_path.startswith("api"):
        raise HTTPException(status_code=404, detail="Not found.")
    return _serve_frontend_file(full_path)
