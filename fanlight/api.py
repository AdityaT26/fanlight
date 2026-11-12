"""HTTP API (FastAPI) and the static web app."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .config import SIGNALS
from .engine import Engine, UnknownReader

log = logging.getLogger("fanlight")

STATIC = Path(__file__).parent / "web" / "static"
Action = Literal["view", "open", "like", "save", "hide"]


class ReaderIn(BaseModel):
    name: str = Field(default="Reader", max_length=60)
    languages: list[str] = Field(default_factory=lambda: ["english"], min_length=1, max_length=4)
    categories: list[str] = Field(default_factory=list, max_length=60)


class ReaderPatch(BaseModel):
    name: str | None = Field(default=None, max_length=60)
    languages: list[str] | None = Field(default=None, min_length=1, max_length=4)
    categories: list[str] | None = Field(default=None, max_length=60)


class EventIn(BaseModel):
    article_id: str = Field(max_length=64)
    action: Action


class EventsIn(BaseModel):
    events: list[EventIn] = Field(min_length=1, max_length=200)


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(
        title="Fanlight",
        version=__version__,
        description="Transparent, personalised news ranking.",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )
    app.state.engine = engine

    @app.exception_handler(UnknownReader)
    async def _unknown(_: Request, exc: UnknownReader) -> JSONResponse:
        return JSONResponse({"detail": f"No reader with id {exc.args[0]!r}"}, status_code=404)

    @app.exception_handler(ValueError)
    async def _invalid(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(PermissionError)
    async def _forbidden(_: Request, exc: PermissionError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=403)

    # ------------------------------------------------------------------ #

    @app.get("/healthz", include_in_schema=False)
    def health() -> dict[str, Any]:
        return {"ok": True, "articles": engine.index.size}

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        return engine.meta()

    @app.get("/api/readers")
    def readers(
        limit: Annotated[int, Query(ge=1, le=500)] = 60,
        ids: Annotated[str | None, Query(max_length=2000, description="Comma-separated reader ids")] = None,
    ) -> list[dict[str, Any]]:
        if ids is not None:
            return engine.list_readers(ids=[x for x in ids.split(",") if x][:20])
        if engine.settings.public:
            return []  # in public mode nobody can list other people's readers
        return engine.list_readers(limit)

    @app.post("/api/readers", status_code=201)
    def create_reader(body: ReaderIn) -> dict[str, Any]:
        profile = engine.create_reader(body.name, body.languages, body.categories)
        return engine.describe_reader(profile.id)

    @app.get("/api/readers/{reader_id}")
    def reader(reader_id: str) -> dict[str, Any]:
        return engine.describe_reader(reader_id)

    @app.patch("/api/readers/{reader_id}")
    def update_reader(reader_id: str, body: ReaderPatch) -> dict[str, Any]:
        engine.update_reader(reader_id, body.name, body.languages, body.categories)
        return engine.describe_reader(reader_id)

    @app.delete("/api/readers/{reader_id}", status_code=204)
    def delete_reader(reader_id: str) -> None:
        engine.delete_reader(reader_id)

    @app.get("/api/readers/{reader_id}/feed")
    def feed(
        reader_id: str,
        k: Annotated[int, Query(ge=1, le=100)] = 20,
        category: Annotated[list[str] | None, Query()] = None,
        diversity: Annotated[float | None, Query(ge=0, le=1)] = None,
        discovery: Annotated[float | None, Query(ge=0, le=0.5)] = None,
        half_life_hours: Annotated[float | None, Query(ge=1, le=2160)] = None,
        hide_read: bool | None = None,
        w_category: Annotated[float | None, Query(ge=0, le=10)] = None,
        w_tags: Annotated[float | None, Query(ge=0, le=10)] = None,
        w_content: Annotated[float | None, Query(ge=0, le=10)] = None,
        w_freshness: Annotated[float | None, Query(ge=0, le=10)] = None,
        w_source: Annotated[float | None, Query(ge=0, le=10)] = None,
        w_popularity: Annotated[float | None, Query(ge=0, le=10)] = None,
    ) -> dict[str, Any]:
        given = locals()
        weights = {s: given[f"w_{s}"] for s in SIGNALS if given[f"w_{s}"] is not None}
        overrides: dict[str, Any] = {
            "diversity": diversity,
            "discovery": discovery,
            "half_life_hours": half_life_hours,
            "hide_read": hide_read,
        }
        if weights:
            overrides["weights"] = weights
        return engine.feed(reader_id, k=k, overrides=overrides, categories=category)

    @app.post("/api/readers/{reader_id}/events")
    def record(reader_id: str, body: EventsIn) -> dict[str, int]:
        stored = engine.record(reader_id, [(e.article_id, e.action) for e in body.events])
        return {"stored": stored}

    @app.delete("/api/readers/{reader_id}/events")
    def undo(reader_id: str, body: Annotated[EventIn, Body()]) -> dict[str, int]:
        return {"removed": engine.undo(reader_id, body.article_id, body.action)}

    @app.delete("/api/readers/{reader_id}/history")
    def reset(reader_id: str) -> dict[str, int]:
        return {"removed": engine.reset_history(reader_id)}

    @app.get("/api/feeds")
    def feeds() -> dict[str, Any]:
        if engine.live is None:
            raise HTTPException(404, "Live feeds are off. Start the server with --source live.")
        return engine.live.status()

    @app.get("/api/articles/{article_id}")
    def article(article_id: str) -> dict[str, Any]:
        i = engine.index.position.get(article_id)
        if i is None:
            raise HTTPException(404, f"No article with id {article_id!r}")
        return engine.article_json(i)

    # ------------------------------------------------------------------ #
    # Web app
    # ------------------------------------------------------------------ #

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/manifest.webmanifest", include_in_schema=False)
    def manifest() -> FileResponse:
        return FileResponse(STATIC / "manifest.webmanifest", media_type="application/manifest+json")

    return app
