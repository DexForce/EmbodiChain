# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Same-origin HTTP boundary for the local image-frame annotation application."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator, Callable
import json
from pathlib import Path
import secrets
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ._cfg import AnnotationCfg
from ._models import ConflictError
from ._storage import AnnotationStore
from ._frames import SupersededError, FrameWorker
from .frames import FrameSource

__all__ = ["create_app"]


def create_app(
    source_factory: Callable[[], FrameSource],
    annotations: Path,
    cfg: AnnotationCfg | None = None,
) -> FastAPI:
    """Create one local app with a lifespan-owned frame source and annotation file.

    Args:
        source_factory: Create a fresh frame source on its owning worker thread.
        annotations: Annotation JSON destination.
        cfg: Optional tool resource configuration.

    Returns:
        An ASGI app ready for a single Uvicorn process.
    """
    cfg = cfg or AnnotationCfg()
    cfg.validate()
    store = AnnotationStore(annotations, cfg.document_bytes)
    worker = FrameWorker(source_factory, cfg, store.initialize)
    token = secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal worker, store
        store = AnnotationStore(annotations, cfg.document_bytes)
        worker = FrameWorker(source_factory, cfg, store.initialize)
        worker.start()
        try:
            yield
        finally:
            await run_in_threadpool(worker.close)

    app = FastAPI(
        title="Trajectory Annotator", lifespan=lifespan, docs_url=None, redoc_url=None
    )
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"]
    )

    def ready() -> None:
        status = worker.status()
        if status["phase"] != "ready":
            raise HTTPException(503, status["error"] or "帧源仍在初始化，或已经关闭。")

    def authorize(request: Request) -> None:
        if not secrets.compare_digest(
            request.headers.get("X-Annotation-Token", ""), token
        ):
            raise HTTPException(403, "请通过本工具页面提交修改。")

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return {**worker.status(), "session_token": token}

    @app.get("/api/project")
    async def project() -> dict[str, Any]:
        ready()
        snapshot = store.snapshot()
        return {
            **snapshot,
            "source_name": snapshot["document"]["source"]["name"],
            "annotation_path": str(store.path),
            "session_token": token,
            "autosave_ms": cfg.autosave_ms,
        }

    @app.get("/api/frames/{index}")
    async def frame(index: int, request_id: int = 0) -> Response:
        ready()
        try:
            preview = await asyncio.wrap_future(worker.request(index))
        except SupersededError as error:
            raise HTTPException(409, str(error)) from error
        except (ValueError, IndexError) as error:
            raise HTTPException(422, str(error)) from error
        except (RuntimeError, OSError) as error:
            raise HTTPException(500, str(error)) from error
        return Response(
            preview.jpeg,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store",
                "X-Frame-Index": str(preview.index),
                "X-Frame-PTS": str(preview.pts) if preview.pts is not None else "",
                "X-Frame-Time": str(preview.time_seconds),
                "X-Frame-Delay": str(preview.delay_seconds),
                "X-Request-ID": str(request_id),
            },
        )

    @app.put("/api/annotations")
    async def save(request: Request) -> dict[str, Any]:
        authorize(request)
        ready()
        if (
            request.headers.get("content-type", "").split(";", 1)[0]
            != "application/json"
        ):
            raise HTTPException(415, "请求必须使用 application/json。")
        content = bytearray()
        async for chunk in request.stream():
            if len(content) + len(chunk) > cfg.document_bytes:
                raise HTTPException(413, "标注内容超过大小上限。")
            content.extend(chunk)
        try:
            payload = json.loads(content)
            if not isinstance(payload, dict) or payload.keys() != {
                "segments",
                "expected_revision",
            }:
                raise ValueError("请求需要 segments 和 expected_revision 字段。")

            def persist() -> dict[str, Any]:
                if worker.reader is None:
                    raise RuntimeError("帧源尚未准备。")
                worker.reader.check_unchanged()
                return store.save(payload["segments"], payload["expected_revision"])

            return await run_in_threadpool(persist)
        except ConflictError as error:
            raise HTTPException(409, str(error)) from error
        except (ValueError, UnicodeError) as error:
            raise HTTPException(422, str(error)) from error
        except OSError as error:
            raise HTTPException(507, f"保存失败：{error}") from error

    @app.post("/api/cancel")
    async def cancel(request: Request) -> dict[str, Any]:
        authorize(request)
        if worker.status()["phase"] == "ready":
            raise HTTPException(409, "帧源已就绪，无需取消初始化。")
        await run_in_threadpool(worker.close)
        return worker.status()

    web = Path(__file__).with_name("web")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(web / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=web), name="static")
    return app
