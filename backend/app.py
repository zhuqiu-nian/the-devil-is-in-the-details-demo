from __future__ import annotations

from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .codec import KODAK_DIR, OUTPUTS_DIR, available_models, batch_eval, get_cached_result, list_kodak_images, run_codec

app = FastAPI(
    title="Window-Based Attention Image Compression Demo",
    description="Local reproduction dashboard for STF/CNN+WAM image-compression comparisons.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/data", StaticFiles(directory=str(KODAK_DIR)), name="data")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


class RunRequest(BaseModel):
    image_id: str = Field(..., examples=["kodim07"])
    model_id: str = Field(..., examples=["stf"])
    quality: str = Field(..., examples=["0.0035"])
    force: bool = False


class BatchSelection(BaseModel):
    model_id: str
    quality: str


class BatchEvalRequest(BaseModel):
    selections: List[BatchSelection]
    image_ids: Optional[List[str]] = None
    max_images: Optional[int] = Field(default=None, ge=1, le=24)
    force: bool = False


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return """
    <!doctype html>
    <html lang="zh-CN">
      <head>
        <meta charset="utf-8" />
        <title>STF Compression Backend</title>
        <style>
          body { font-family: system-ui, sans-serif; margin: 40px; line-height: 1.6; color: #172033; }
          code { background: #eef1f3; padding: 2px 6px; border-radius: 4px; }
        </style>
      </head>
      <body>
        <h1>STF Compression Backend is running</h1>
        <p>This FastAPI service listens on <code>8000</code> and only serves API/static assets.</p>
        <p>Open the React web demo on frontend port <code>5173</code>.</p>
      </body>
    </html>
    """


@app.get("/api/images")
def images() -> list[dict[str, Any]]:
    return list_kodak_images()


@app.get("/api/models")
def models() -> list[dict[str, Any]]:
    return available_models()


@app.post("/api/run")
def run(request: RunRequest) -> dict[str, Any]:
    try:
        return run_codec(request.image_id, request.model_id, request.quality, force=request.force)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/batch_eval")
def run_batch(request: BatchEvalRequest) -> dict[str, Any]:
    try:
        selections = [
            selection.model_dump() if hasattr(selection, "model_dump") else selection.dict()
            for selection in request.selections
        ]
        return batch_eval(
            selections=selections,
            image_ids=request.image_ids,
            max_images=request.max_images,
            force=request.force,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/results/{result_id}")
def result(result_id: str) -> dict[str, Any]:
    try:
        return get_cached_result(result_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Result not found: {result_id}") from exc
