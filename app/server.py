"""Servidor local de AppMusica.  Uso: .venv/bin/python -m app.server"""
from __future__ import annotations

import json
import math
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import pipeline

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "data" / "jobs"
STATIC_DIR = Path(__file__).resolve().parent / "static"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".aiff", ".aif", ".mp4", ".webm"}
MODES = {"teclado", "separar", "piano", "instrumento"}
JOB_ID = re.compile(r"^[a-f0-9]{12}$")
MAX_TIME = 24 * 3600  # tope razonable para los tiempos de recorte (s)

app = FastAPI(title="AppMusica")
work_queue: queue.Queue[str] = queue.Queue()
state_lock = threading.Lock()


@app.middleware("http")
async def revalidate_static(request, call_next):
    response = await call_next(request)
    if not request.url.path.startswith(("/api/", "/files/")):
        # La interfaz se revalida en cada carga: tras actualizar la app basta con recargar.
        response.headers["Cache-Control"] = "no-cache"
    return response


def job_file(job_id: str) -> Path:
    return JOBS_DIR / job_id / "job.json"


def read_job(job_id: str) -> dict:
    if not JOB_ID.match(job_id) or not job_file(job_id).exists():
        raise HTTPException(404, "Trabajo no encontrado")
    return json.loads(job_file(job_id).read_text())


def patch_job(job_id: str, **fields) -> None:
    with state_lock:
        data = json.loads(job_file(job_id).read_text())
        data.update(fields)
        pipeline.save_json(job_file(job_id), data)


def worker() -> None:
    while True:
        job_id = work_queue.get()
        job = read_job(job_id)
        job_dir = JOBS_DIR / job_id
        patch_job(job_id, status="procesando", step="Cargando modelos de IA…", progress=0.01)

        def update(step, progress):
            fields = {"progress": round(progress, 3)}
            if step:
                fields["step"] = step
            patch_job(job_id, **fields)

        try:
            started = time.time()
            result = pipeline.run_job(
                job_dir, job_dir / job["input"], job["mode"], job["piano_hq"], update,
                clip=job.get("clip"))
            patch_job(job_id, status="listo", result=result,
                      elapsed=round(time.time() - started, 1))
        except Exception as exc:  # noqa: BLE001 — se muestra al usuario
            traceback.print_exc()
            patch_job(job_id, status="error", step=f"Error: {exc}")
        finally:
            work_queue.task_done()


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    mode: str = Form("separar"),
    piano_hq: bool = Form(True),
    start: float | None = Form(None),
    end: float | None = Form(None),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in AUDIO_EXT:
        raise HTTPException(400, f"Formato no soportado: {ext or 'desconocido'}")
    if mode not in MODES:
        raise HTTPException(400, "Modo inválido")

    clip = None
    if start is not None or end is not None:
        s = start or 0.0
        if not all(math.isfinite(t) and 0 <= t <= MAX_TIME for t in (s, end) if t is not None):
            raise HTTPException(400, "Tiempos de recorte inválidos")
        if end is not None and end - s < pipeline.MIN_CLIP:
            raise HTTPException(400, f"El recorte debe durar al menos {pipeline.MIN_CLIP:g} s")
        clip = {"start": round(s, 3), "end": None if end is None else round(end, 3)}

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    with open(job_dir / f"input{ext}", "wb") as out:
        shutil.copyfileobj(file.file, out)

    pipeline.save_json(job_file(job_id), {
        "id": job_id, "name": Path(file.filename).stem, "input": f"input{ext}",
        "mode": mode, "piano_hq": piano_hq, "clip": clip, "created": time.time(),
        "status": "en cola", "step": "En cola…", "progress": 0,
    })
    work_queue.put(job_id)
    return {"id": job_id}


@app.get("/api/jobs")
def list_jobs():
    jobs = []
    for f in JOBS_DIR.glob("*/job.json"):
        try:
            j = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        jobs.append({k: j.get(k) for k in ("id", "name", "status", "created", "mode", "clip")})
    return sorted(jobs, key=lambda j: j["created"] or 0, reverse=True)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return read_job(job_id)


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job = read_job(job_id)
    if job["status"] in ("en cola", "procesando"):
        raise HTTPException(409, "El trabajo todavía se está procesando")
    shutil.rmtree(JOBS_DIR / job_id)
    return {"ok": True}


@app.get("/api/info")
def info():
    return {"piano_hq": pipeline.bytedance_available(), "device": pipeline._device()}


@app.get("/files/{job_id}/{name}")
def get_file(job_id: str, name: str):
    read_job(job_id)
    path = (JOBS_DIR / job_id / name).resolve()
    if path.parent != (JOBS_DIR / job_id).resolve() or not path.is_file():
        raise HTTPException(404, "Archivo no encontrado")
    return FileResponse(path)


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def resume_pending() -> None:
    """Re-encola trabajos que quedaron a medias si el servidor se cerró."""
    pending = []
    for f in JOBS_DIR.glob("*/job.json"):
        j = json.loads(f.read_text())
        if j.get("status") in ("en cola", "procesando"):
            pending.append(j)
    for j in sorted(pending, key=lambda j: j["created"]):
        patch_job(j["id"], status="en cola", step="En cola…", progress=0)
        work_queue.put(j["id"])


if __name__ == "__main__":
    resume_pending()
    threading.Thread(target=worker, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
