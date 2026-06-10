import os
import uuid
import shutil
import json
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.database import engine, Base
from backend.pipeline.interfaces import BatchJob, ComponentMetadata
from backend.pipeline.queue_manager import QueueManager, STORAGE_DIR

app = FastAPI(
    title="Educational Diagram Intelligence Platform (V2) API",
    description="Backend API for zero-shot diagram decomposition and PPT reconstruction",
    version="2.0.0"
)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

queue_manager = QueueManager()

# Startup and Shutdown events
@app.on_event("startup")
def startup_event():
    # Initialize SQL database tables
    Base.metadata.create_all(bind=engine)
    queue_manager.start()
    print("Database tables initialized. V2 main server active.")

@app.on_event("shutdown")
async def shutdown_event():
    await queue_manager.stop()

# Helper to save uploaded files
def save_upload(upload_file: UploadFile, temp_dir: str) -> str:
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, upload_file.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return temp_path

# Endpoints

@app.post("/api/upload", response_model=BatchJob)
async def upload_diagrams(
    background_tasks: BackgroundTasks, 
    files: List[UploadFile] = File(...)
):
    """
    Accepts a batch upload of one or more diagrams. Queues a background job.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
        
    batch_id = uuid.uuid4().hex
    temp_dir = os.path.join(STORAGE_DIR, "temp", batch_id)
    
    saved_paths = []
    try:
        for f in files:
            ext = os.path.splitext(f.filename)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png", ".webp", ".svg", ".pdf"]:
                raise HTTPException(status_code=400, detail=f"Unsupported format: {f.filename}")
            
            temp_path = save_upload(f, temp_dir)
            saved_paths.append(temp_path)
            
        # Create and queue V2 batch job
        job = queue_manager.create_batch_job(batch_id, saved_paths)
        return job
        
    except HTTPException as he:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise he
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Internal Upload Error: {str(e)}")
    finally:
        # Clean temp directory asynchronously
        background_tasks.add_task(shutil.rmtree, temp_dir, ignore_errors=True)

@app.get("/api/status/{batch_id}")
async def get_batch_status(batch_id: str):
    """
    Returns slide metadata, quality routing statuses, and performance latency stats.
    """
    job = queue_manager.get_job(batch_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
        
    # Enrich job dictionary with performance metrics from metadata.json
    job_dict = job.dict()
    for idx, slide in enumerate(job.slides):
        metadata_file = os.path.join(STORAGE_DIR, "batches", batch_id, "slides", f"slide_{idx}", "metadata", "metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                job_dict["slides"][idx]["performance_metrics"] = meta_data.get("performance_metrics", {})
                job_dict["slides"][idx]["relationships"] = meta_data.get("relationships", [])
            except Exception:
                pass
    return job_dict

class UpdateComponentsRequest(BaseModel):
    components: List[ComponentMetadata]

@app.post("/api/batches/{batch_id}/slides/{slide_idx}/components")
async def update_slide_components(batch_id: str, slide_idx: int, req: UpdateComponentsRequest):
    """
    Updates manual bounding boxes, text edits, and visibility. Triggers PPTX rebuild.
    """
    success = queue_manager.update_slide_components(batch_id, slide_idx, req.components)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to update slide components.")
    return {"status": "success", "message": "Components saved and PPT rebuilt successfully."}

@app.get("/api/batches/{batch_id}/slides/{slide_idx}/logs", response_class=PlainTextResponse)
async def get_slide_pipeline_logs(batch_id: str, slide_idx: int):
    """
    Fetches the running pipeline execution log file contents.
    """
    log_path = os.path.join(STORAGE_DIR, "batches", batch_id, "slides", f"slide_{slide_idx}", "logs", "pipeline.log")
    if not os.path.exists(log_path):
        return "Log file not found or pipeline hasn't started yet."
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading logs: {e}"

@app.get("/api/download/{batch_id}")
async def download_pptx(batch_id: str):
    """
    Downloads the compiled PowerPoint file.
    """
    job = queue_manager.get_job(batch_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
        
    if job.status != "completed" or not job.pptx_path:
        raise HTTPException(status_code=400, detail="Presentation is not compiled yet.")
        
    pptx_abs_path = os.path.abspath(os.path.join(STORAGE_DIR, "..", job.pptx_path))
    if not os.path.exists(pptx_abs_path):
        raise HTTPException(status_code=404, detail="PPTX file not found on disk")
        
    return FileResponse(
        pptx_abs_path, 
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=f"diagram_decomposition_{batch_id[:6]}.pptx"
    )

# Static Storage Mount (for transparent PNG masks & background crops)
app.mount("/storage", StaticFiles(directory=STORAGE_DIR), name="storage")

# Serve Frontend dashboard
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    print(f"Warning: frontend directory not found at {frontend_dir}")
