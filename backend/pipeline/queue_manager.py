import os
import time
import cv2
import numpy as np
import json
import asyncio
import shutil
import traceback
from typing import Dict, List, Optional

# Core datatypes
from backend.pipeline.interfaces import BatchJob, SlideMetadata, ComponentMetadata, RelationshipMetadata
from backend.pipeline.classifier import ComponentClassifier
from backend.pipeline.detectors.detector_provider import DetectorProvider
from backend.pipeline.segmenters.segmenter_provider import SegmenterProvider
from backend.pipeline.ocr.ocr_provider import OCRProvider
from backend.pipeline.understanding_engine import DiagramUnderstandingEngine
from backend.pipeline.ppt_generator import PPTGenerator

# Database layer
from backend.database import SessionLocal
from backend.models import DbBatchJob, DbSlideMetadata, DbComponentMetadata, DbRelationship

STORAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "storage"))

class QueueManager:
    def __init__(self):
        self.jobs: Dict[str, BatchJob] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self.worker_task: Optional[asyncio.Task] = None
        
        # Instantiate pipeline providers
        self.classifier = ComponentClassifier()
        self.detector = DetectorProvider.get_detector("GroundingDINO")
        self.segmenter = SegmenterProvider.get_segmenter("SAM2")
        self.ocr_engine = OCRProvider.get_ocr("PaddleOCR")
        self.understanding_engine = DiagramUnderstandingEngine()
        self.ppt_generator = PPTGenerator()
        
        # Ensure base storage exists
        os.makedirs(STORAGE_DIR, exist_ok=True)
        os.makedirs(os.path.join(STORAGE_DIR, "batches"), exist_ok=True)

    def start(self):
        """Starts the background processing worker."""
        if self.worker_task is None:
            self.worker_task = asyncio.create_task(self._worker_loop())
            print("V2 Queue worker started.")

    async def stop(self):
        """Stops the background worker."""
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
            self.worker_task = None
            print("V2 Queue worker stopped.")

    def create_batch_job(self, batch_id: str, files: List[str]) -> BatchJob:
        """
        Registers a new batch job.
        Files are absolute paths to uploaded files in storage.
        """
        job = BatchJob(
            batch_id=batch_id,
            status="queued",
            slides=[],
            created_at=time.time()
        )
        self.jobs[batch_id] = job
        
        # Save batch directories
        batch_dir = os.path.join(STORAGE_DIR, "batches", batch_id)
        os.makedirs(batch_dir, exist_ok=True)
        
        # Create temp original uploads
        uploads_dir = os.path.join(batch_dir, "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        
        saved_paths = []
        for idx, file_path in enumerate(files):
            filename = os.path.basename(file_path)
            dest = os.path.join(uploads_dir, f"slide_{idx}{os.path.splitext(filename)[1]}")
            shutil.copy(file_path, dest)
            saved_paths.append((idx, dest, filename))
            
        # Write initial metadata file
        self._save_job_meta(job)
        
        # Write initial DB Batch Job Row
        db = SessionLocal()
        db_job = DbBatchJob(
            batch_id=batch_id,
            status="queued",
            created_at=time.time()
        )
        db.merge(db_job)
        db.commit()
        db.close()
        
        # Enqueue job for background processing
        self.queue.put_nowait((batch_id, saved_paths))
        return job

    def get_job(self, batch_id: str) -> Optional[BatchJob]:
        """Gets batch job details from memory, DB, or storage disk."""
        if batch_id in self.jobs:
            return self.jobs[batch_id]
            
        # Try loading from database
        db = SessionLocal()
        db_job = db.query(DbBatchJob).filter(DbBatchJob.batch_id == batch_id).first()
        if db_job:
            # Reconstruct BatchJob model
            slides = []
            for db_slide in db_job.slides:
                # Load components
                components = []
                for db_comp in db_slide.components:
                    components.append(ComponentMetadata(
                        id=db_comp.component_id,
                        type=db_comp.type,
                        semantic_name=db_comp.semantic_name,
                        box=[db_comp.box_x, db_comp.box_y, db_comp.box_w, db_comp.box_h],
                        mask_path=db_comp.mask_path,
                        crop_path=db_comp.crop_path,
                        text=db_comp.text,
                        confidence=db_comp.confidence,
                        visible=db_comp.visible,
                        z_index=db_comp.z_index,
                        associated_label_id=db_comp.associated_label_id,
                        associated_object_id=db_comp.associated_object_id
                    ))
                
                # Load relationships
                rels = []
                for db_rel in db_slide.relationships:
                    rels.append(RelationshipMetadata(
                        label_id=db_rel.label_id,
                        label_text=db_rel.label_text,
                        arrow_id=db_rel.arrow_id,
                        target_id=db_rel.target_id
                    ))
                
                # Combine slide meta
                slide_meta = SlideMetadata(
                    slide_index=db_slide.slide_index,
                    original_filename=db_slide.original_filename,
                    width=db_slide.width,
                    height=db_slide.height,
                    components=components,
                    routing_status=db_slide.routing_status,
                    average_confidence=db_slide.average_confidence,
                    file_type=db_slide.file_type,
                    content_type=db_slide.content_type
                )
                slides.append(slide_meta)
                
            job = BatchJob(
                batch_id=batch_id,
                status=db_job.status,
                slides=slides,
                created_at=db_job.created_at,
                completed_at=db_job.completed_at,
                error_message=db_job.error_message,
                pptx_path=db_job.pptx_path
            )
            self.jobs[batch_id] = job
            db.close()
            return job
            
        db.close()
        return None

    def update_slide_components(self, batch_id: str, slide_idx: int, components: List[ComponentMetadata]) -> bool:
        """Updates slide metadata with modified components from the UI editor."""
        job = self.get_job(batch_id)
        if not job or slide_idx >= len(job.slides):
            return False
            
        slide = job.slides[slide_idx]
        slide.components = components
        
        # Review status changes to approved
        slide.routing_status = "Approved"
        
        # Save V2 workspace metadata.json
        slide_dir = os.path.join(STORAGE_DIR, "batches", batch_id, "slides", f"slide_{slide_idx}")
        self._generate_erased_background(slide_dir, slide.components)
        
        metadata_dir = os.path.join(slide_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        
        slide_meta_dict = slide.dict()
        # Retrieve existing metadata to preserve relationships and metrics
        existing_metrics = {}
        relationships_list = []
        try:
            with open(os.path.join(metadata_dir, "metadata.json"), "r", encoding="utf-8") as f:
                old_data = json.load(f)
                existing_metrics = old_data.get("performance_metrics", {})
                relationships_list = old_data.get("relationships", [])
        except Exception:
            pass
            
        slide_meta_dict["performance_metrics"] = existing_metrics
        slide_meta_dict["relationships"] = relationships_list
        
        with open(os.path.join(metadata_dir, "metadata.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(slide_meta_dict, indent=2))
            
        self._save_job_meta(job)
        
        # Update database entries
        db = SessionLocal()
        try:
            db_slide = db.query(DbSlideMetadata).filter(
                DbSlideMetadata.batch_id == batch_id, 
                DbSlideMetadata.slide_index == slide_idx
            ).first()
            
            if db_slide:
                db_slide.routing_status = "Approved"
                # Remove existing components in DB for this slide
                db.query(DbComponentMetadata).filter(DbComponentMetadata.slide_id == db_slide.id).delete()
                # Insert updated components
                for comp in components:
                    db_comp = DbComponentMetadata(
                        slide_id=db_slide.id,
                        component_id=comp.id,
                        type=comp.type,
                        semantic_name=comp.semantic_name,
                        box_x=comp.box[0],
                        box_y=comp.box[1],
                        box_w=comp.box[2],
                        box_h=comp.box[3],
                        mask_path=comp.mask_path,
                        crop_path=comp.crop_path,
                        text=comp.text,
                        confidence=comp.confidence,
                        visible=comp.visible,
                        z_index=comp.z_index,
                        associated_label_id=comp.associated_label_id,
                        associated_object_id=comp.associated_object_id
                    )
                    db.add(db_comp)
                db.commit()
        except Exception as dbe:
            print(f"Database update failed: {dbe}")
            db.rollback()
        finally:
            db.close()
            
        # Rebuild PPTX to incorporate edits
        try:
            self._compile_pptx(job)
            return True
        except Exception as e:
            print(f"Failed to rebuild PPTX after update: {e}")
            return False

    def _save_job_meta(self, job: BatchJob):
        """Saves overall batch state to disk."""
        batch_dir = os.path.join(STORAGE_DIR, "batches", job.batch_id)
        with open(os.path.join(batch_dir, "batch.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(job.dict(), indent=2))

    def _generate_erased_background(self, slide_dir: str, components: List[ComponentMetadata]):
        """
        Creates a 'background.png' template inside masks/ where the bounding boxes
        of all currently visible components are filled with the sampled background color.
        """
        original_file = None
        dir_original = os.path.join(slide_dir, "original")
        if os.path.exists(dir_original):
            files = os.listdir(dir_original)
            if files:
                original_file = files[0]
                
        if not original_file:
            return
            
        original_path = os.path.join(dir_original, original_file)
        img = cv2.imread(original_path)
        if img is None:
            return
            
        # Sample background color from corners
        corners = [img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]]
        bg_color = np.mean(corners, axis=0).astype(int).tolist()
        bg_color_tuple = (bg_color[0], bg_color[1], bg_color[2])
        
        # Erase bounding boxes
        for comp in components:
            if not comp.visible:
                continue
            x, y, w, h = comp.box
            x_start = max(0, x - 2)
            y_start = max(0, y - 2)
            x_end = min(img.shape[1], x + w + 2)
            y_end = min(img.shape[0], y + h + 2)
            cv2.rectangle(img, (x_start, y_start), (x_end, y_end), bg_color_tuple, -1)
            
        background_path = os.path.join(slide_dir, "masks", "background.png")
        os.makedirs(os.path.dirname(background_path), exist_ok=True)
        cv2.imwrite(background_path, img)

    async def _worker_loop(self):
        """Core background loop processing jobs one by one."""
        while True:
            try:
                batch_id, slide_files = await self.queue.get()
                job = self.jobs.get(batch_id)
                if not job:
                    self.queue.task_done()
                    continue
                    
                # Update DB state
                db = SessionLocal()
                db_job = db.query(DbBatchJob).filter(DbBatchJob.batch_id == batch_id).first()
                if db_job:
                    db_job.status = "processing"
                    db.commit()
                db.close()
                
                job.status = "processing"
                self._save_job_meta(job)
                
                try:
                    slides_meta = []
                    
                    for idx, dest_path, original_filename in slide_files:
                        print(f"Processing Batch {batch_id} - Slide {idx}...")
                        slide_meta = self._process_single_slide(batch_id, idx, dest_path, original_filename)
                        slides_meta.append(slide_meta)
                        
                    job.slides = slides_meta
                    job.status = "completed"
                    job.completed_at = time.time()
                    
                    # Generate PowerPoint Presentation
                    self._compile_pptx(job)
                    
                    # Update DB completed state
                    db = SessionLocal()
                    db_job = db.query(DbBatchJob).filter(DbBatchJob.batch_id == batch_id).first()
                    if db_job:
                        db_job.status = "completed"
                        db_job.completed_at = job.completed_at
                        db_job.pptx_path = job.pptx_path
                        db.commit()
                    db.close()
                    
                except Exception as e:
                    traceback.print_exc()
                    job.status = "error"
                    job.error_message = f"{str(e)}\n{traceback.format_exc()}"
                    job.completed_at = time.time()
                    
                    # Update DB error state
                    db = SessionLocal()
                    db_job = db.query(DbBatchJob).filter(DbBatchJob.batch_id == batch_id).first()
                    if db_job:
                        db_job.status = "error"
                        db_job.error_message = job.error_message
                        db_job.completed_at = job.completed_at
                        db.commit()
                    db.close()
                    
                self._save_job_meta(job)
                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Queue worker encountered error: {e}")
                await asyncio.sleep(2)
 
    def _process_single_slide(self, batch_id: str, slide_idx: int, file_path: str, original_filename: str) -> SlideMetadata:
        """Runs the pipeline stages on a single slide and saves files to V2 workspace."""
        slide_dir = os.path.join(STORAGE_DIR, "batches", batch_id, "slides", f"slide_{slide_idx}")
        
        # Setup V2 Workspace Directories
        dir_original = os.path.join(slide_dir, "original")
        dir_detections = os.path.join(slide_dir, "detections")
        dir_masks = os.path.join(slide_dir, "masks")
        dir_ocr = os.path.join(slide_dir, "ocr")
        dir_metadata = os.path.join(slide_dir, "metadata")
        dir_ppt = os.path.join(slide_dir, "ppt")
        dir_logs = os.path.join(slide_dir, "logs")
        
        for d in [dir_original, dir_detections, dir_masks, dir_ocr, dir_metadata, dir_ppt, dir_logs]:
            os.makedirs(d, exist_ok=True)
            
        # Save original filename in a text file for mock resolvers
        with open(os.path.join(slide_dir, "original_filename.txt"), "w", encoding="utf-8") as f:
            f.write(original_filename)
            
        log_file = open(os.path.join(dir_logs, "pipeline.log"), "w", encoding="utf-8")
        
        def log_message(msg):
            # Print with backslashreplace for safe terminal printing
            print(f"    {msg}".encode('ascii', errors='backslashreplace').decode('ascii'))
            log_file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
            
        # Copy original slide into its specific slide output directory
        slide_ext = os.path.splitext(file_path)[1]
        slide_original = os.path.join(dir_original, f"original{slide_ext}")
        shutil.copy(file_path, slide_original)
        
        # 1. Classification
        t_start = time.time()
        class_res = self.classifier.classify(slide_original)
        t_classification = int((time.time() - t_start) * 1000)
        log_message(f"Stage 1: Classification -> Type: {class_res['contentType']}, File: {class_res['fileType']} ({t_classification}ms)")
        
        # Get dimensions
        img = cv2.imread(slide_original)
        if img is not None:
            height, width = img.shape[:2]
        else:
            width, height = 1280, 720
            
        # 2. Bounding Box Detection
        t_start = time.time()
        detections = self.detector.detect(slide_original)
        t_detection = int((time.time() - t_start) * 1000)
        log_message(f"Stage 2: Detection -> Found {len(detections)} boxes ({t_detection}ms)")
        
        # Save raw detections.json
        with open(os.path.join(dir_detections, "detections.json"), "w", encoding="utf-8") as f:
            json.dump([d.dict() for d in detections], f, indent=2)
            
        # 3. Shape Segmentation
        t_start = time.time()
        segmentations = self.segmenter.segment(slide_original, detections, dir_masks)
        t_segmentation = int((time.time() - t_start) * 1000)
        log_message(f"Stage 3: Segmentation -> Segmented {len(segmentations)} crops/masks ({t_segmentation}ms)")
        
        with open(os.path.join(dir_masks, "segmentation.json"), "w", encoding="utf-8") as f:
            json.dump([s.dict() for s in segmentations], f, indent=2)
            
        # 4. OCR text extraction
        t_start = time.time()
        ocr_results = self.ocr_engine.extract_text(slide_original, detections)
        t_ocr = int((time.time() - t_start) * 1000)
        log_message(f"Stage 4: OCR -> Extracted {len(ocr_results)} text boxes ({t_ocr}ms)")
        
        with open(os.path.join(dir_ocr, "ocr.json"), "w", encoding="utf-8") as f:
            json.dump([o.dict() for o in ocr_results], f, indent=2)
            
        # 5. Diagram Understanding Engine
        t_start = time.time()
        components, relationships = self.understanding_engine.process_diagram(
            width, height, detections, segmentations, ocr_results
        )
        t_understanding = int((time.time() - t_start) * 1000)
        log_message(f"Stage 5: Understanding & Reconstruction -> Compiled {len(components)} components with {len(relationships)} relationships ({t_understanding}ms)")
        
        # Generate background.png
        self._generate_erased_background(slide_dir, components)
        
        # Quality routing score
        conf_scores = [c.confidence for c in components]
        avg_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0.85
        if avg_conf > 0.90:
            routing_status = "Auto Export"
        elif avg_conf >= 0.70:
            routing_status = "Warning"
        else:
            routing_status = "Manual Review"
            
        metrics = {
            "classification_time_ms": t_classification,
            "detection_time_ms": t_detection,
            "segmentation_time_ms": t_segmentation,
            "ocr_time_ms": t_ocr,
            "understanding_time_ms": t_understanding,
            "ppt_compile_time_ms": 0
        }
        
        slide_meta = SlideMetadata(
            slide_index=slide_idx,
            original_filename=original_filename,
            width=width,
            height=height,
            components=components,
            routing_status=routing_status,
            average_confidence=float(round(avg_conf, 2)),
            file_type=class_res["fileType"],
            content_type=class_res["contentType"]
        )
        
        # Save metadata.json
        slide_meta_dict = slide_meta.dict()
        slide_meta_dict["performance_metrics"] = metrics
        slide_meta_dict["relationships"] = [r.dict() for r in relationships]
        with open(os.path.join(dir_metadata, "metadata.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(slide_meta_dict, indent=2))
            
        # Log to Database
        db = SessionLocal()
        try:
            # Remove any existing slide record for this batch and index to prevent duplication
            existing_slides = db.query(DbSlideMetadata).filter(
                DbSlideMetadata.batch_id == batch_id,
                DbSlideMetadata.slide_index == slide_idx
            ).all()
            for old_slide in existing_slides:
                db.query(DbComponentMetadata).filter(DbComponentMetadata.slide_id == old_slide.id).delete()
                db.query(DbRelationship).filter(DbRelationship.slide_id == old_slide.id).delete()
                db.delete(old_slide)
            db.commit()

            db_slide = DbSlideMetadata(
                batch_id=batch_id,
                slide_index=slide_idx,
                original_filename=original_filename,
                width=width,
                height=height,
                routing_status=routing_status,
                average_confidence=float(round(avg_conf, 2)),
                file_type=class_res["fileType"],
                content_type=class_res["contentType"],
                metrics_json=metrics
            )
            db.add(db_slide)
            db.flush()
            
            # Save components to DB
            for comp in components:
                db_comp = DbComponentMetadata(
                    slide_id=db_slide.id,
                    component_id=comp.id,
                    type=comp.type,
                    semantic_name=comp.semantic_name,
                    box_x=comp.box[0],
                    box_y=comp.box[1],
                    box_w=comp.box[2],
                    box_h=comp.box[3],
                    mask_path=comp.mask_path,
                    crop_path=comp.crop_path,
                    text=comp.text,
                    confidence=comp.confidence,
                    visible=comp.visible,
                    z_index=comp.z_index,
                    associated_label_id=comp.associated_label_id,
                    associated_object_id=comp.associated_object_id
                )
                db.add(db_comp)
                
            # Save relationships to DB
            for rel in relationships:
                db_rel = DbRelationship(
                    slide_id=db_slide.id,
                    label_id=rel.label_id,
                    label_text=rel.label_text,
                    arrow_id=rel.arrow_id,
                    target_id=rel.target_id
                )
                db.add(db_rel)
            db.commit()
        except Exception as e:
            print(f"Error logging slide details to DB: {e}")
            db.rollback()
        finally:
            db.close()
            
        log_file.close()
        return slide_meta

    def _compile_pptx(self, job: BatchJob):
        """Invokes the PPT generator to compile slides and saves the PPTX file."""
        batch_dir = os.path.join(STORAGE_DIR, "batches", job.batch_id)
        pptx_filename = "presentation.pptx"
        output_pptx_path = os.path.join(batch_dir, pptx_filename)
        
        import time
        t_start = time.time()
        # Compile PPTX
        self.ppt_generator.generate_batch_pptx(job, batch_dir, output_pptx_path)
        job.pptx_path = f"storage/batches/{job.batch_id}/{pptx_filename}"
        
        compile_time_ms = int((time.time() - t_start) * 1000)
        print(f"PPTX compiled in {compile_time_ms}ms")
        
        # Update metadata.json metrics for each slide
        for idx in range(len(job.slides)):
            metadata_file = os.path.join(batch_dir, "slides", f"slide_{idx}", "metadata", "metadata.json")
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        meta_data = json.load(f)
                    if "performance_metrics" in meta_data:
                        meta_data["performance_metrics"]["ppt_compile_time_ms"] = compile_time_ms
                    with open(metadata_file, "w", encoding="utf-8") as f:
                        json.dump(meta_data, f, indent=2)
                except Exception as ex:
                    print(f"Failed to update metadata compile metric for slide {idx}: {ex}")
