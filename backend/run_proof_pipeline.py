import os
from typing import List
import time
import json
import shutil
import cv2
import numpy as np

# Database imports
from backend.database import SessionLocal, engine, Base
from backend.models import DbBatchJob, DbSlideMetadata, DbComponentMetadata, DbRelationship
from backend.storage import StorageProvider

# Pipeline imports
from backend.pipeline.classifier import ComponentClassifier
from backend.pipeline.detectors.detector_provider import DetectorProvider
from backend.pipeline.segmenters.segmenter_provider import SegmenterProvider
from backend.pipeline.ocr.ocr_provider import OCRProvider
from backend.pipeline.understanding_engine import DiagramUnderstandingEngine
from backend.pipeline.ppt_generator import PPTGenerator
from backend.pipeline.interfaces import BatchJob, SlideMetadata, ComponentMetadata

# Define workspace directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "test_data"))
BATCHES_DIR = os.path.join(BASE_DIR, "storage", "batches")
PROOF_BATCH_ID = "proof_batch"
PROOF_BATCH_DIR = os.path.join(BATCHES_DIR, PROOF_BATCH_ID)

# Combined slide test set
TEST_FILES = [
    "biology_digestive_system.png",
    "biology_plant_cell.png",
    "physics_solar_system.png",
    "physics_water_cycle.png",
    "physics_electric_circuit.png",
    "flowchart_student_admission_process.png",
    "difficult_textbook_scan_1.png"
]

def generate_erased_background(original_path: str, components: List[ComponentMetadata], dest_path: str):
    """Fills the bounding boxes of visible components using LaMa inpainting."""
    img = cv2.imread(original_path)
    if img is None:
        return
        
    h, w = img.shape[:2]
    inpaint_mask = np.zeros((h, w), dtype=np.uint8)
    has_elements = False
    
    for comp in components:
        if not comp.visible:
            continue
        x, y, cw, ch = comp.box
        x_start = max(0, x - 3)
        y_start = max(0, y - 3)
        x_end = min(w, x + cw + 3)
        y_end = min(h, y + ch + 3)
        cv2.rectangle(inpaint_mask, (x_start, y_start), (x_end, y_end), 255, -1)
        has_elements = True
        
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    inpainted_success = False
    if has_elements:
        try:
            from simple_lama_inpainting import SimpleLama
            from PIL import Image
            
            simple_lama = SimpleLama()
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pil_mask = Image.fromarray(inpaint_mask).convert("L")
            
            inpainted_pil = simple_lama(pil_img, pil_mask)
            inpainted_bg = cv2.cvtColor(np.array(inpainted_pil), cv2.COLOR_RGB2BGR)
            cv2.imwrite(dest_path, inpainted_bg)
            inpainted_success = True
        except Exception as e:
            print(f"[ProofPipeline] LaMa inpainting failed: {e}. Using solid fallback.")
            
    if not inpainted_success:
        corners = [img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]]
        bg_color = np.mean(corners, axis=0).astype(int).tolist()
        bg_color_tuple = (bg_color[0], bg_color[1], bg_color[2])
        
        for comp in components:
            if not comp.visible:
                continue
            x, y, cw, ch = comp.box
            x_start = max(0, x - 2)
            y_start = max(0, y - 2)
            x_end = min(w, x + cw + 2)
            y_end = min(h, y + ch + 2)
            cv2.rectangle(img, (x_start, y_start), (x_end, y_end), bg_color_tuple, -1)
        cv2.imwrite(dest_path, img)


def run_proof_pipeline():
    print("==========================================================")
    print("STARTING V2 PROOF-OF-CAPABILITY DIAGRAM INTELLIGENCE PIPELINE")
    print("==========================================================\n")
    
    # 1. Initialize database tables
    Base.metadata.create_all(bind=engine)
    
    # Self-healing V3 schema migrations
    from sqlalchemy import text
    with engine.connect() as conn:
        for col, col_type in [("is_occluded", "BOOLEAN DEFAULT 0"), 
                              ("amodal_mask_path", "VARCHAR"),
                              ("polygon_vertices_json", "VARCHAR"),
                              ("reconstruction_confidence", "FLOAT"),
                              ("reconstruction_source", "VARCHAR")]:
            try:
                conn.execute(text(f"SELECT {col} FROM components_metadata LIMIT 1"))
            except Exception:
                print(f"[Migration] Adding {col} column to components_metadata...")
                conn.execute(text(f"ALTER TABLE components_metadata ADD COLUMN {col} {col_type}"))
                conn.commit()
                
        try:
            conn.execute(text("SELECT occlusion_graph_json FROM slides_metadata LIMIT 1"))
        except Exception:
            print("[Migration] Adding occlusion_graph_json column to slides_metadata...")
            conn.execute(text("ALTER TABLE slides_metadata ADD COLUMN occlusion_graph_json VARCHAR"))
            conn.commit()
            
    db = SessionLocal()

    
    # Clean previous batch run
    if os.path.exists(PROOF_BATCH_DIR):
        shutil.rmtree(PROOF_BATCH_DIR)
    os.makedirs(PROOF_BATCH_DIR, exist_ok=True)
    
    # Initialize DB Batch row
    db_batch = DbBatchJob(
        batch_id=PROOF_BATCH_ID,
        status="processing",
        created_at=time.time()
    )
    db.merge(db_batch)
    db.commit()
    
    # Instantiate Providers
    classifier = ComponentClassifier()
    detector = DetectorProvider.get_detector("GroundingDINO")
    segmenter = SegmenterProvider.get_segmenter("SAM2")
    ocr = OCRProvider.get_ocr("PaddleOCR")
    understanding_engine = DiagramUnderstandingEngine()
    ppt_generator = PPTGenerator()
    storage = StorageProvider.get_provider()
    
    slides_metadata: List[SlideMetadata] = []
    
    for idx, filename in enumerate(TEST_FILES):
        src_path = os.path.join(TEST_DATA_DIR, filename)
        if not os.path.exists(src_path):
            print(f"Skipping missing file: {filename}")
            continue
            
        print(f"--> Processing Slide {idx}: {filename}...")
        
        # Setup V2 Workspace Directories for this slide task
        slide_task_dir = os.path.join(PROOF_BATCH_DIR, "slides", f"slide_{idx}")
        dir_original = os.path.join(slide_task_dir, "original")
        dir_detections = os.path.join(slide_task_dir, "detections")
        dir_masks = os.path.join(slide_task_dir, "masks")
        dir_ocr = os.path.join(slide_task_dir, "ocr")
        dir_metadata = os.path.join(slide_task_dir, "metadata")
        dir_ppt = os.path.join(slide_task_dir, "ppt")
        dir_logs = os.path.join(slide_task_dir, "logs")
        
        for d in [dir_original, dir_detections, dir_masks, dir_ocr, dir_metadata, dir_ppt, dir_logs]:
            os.makedirs(d, exist_ok=True)
            
        log_file = open(os.path.join(dir_logs, "pipeline.log"), "w", encoding="utf-8")
        
        def log_message(msg):
            # Print to stdout, encoding with backslashreplace for safe terminal rendering
            print(f"    {msg}".encode('ascii', errors='backslashreplace').decode('ascii'))
            log_file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
            
        # Copy original into original/
        original_dest = os.path.join(dir_original, filename)
        shutil.copy(src_path, original_dest)
        
        # --- Stage 1: File Classification ---
        t_start = time.time()
        class_res = classifier.classify(original_dest)
        t_classification = int((time.time() - t_start) * 1000)
        log_message(f"Stage 1: Classification -> Type: {class_res['contentType']}, File: {class_res['fileType']} ({t_classification}ms)")
        
        # Determine image dimensions
        img = cv2.imread(original_dest)
        h, w = img.shape[:2] if img is not None else (600, 800)
        
        # --- Stage 2: Bounding Box Detection ---
        t_start = time.time()
        detections = detector.detect(original_dest)
        t_detection = int((time.time() - t_start) * 1000)
        log_message(f"Stage 2: Detection -> Found {len(detections)} elements ({t_detection}ms)")
        
        # Save detections.json
        detections_json_path = os.path.join(dir_detections, "detections.json")
        with open(detections_json_path, "w", encoding="utf-8") as f:
            json.dump([d.dict() for d in detections], f, indent=2)
            
        # --- Stage 3: Segmentation ---
        t_start = time.time()
        # Segment and write transparent PNG crops into masks/ directory
        segmentations = segmenter.segment(original_dest, detections, dir_masks)
        t_segmentation = int((time.time() - t_start) * 1000)
        log_message(f"Stage 3: Segmentation -> Extracted {len(segmentations)} shapes/arrows ({t_segmentation}ms)")
        
        # Save segmentation.json
        segmentation_json_path = os.path.join(dir_masks, "segmentation.json")
        with open(segmentation_json_path, "w", encoding="utf-8") as f:
            json.dump([s.dict() for s in segmentations], f, indent=2)
            
        # --- Stage 4: OCR ---
        t_start = time.time()
        ocr_results = ocr.extract_text(original_dest, detections)
        t_ocr = int((time.time() - t_start) * 1000)
        log_message(f"Stage 4: OCR -> Extracted {len(ocr_results)} text boxes ({t_ocr}ms)")
        
        # Save ocr.json
        ocr_json_path = os.path.join(dir_ocr, "ocr.json")
        with open(ocr_json_path, "w", encoding="utf-8") as f:
            json.dump([o.dict() for o in ocr_results], f, indent=2)
            
        # --- Stage 5 & 6: Diagram Understanding Engine & Component Reconstruction ---
        t_start = time.time()
        components, relationships = understanding_engine.process_diagram(
            w, h, detections, segmentations, ocr_results
        )
        t_understanding = int((time.time() - t_start) * 1000)
        log_message(f"Stage 5: Understanding & Reconstruction -> Compiled {len(components)} components with {len(relationships)} relationships ({t_understanding}ms)")
        
        # Print relationship details
        for rel in relationships:
            log_message(f"      Mapped: [Label: '{rel.label_text}'] -> [Arrow: {rel.arrow_id}] -> [Target Shape ID: {rel.target_id}]")
            
        # --- Stage 5b: V3.5 Generic Amodal Segmentation ---
        t_start_v3 = time.time()
        occlusion_graph_json = "[]"
        try:
            from backend.pipeline.path_inference_engine import AmodalOcclusionEngine
            from backend.pipeline.amodal_renderer import AmodalSAMSegmenter
            
            solver = AmodalOcclusionEngine()
            renderer = AmodalSAMSegmenter()
            
            occlusion_graph, occlusion_pairs = solver.solve_occlusion(original_dest, components)
            occlusion_graph_json = json.dumps(occlusion_graph)
            
            # Save occlusion graph under metadata/occlusion_graph.json
            occlusion_graph_path = os.path.join(dir_metadata, "occlusion_graph.json")
            with open(occlusion_graph_path, "w", encoding="utf-8") as f:
                json.dump(occlusion_graph, f, indent=2)
                
            for occluded_comp, occluder_comp in occlusion_pairs:
                log_message(f"V3.5: Reconstructing amodal layer '{occluded_comp.semantic_name}' ({occluded_comp.id}) occluded by '{occluder_comp.semantic_name}' ({occluder_comp.id})...")
                rel_amodal_path = renderer.render_amodal(original_dest, slide_task_dir, occluded_comp, occluder_comp)
                occluded_comp.amodal_mask_path = rel_amodal_path
                log_message(f"V3.5: Reconstructed amodal mask successfully. Saved to {rel_amodal_path}.")
        except Exception as e_v3:
            log_message(f"V3.5: Error during amodal segmentation: {e_v3}")
        t_reconstruction = int((time.time() - t_start_v3) * 1000)
        log_message(f"Stage 5b: Amodal Layer Reconstruction complete ({t_reconstruction}ms)")
        
        # Generate erased background image in masks/background.png
        background_dest = os.path.join(dir_masks, "background.png")
        generate_erased_background(original_dest, components, background_dest)
        
        # Quality score checks
        conf_scores = [c.confidence for c in components]
        avg_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0.85
        
        # Quality Engine Routing:
        # Confidence > 0.90 -> Auto Export, 0.70-0.90 -> Warning, <0.70 -> Manual Review
        if avg_conf > 0.90:
            routing_status = "Auto Export"
        elif avg_conf >= 0.70:
            routing_status = "Warning"
        else:
            routing_status = "Manual Review"
            
        # Metrics json structure
        metrics = {
            "classification_time_ms": t_classification,
            "detection_time_ms": t_detection,
            "segmentation_time_ms": t_segmentation,
            "ocr_time_ms": t_ocr,
            "understanding_time_ms": t_understanding,
            "reconstruction_time_ms": t_reconstruction,
            "ppt_compile_time_ms": 0 # compiled later
        }
        
        # Create standard SlideMetadata object
        slide_meta = SlideMetadata(
            slide_index=idx,
            original_filename=filename,
            width=w,
            height=h,
            components=components,
            routing_status=routing_status,
            average_confidence=float(round(avg_conf, 2)),
            file_type=class_res["fileType"],
            content_type=class_res["contentType"],
            occlusion_graph_json=occlusion_graph_json
        )
        slides_metadata.append(slide_meta)
        
        # Write metadata.json (includes relationships and latency metrics)
        slide_meta_dict = slide_meta.dict()
        slide_meta_dict["relationships"] = [r.dict() for r in relationships]
        slide_meta_dict["performance_metrics"] = metrics
        
        metadata_json_path = os.path.join(dir_metadata, "metadata.json")
        with open(metadata_json_path, "w", encoding="utf-8") as f:
            json.dump(slide_meta_dict, f, indent=2)
            
        # Write DB Slide Metadata Row
        db_slide = DbSlideMetadata(
            batch_id=PROOF_BATCH_ID,
            slide_index=idx,
            original_filename=filename,
            width=w,
            height=h,
            routing_status=routing_status,
            average_confidence=float(round(avg_conf, 2)),
            file_type=class_res["fileType"],
            content_type=class_res["contentType"],
            metrics_json=metrics,
            occlusion_graph_json=occlusion_graph_json
        )
        db.add(db_slide)
        db.flush() # flush to get primary key ID
        
        # Write DB Components Rows
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
                associated_object_id=comp.associated_object_id,
                is_occluded=comp.is_occluded,
                amodal_mask_path=comp.amodal_mask_path,
                polygon_vertices_json=comp.polygon_vertices_json,
                reconstruction_confidence=comp.reconstruction_confidence,
                reconstruction_source=comp.reconstruction_source
            )
            db.add(db_comp)
            
        # Write DB Relationship Rows
        for rel in relationships:
            db_rel = DbRelationship(
                slide_id=db_slide.id,
                label_id=rel.label_id,
                label_text=rel.label_text,
                arrow_id=rel.arrow_id,
                target_id=rel.target_id
            )
            db.add(db_rel)
            
        log_file.close()
        print(f"   [OK] Slide {idx} processed and saved to V2 workspace.")
        
    db.commit()
    
    # --- PPT Generation Stage ---
    # Create mock BatchJob for PPT compiler
    job = BatchJob(
        batch_id=PROOF_BATCH_ID,
        status="completed",
        slides=slides_metadata,
        created_at=time.time()
    )
    
    t_start = time.time()
    pptx_filename = "dry_run_combined_proof.pptx"
    output_pptx_path = os.path.join(BASE_DIR, pptx_filename)
    
    # Generate presentation
    ppt_generator.generate_batch_pptx(job, PROOF_BATCH_DIR, output_pptx_path)
    t_ppt = int((time.time() - t_start) * 1000)
    
    # Update PPT export path in DB
    db_batch = db.query(DbBatchJob).filter(DbBatchJob.batch_id == PROOF_BATCH_ID).first()
    if db_batch:
        db_batch.status = "completed"
        db_batch.completed_at = time.time()
        db_batch.pptx_path = pptx_filename
        db.commit()
        
    # Update metrics for each slide with the ppt compilation latency
    db_slides = db.query(DbSlideMetadata).filter(DbSlideMetadata.batch_id == PROOF_BATCH_ID).all()
    for ds in db_slides:
        m = ds.metrics_json
        m["ppt_compile_time_ms"] = t_ppt
        ds.metrics_json = m
    db.commit()
    db.close()
    
    print("\n==========================================================")
    print("V2 PIPELINE PROOF-OF-CAPABILITY COMPLETE")
    print(f"Compiled PPTX presentation: {output_pptx_path}")
    print(f"PPTX file size: {os.path.getsize(output_pptx_path)} bytes")
    print(f"All outputs cataloged in database: storage/diagram_v2.db")
    print("==========================================================")

if __name__ == "__main__":
    run_proof_pipeline()
