import os
import shutil
import json
from backend.pipeline.classifier import ComponentClassifier
from backend.pipeline.detectors.opencv_detector import OpenCVDetector
from backend.pipeline.segmenters.opencv_segmenter import OpenCVSegmenter
from backend.pipeline.ocr.ocr_engine import OCREngine
from backend.pipeline.reconstructor import ComponentReconstructor
from backend.pipeline.ppt_generator import PPTGenerator
from backend.pipeline.interfaces import BatchJob, SlideMetadata

def dry_run_image(image_filename: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "test_data", image_filename)
    output_dir = os.path.join(base_dir, "dry_run_output")
    
    if not os.path.exists(input_path):
        print(f"Error: Input image not found at {input_path}")
        return
        
    print(f"\n==================================================")
    print(f"DRY RUNNING DIAGRAM DECOMPOSITION ON: {image_filename}")
    print(f"==================================================")
    
    # 1. Setup workspace structure
    slide_dir = os.path.join(output_dir, "slides", "slide_0")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(slide_dir, exist_ok=True)
    
    # Copy source image into workspace
    ext = os.path.splitext(image_filename)[1]
    shutil.copy(input_path, os.path.join(slide_dir, f"original{ext}"))
    original_copy = os.path.join(slide_dir, f"original{ext}")
    
    # 2. Step 1: Classification
    classifier = ComponentClassifier()
    class_res = classifier.classify(original_copy)
    print(f"[1] Classification Result: {class_res}")
    
    # 3. Step 2: Detection
    detector = OpenCVDetector()
    detections = detector.detect(original_copy)
    print(f"[2] Bounding Box Detector found {len(detections)} elements.")
    for d in detections[:10]:
        print(f"    - ID: {d.id}, Category: {d.category}, Box: {d.box}, Confidence: {d.confidence:.2f}")
    if len(detections) > 10:
        print(f"    - ... and {len(detections) - 10} more elements")
        
    # 4. Step 3: Segmentation
    segmenter = OpenCVSegmenter()
    segmentations = segmenter.segment(original_copy, detections, slide_dir)
    print(f"[3] Segmenter created {len(segmentations)} crop/mask pairs in storage.")
    
    # 5. Step 4: OCR Engine
    ocr = OCREngine()
    ocr_results = ocr.extract_text(original_copy, detections)
    print(f"[4] OCR Engine extracted {len(ocr_results)} text values:")
    for o in ocr_results[:10]:
        print(f"    - Text: '{o.text}' at Box: {o.box}")
    if len(ocr_results) > 10:
        print(f"    - ... and {len(ocr_results) - 10} more text labels")
        
    # 6. Step 5: Reconstruction
    reconstructor = ComponentReconstructor()
    # Read actual image dimensions
    import cv2
    img = cv2.imread(original_copy)
    h, w = img.shape[:2] if img is not None else (600, 800)
    
    components = reconstructor.reconstruct(w, h, detections, segmentations, ocr_results)
    print(f"[5] Reconstructor compiled {len(components)} final components.")
    
    # 7. Step 6: PowerPoint Compile
    slide_meta = SlideMetadata(
        slide_index=0,
        original_filename=image_filename,
        width=w,
        height=h,
        components=components,
        routing_status="Warning",
        average_confidence=0.85,
        file_type=class_res["fileType"],
        content_type=class_res["contentType"]
    )
    job = BatchJob(
        batch_id="dry_run_batch",
        status="completed",
        slides=[slide_meta],
        created_at=0.0
    )
    
    ppt_gen = PPTGenerator()
    pptx_filename = f"dry_run_{os.path.splitext(image_filename)[0]}.pptx"
    pptx_path = os.path.join(base_dir, pptx_filename)
    ppt_gen.generate_batch_pptx(job, output_dir, pptx_path)
    
    print(f"[6] PowerPoint slides successfully compiled!")
    print(f"    - Output PPTX location: {pptx_path}")
    print(f"    - Output PPTX size: {os.path.getsize(pptx_path)} bytes")
    print(f"==================================================\n")

if __name__ == "__main__":
    # Test on a biology cell diagram and a physics circuit diagram
    dry_run_image("biology_plant_cell.png")
    dry_run_image("physics_electric_circuit.png")
