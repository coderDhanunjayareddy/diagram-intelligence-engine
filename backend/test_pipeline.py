import os
import cv2
import numpy as np
import shutil
from backend.pipeline.classifier import ComponentClassifier
from backend.pipeline.detectors.opencv_detector import OpenCVDetector
from backend.pipeline.segmenters.opencv_segmenter import OpenCVSegmenter
from backend.pipeline.ocr.ocr_engine import OCREngine
from backend.pipeline.reconstructor import ComponentReconstructor
from backend.pipeline.ppt_generator import PPTGenerator
from backend.pipeline.interfaces import BatchJob, SlideMetadata

def generate_test_image(path: str):
    """
    Draws a synthetic biology-style diagram using OpenCV.
    Draws white background, shapes, connecting arrows, and labels.
    """
    # 800x600 white canvas
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Draw "Mouth" Shape (Circle at top)
    cv2.circle(img, (400, 100), 30, (50, 50, 240), 2) # red boundary
    # Draw Label "Mouth"
    cv2.putText(img, "Mouth", (150, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    # Draw Arrow from Label to Shape
    cv2.arrowedLine(img, (230, 100), (360, 100), (80, 80, 80), 2, tipLength=0.1)
    
    # Draw "Stomach" Shape (Oval at middle-right)
    cv2.ellipse(img, (480, 280), (60, 40), 0, 0, 360, (240, 50, 50), 2) # blue boundary
    # Draw Label "Stomach"
    cv2.putText(img, "Stomach", (150, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    # Draw Arrow
    cv2.arrowedLine(img, (250, 280), (410, 280), (80, 80, 80), 2, tipLength=0.1)
    
    # Draw "Liver" Shape (Triangle at middle-left)
    pts = np.array([[300, 240], [360, 320], [280, 320]], np.int32)
    cv2.polylines(img, [pts], True, (50, 200, 50), 2) # green boundary
    # Draw Label "Liver"
    cv2.putText(img, "Liver", (550, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    # Draw Arrow
    cv2.arrowedLine(img, (540, 220), (370, 260), (80, 80, 80), 2, tipLength=0.1)
    
    # Draw "Intestine" Shape (Wavy polyline at bottom)
    cv2.rectangle(img, (340, 430), (460, 510), (150, 50, 150), 2)
    # Draw Label "Intestine"
    cv2.putText(img, "Intestine", (150, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    # Draw Arrow
    cv2.arrowedLine(img, (250, 470), (330, 470), (80, 80, 80), 2, tipLength=0.1)

    cv2.imwrite(path, img)
    print(f"Generated synthetic test diagram image at: {path}")

def run_test():
    test_dir = os.path.dirname(__file__)
    img_path = os.path.join(test_dir, "test_diagram.png")
    task_dir = os.path.join(test_dir, "test_task_dir")
    
    # Create slide subdirectory matching production layout
    slide_dir = os.path.join(task_dir, "slides", "slide_0")
    
    # 1. Generate image
    generate_test_image(img_path)
    
    # Clean output task folder
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)
    os.makedirs(slide_dir, exist_ok=True)
    
    # Copy original to slide folder
    original_copy = os.path.join(slide_dir, "original.png")
    shutil.copy(img_path, original_copy)
    
    print("\n--- Starting Pipeline Verification Test ---")
    
    # 2. File Classifier
    classifier = ComponentClassifier()
    class_res = classifier.classify(original_copy)
    print(f"1. Classification Result: {class_res}")
    assert class_res["fileType"] == "PNG", "File type must be classified as PNG"
    assert class_res["contentType"] == "Biology Diagram", "Content type must be classified as Biology Diagram"
    
    # 3. Object Detector
    detector = OpenCVDetector()
    detections = detector.detect(original_copy)
    print(f"2. Detected {len(detections)} component bounding boxes.")
    for d in detections:
        print(f"   - Box ID: {d.id}, Category: {d.category}, Coordinates: {d.box}, Confidence: {d.confidence}")
    assert len(detections) > 0, "Pipeline should detect at least a few bounding boxes"
    
    # 4. Image Segmenter
    segmenter = OpenCVSegmenter()
    segmentations = segmenter.segment(original_copy, detections, slide_dir)
    print(f"3. Segmented {len(segmentations)} transparent crops/masks.")
    assert len(segmentations) == len(detections), "Each detection should have a segmentation crop/mask output"
    
    # Verify crop files exist
    for seg in segmentations:
        crop_abs = os.path.join(slide_dir, seg.crop_path)
        mask_abs = os.path.join(slide_dir, seg.mask_path)
        assert os.path.exists(crop_abs), f"Crop file does not exist: {crop_abs}"
        assert os.path.exists(mask_abs), f"Mask file does not exist: {mask_abs}"
    print("   [OK] All transparent PNG masks and raw crops written to storage.")
    
    # 5. OCR Engine
    ocr = OCREngine()
    ocr_results = ocr.extract_text(original_copy, detections)
    print(f"4. Extracted OCR results: {[o.text for o in ocr_results]}")
    labels_count = len([d for d in detections if d.category == "label"])
    assert len(ocr_results) == labels_count, "OCR results count should match number of label detections"
    
    # 6. Component Reconstructor
    reconstructor = ComponentReconstructor()
    components = reconstructor.reconstruct(
        800, 600, detections, segmentations, ocr_results
    )
    print(f"5. Component Reconstructor linked layout. Total structured components: {len(components)}")
    for comp in components:
        if comp.type == "text_label":
            print(f"   - Text label '{comp.text}' (associated to object: {comp.associated_object_id})")
        elif comp.type == "image_object":
            print(f"   - Image object ID: {comp.id} (associated to label: {comp.associated_label_id})")
            
    # 7. PPTX Generator
    # Set up mock batch job
    slide_meta = SlideMetadata(
        slide_index=0,
        original_filename="original.png",
        width=800,
        height=600,
        components=components,
        routing_status="Manual Review",
        average_confidence=0.85,
        file_type="PNG",
        content_type="Biology Diagram"
    )
    job = BatchJob(
        batch_id="test_batch",
        status="completed",
        slides=[slide_meta],
        created_at=0.0
    )
    
    ppt_gen = PPTGenerator()
    pptx_path = os.path.join(task_dir, "presentation.pptx")
    ppt_gen.generate_batch_pptx(job, task_dir, pptx_path)
    
    assert os.path.exists(pptx_path), f"PowerPoint file was not created: {pptx_path}"
    print(f"6. PPTX Generation Result: Exported successfully to {pptx_path} (size: {os.path.getsize(pptx_path)} bytes)")
    
    # Cleanup test files
    os.remove(img_path)
    shutil.rmtree(task_dir)
    print("\n[Success] Automated Pipeline Verification test completed successfully!")

if __name__ == "__main__":
    run_test()
