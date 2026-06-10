import os
import cv2
import numpy as np
from typing import List
from backend.pipeline.interfaces import ISegmenter, DetectionResult, SegmentationResult

class OpenCVSegmenter(ISegmenter):
    def segment(self, image_path: str, detections: List[DetectionResult], task_dir: str) -> List[SegmentationResult]:
        """
        Segments detected objects out of the main image.
        Crops bounding boxes and applies transparency masks by keying out background pixels.
        Saves crops and masks to task_dir/artifacts/ and returns SegmentationResults.
        """
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return []
            
        h_img, w_img = img.shape[:2]
        
        # Ensure image has 3 channels for processing
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
        # Sample global background color from 4 corners of the full slide image
        corners = [img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]]
        corners_bgr = [c[:3] for c in corners]
        global_bg = np.mean(corners_bgr, axis=0)
            
        # Create artifacts folder if it doesn't exist
        artifacts_dir = os.path.join(task_dir, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        
        results = []
        
        for det in detections:
            x, y, w, h = det.box
            
            # Constrain bounding boxes to image boundaries
            x_start = max(0, min(x, w_img - 1))
            y_start = max(0, min(y, h_img - 1))
            x_end = max(1, min(x + w, w_img))
            y_end = max(1, min(y + h, h_img))
            
            crop_w = x_end - x_start
            crop_h = y_end - y_start
            
            if crop_w <= 0 or crop_h <= 0:
                continue
                
            crop = img[y_start:y_end, x_start:x_end]
            
            # Save raw crop
            crop_filename = f"crop_{det.id}.png"
            crop_path = os.path.join(artifacts_dir, crop_filename)
            cv2.imwrite(crop_path, crop)
            
            # --- Transparent Mask Generation ---
            # Create RGBA canvas for mask
            rgba_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
            
            # Use global background color of the slide to prevent color contamination from organ borders
            mean_bg = global_bg
            
            # Calculate distance of each pixel to the background color
            diff = np.abs(crop.astype(np.float32) - mean_bg)
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            
            # If distance is small, it's background -> set alpha to 0
            # For labels (text), we want a very strict threshold to preserve thin letters.
            # For large objects, we can be slightly more lenient to clear white borders.
            threshold = 30.0 if det.category == "label" else 45.0
            
            # Create mask: 255 for foreground, 0 for background
            if det.category == "arrow":
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                alpha_channel = np.where(gray_crop < 30, 255, 0).astype(np.uint8)
            else:
                alpha_channel = np.where(dist < threshold, 0, 255).astype(np.uint8)
            
            # Planet-specific geometric cutout mask
            is_planet = det.label in ["sun", "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"]
            if is_planet:
                alpha_channel = np.zeros((crop_h, crop_w), dtype=np.uint8)
                if det.label == "saturn":
                    center = (crop_w // 2, crop_h // 2)
                    axes = (crop_w // 2 - 2, crop_h // 2 - 2)
                    cv2.ellipse(alpha_channel, center, axes, 0, 0, 360, 255, thickness=-1)
                else:
                    center = (crop_w // 2, crop_h // 2)
                    radius = min(crop_w, crop_h) // 2 - 1
                    cv2.circle(alpha_channel, center, radius, 255, thickness=-1)
            
            # For shapes/objects, we run GrabCut to isolate the central foreground organ from touching adjacent border shapes
            if not is_planet and det.category in ["object", "shape"] and crop_w > 15 and crop_h > 15:
                # Initialize GrabCut mask and models
                mask_gc = np.zeros((crop_h, crop_w), np.uint8)
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)
                
                # Determine border width dynamically (narrower for small or thin shapes)
                border = min(3, max(1, min(crop_w, crop_h) // 12))
                rect = (border, border, crop_w - 2 * border, crop_h - 2 * border)
                
                try:
                    # Run GrabCut
                    cv2.grabCut(crop, mask_gc, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
                    # Keep FGD (1) and PR_FGD (3)
                    alpha_channel = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
                    
                    # Also, mask out any pure white pixels (since they are definitely background)
                    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    white_mask = np.where(gray_crop > 245, 0, 255).astype(np.uint8)
                    alpha_channel = cv2.bitwise_and(alpha_channel, white_mask)
                    
                except Exception as e:
                    print(f"GrabCut failed for {det.id}: {e}, falling back to OTSU")
                    # Fallback: OTSU thresholding
                    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    bg_gray = 0.299 * mean_bg[2] + 0.587 * mean_bg[1] + 0.114 * mean_bg[0]
                    thresh_type = cv2.THRESH_BINARY_INV if bg_gray >= 127 else cv2.THRESH_BINARY
                    _, local_thresh = cv2.threshold(gray_crop, 0, 255, thresh_type + cv2.THRESH_OTSU)
                    combined_mask = cv2.bitwise_or(alpha_channel, local_thresh)
                    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    alpha_channel = np.zeros_like(combined_mask)
                    for cnt in contours:
                        if cv2.contourArea(cnt) > 5:
                            cv2.drawContours(alpha_channel, [cnt], -1, 255, thickness=-1)
                
            rgba_crop[:, :, 3] = alpha_channel
            
            mask_filename = f"mask_{det.id}.png"
            mask_path = os.path.join(artifacts_dir, mask_filename)
            cv2.imwrite(mask_path, rgba_crop)
            
            # Relative paths for JSON metadata serialization
            if os.path.basename(task_dir.rstrip("/\\")) == "masks":
                rel_crop_path = os.path.join("masks", "artifacts", crop_filename)
                rel_mask_path = os.path.join("masks", "artifacts", mask_filename)
            else:
                rel_crop_path = os.path.join("artifacts", crop_filename)
                rel_mask_path = os.path.join("artifacts", mask_filename)
            
            # Calculate a basic segmentation confidence based on contrast
            seg_confidence = float(np.mean(alpha_channel) / 255.0)
            # Normalize confidence (avoid 0 or 1 extremes)
            seg_confidence = max(0.65, min(0.95, float(round(0.70 + 0.25 * (1.0 - seg_confidence), 2))))
            
            results.append(SegmentationResult(
                detection_id=det.id,
                mask_path=rel_mask_path.replace("\\", "/"),
                crop_path=rel_crop_path.replace("\\", "/"),
                box=[int(x_start), int(y_start), int(crop_w), int(crop_h)],
                confidence=seg_confidence
            ))
            
        return results
