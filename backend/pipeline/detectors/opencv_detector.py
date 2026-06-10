import cv2
import numpy as np
from typing import List
from backend.pipeline.interfaces import IDetector, DetectionResult

class OpenCVDetector(IDetector):
    def detect(self, image_path: str) -> List[DetectionResult]:
        """
        Detects elements in educational diagrams using computer vision.
        Categorizes elements as 'label' (text boxes), 'arrow' (pointing arrows/lines),
        and 'object' (illustrations, organs, nodes).
        """
        img = cv2.imread(image_path)
        if img is None:
            return []
            
        h_img, w_img, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to preserve edges while removing texture noise
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Adaptive Thresholding to handle local illumination differences
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        detections = []
        detection_counter = 0
        
        # --- 1. Label Detection (Horizontal merging for text blocks) ---
        # Dilate horizontally to join individual letters/words in labels
        kernel_label = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated_label = cv2.dilate(thresh, kernel_label, iterations=2)
        
        contours_lbl, hierarchy = cv2.findContours(
            dilated_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        label_boxes = []
        for i, cnt in enumerate(contours_lbl):
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Constraints for text labels in diagrams:
            # - Not too small (noise)
            # - Aspect ratio is usually horizontal (w > h) or small bounding square
            # - Height is reasonable for typography (usually 10px to 60px)
            # - Doesn't span the entire width of the image (unless it's a title)
            if w > 8 and h > 8 and h < 80 and w < w_img * 0.4:
                # Calculate basic text layout metrics for confidence estimation
                # Clean, sharp-edged horizontal boxes have higher confidence
                aspect = w / h
                confidence = 0.82
                if aspect > 1.2 and aspect < 6.0:
                    confidence += 0.08
                
                label_boxes.append((x, y, w, h, confidence))
                
        # --- 2. Object and Arrow Detection ---
        # Use Canny edge detection followed by slight dilation to connect outlines
        edges = cv2.Canny(blurred, 30, 150)
        kernel_shape = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_shape = cv2.dilate(edges, kernel_shape, iterations=1)
        
        contours_shape, _ = cv2.findContours(
            dilated_shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        object_boxes = []
        arrow_boxes = []
        
        for cnt in contours_shape:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            
            if w < 10 or h < 10:
                continue # Skip tiny noise
                
            # Arrow detection:
            # - Typically long, narrow structures
            # - Aspect ratio is extreme (width >> height or height >> width)
            # - Or rectangular perimeter fits poorly compared to actual contour area
            aspect = w / h
            is_extreme_aspect = aspect > 3.5 or aspect < 0.28
            
            if is_extreme_aspect and (w < w_img * 0.3) and (h < h_img * 0.3):
                confidence = min(0.88, float(0.70 + 0.15 * (1.0 - (area / (w * h)))))
                arrow_boxes.append((x, y, w, h, confidence))
                continue
                
            # Shape/Illustration detection:
            # - Needs to be larger than labels
            # - Higher area fill density or distinct shapes
            if w > 25 and h > 25 and (w < w_img * 0.95 or h < h_img * 0.95):
                # Check circularity or complexity
                approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
                confidence = 0.85
                if len(approx) > 8: # organic shape like an organ/cell
                    confidence += 0.05
                elif len(approx) in [3, 4]: # geometric shape
                    confidence += 0.07
                
                object_boxes.append((x, y, w, h, confidence))
                
        # --- 3. Non-Maximum Suppression (NMS) and Overlap Resolution ---
        # Resolve overlapping detections of different types.
        # Rule: If a 'label' overlaps with an 'object', keep both but adjust z-index later.
        # If two 'object' boxes overlap significantly (IoU > 0.7), merge or keep the larger one.
        
        def calculate_iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
            yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
            
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = boxA[2] * boxA[3]
            boxBArea = boxB[2] * boxB[3]
            
            # IoU
            iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
            return iou

        # Filter duplicates in labels
        filtered_labels = []
        for l in sorted(label_boxes, key=lambda b: b[4], reverse=True):
            overlap = False
            for fl in filtered_labels:
                if calculate_iou(l, fl) > 0.4:
                    overlap = True
                    break
            if not overlap:
                filtered_labels.append(l)
                
        # Filter duplicates in objects
        filtered_objects = []
        for o in sorted(object_boxes, key=lambda b: b[2] * b[3], reverse=True):
            overlap = False
            for fo in filtered_objects:
                # If nested, we keep both (since smaller object might be inside a larger outline)
                # If they are almost identical boundaries, merge
                if calculate_iou(o, fo) > 0.85:
                    overlap = True
                    break
            if not overlap:
                filtered_objects.append(o)
                
        # Remove label areas from object outline boxes to avoid double cropping the label
        final_objects = []
        for o in filtered_objects:
            # Check if this object box is just a container for a text label
            is_just_label = False
            for l in filtered_labels:
                # If label covers > 80% of this object container, skip the object
                intersect_w = max(0, min(o[0]+o[2], l[0]+l[2]) - max(o[0], l[0]))
                intersect_h = max(0, min(o[1]+o[3], l[1]+l[3]) - max(o[1], l[1]))
                intersect_area = intersect_w * intersect_h
                label_area = l[2] * l[3]
                if label_area > 0 and (intersect_area / label_area) > 0.85 and (o[2]*o[3] < label_area * 1.5):
                    is_just_label = True
                    break
            if not is_just_label:
                final_objects.append(o)

        # Build final DetectionResult list
        for l in filtered_labels:
            detection_counter += 1
            detections.append(DetectionResult(
                id=f"det_lbl_{detection_counter}",
                category="label",
                box=[int(l[0]), int(l[1]), int(l[2]), int(l[3])],
                confidence=float(round(l[4], 2))
            ))
            
        for o in final_objects:
            detection_counter += 1
            detections.append(DetectionResult(
                id=f"det_obj_{detection_counter}",
                category="object",
                box=[int(o[0]), int(o[1]), int(o[2]), int(o[3])],
                confidence=float(round(o[4], 2))
            ))
            
        for a in arrow_boxes:
            # Ensure arrow is not overlapping a label
            label_overlap = False
            for l in filtered_labels:
                if calculate_iou(a, l) > 0.3:
                    label_overlap = True
                    break
            if not label_overlap:
                detection_counter += 1
                detections.append(DetectionResult(
                    id=f"det_arr_{detection_counter}",
                    category="arrow",
                    box=[int(a[0]), int(a[1]), int(a[2]), int(a[3])],
                    confidence=float(round(a[4], 2))
                ))
                
        return detections
