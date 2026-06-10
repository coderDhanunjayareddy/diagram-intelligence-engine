import math
from typing import List, Optional, Tuple
from backend.pipeline.interfaces import IReconstructor, DetectionResult, SegmentationResult, OCRResult, ComponentMetadata

class ComponentReconstructor(IReconstructor):
    def reconstruct(
        self,
        width: int,
        height: int,
        detections: List[DetectionResult],
        segmentations: List[SegmentationResult],
        ocr_results: List[OCRResult]
    ) -> List[ComponentMetadata]:
        """
        Reconstructs the diagram components.
        Links labels (OCR text) to their nearest physical objects, assigns Z-indexes,
        and constructs the parent/child component graph.
        """
        components = []
        seg_map = {s.detection_id: s for s in segmentations}
        
        # 1. First, create basic components for all detections
        labels = []
        objects = []
        arrows = []
        
        for det in detections:
            seg = seg_map.get(det.id)
            mask_path = seg.mask_path if seg else None
            crop_path = seg.crop_path if seg else None
            
            if det.category == "label":
                # Match label detection box with OCR results
                # Find OCR result with highest overlap/proximity
                matching_ocr = self._find_matching_ocr(det.box, ocr_results)
                text_val = matching_ocr.text if matching_ocr else "Label"
                ocr_conf = matching_ocr.confidence if matching_ocr else 0.50
                
                comp = ComponentMetadata(
                    id=det.id,
                    type="text_label",
                    box=det.box,
                    text=text_val,
                    confidence=float(round(0.6 * det.confidence + 0.4 * ocr_conf, 2)),
                    z_index=10 # labels go on top
                )
                labels.append(comp)
                
            elif det.category == "arrow":
                comp = ComponentMetadata(
                    id=det.id,
                    type="arrow",
                    box=det.box,
                    mask_path=mask_path,
                    crop_path=crop_path,
                    confidence=det.confidence,
                    z_index=5 # arrows go below labels but above shapes
                )
                arrows.append(comp)
                
            elif det.category == "object":
                comp = ComponentMetadata(
                    id=det.id,
                    type="image_object",
                    box=det.box,
                    mask_path=mask_path,
                    crop_path=crop_path,
                    confidence=det.confidence,
                    z_index=2 # shapes go near bottom
                )
                objects.append(comp)
                
        # 2. Geometry-Based Label & Object Linking
        # For each label, find the nearest object. If they are close, associate them.
        for label in labels:
            nearest_obj = self._find_nearest_component(label.box, objects)
            if nearest_obj:
                dist = self._calculate_box_distance(label.box, nearest_obj.box)
                # If they are within 180 pixels, associate them
                if dist < 180.0:
                    label.associated_object_id = nearest_obj.id
                    nearest_obj.associated_label_id = label.id
                    
        # 3. Arrow Routing Associations
        # For each arrow, associate it with the source label and target object
        for arrow in arrows:
            nearest_label = self._find_nearest_component(arrow.box, labels)
            nearest_obj = self._find_nearest_component(arrow.box, objects)
            
            if nearest_label:
                arrow.associated_label_id = nearest_label.id
            if nearest_obj:
                arrow.associated_object_id = nearest_obj.id

        # Combine all components
        components.extend(objects)
        components.extend(arrows)
        components.extend(labels)
        
        # Sort components by Z-index so they draw/layer correctly
        components.sort(key=lambda c: c.z_index)
        
        return components

    def _find_matching_ocr(self, box: List[int], ocr_results: List[OCRResult]) -> Optional[OCRResult]:
        """
        Finds the OCRResult that overlaps or is closest to the detection bounding box.
        """
        best_ocr = None
        best_overlap = -1.0
        
        for ocr in ocr_results:
            overlap = self._calculate_iou(box, ocr.box)
            if overlap > best_overlap:
                best_overlap = overlap
                best_ocr = ocr
                
        # If there's an OCR result with any overlap, return it
        if best_overlap > 0.1:
            return best_ocr
            
        # Otherwise find the closest OCR result by distance
        if ocr_results:
            closest_ocr = min(ocr_results, key=lambda o: self._calculate_box_distance(box, o.box))
            dist = self._calculate_box_distance(box, closest_ocr.box)
            if dist < 30.0:
                return closest_ocr
                
        return None

    def _find_nearest_component(self, target_box: List[int], candidates: List[ComponentMetadata]) -> Optional[ComponentMetadata]:
        """
        Finds the nearest candidate component to the target bounding box.
        """
        if not candidates:
            return None
            
        return min(candidates, key=lambda c: self._calculate_box_distance(target_box, c.box))

    def _calculate_box_distance(self, boxA: List[int], boxB: List[int]) -> float:
        """
        Calculates the Euclidean distance between the centers of two bounding boxes.
        """
        cxA = boxA[0] + boxA[2] / 2
        cyA = boxA[1] + boxA[3] / 2
        cxB = boxB[0] + boxB[2] / 2
        cyB = boxB[1] + boxB[3] / 2
        
        return math.sqrt((cxA - cxB) ** 2 + (cyA - cyB) ** 2)

    def _calculate_iou(self, boxA: List[int], boxB: List[int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        
        return interArea / float(boxAArea + boxBArea - interArea + 1e-5)
