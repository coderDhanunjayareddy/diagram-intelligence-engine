import math
from typing import List, Optional, Tuple, Dict
from backend.pipeline.interfaces import (
    DetectionResult,
    SegmentationResult,
    OCRResult,
    ComponentMetadata,
    RelationshipMetadata
)

class DiagramUnderstandingEngine:
    def process_diagram(
        self,
        width: int,
        height: int,
        detections: List[DetectionResult],
        segmentations: List[SegmentationResult],
        ocr_results: List[OCRResult]
    ) -> Tuple[List[ComponentMetadata], List[RelationshipMetadata]]:
        """
        Runs the Diagram Understanding Engine (V2):
        1. Correlates labels (OCR text) to their bounding boxes.
        2. Matches labels to their adjacent connector arrows.
        3. Follows/matches arrows to their targeted physical shapes/organs.
        4. Transmits semantic labels to physical shapes (assigns semantic_name).
        5. Generates the structured relationship graph.
        """
        components: List[ComponentMetadata] = []
        relationships: List[RelationshipMetadata] = []
        
        seg_map = {s.detection_id: s for s in segmentations}
        
        # 1. Initialize components categorized by detection type
        labels: List[ComponentMetadata] = []
        objects: List[ComponentMetadata] = []
        arrows: List[ComponentMetadata] = []
        
        for det in detections:
            seg = seg_map.get(det.id)
            mask_path = seg.mask_path if seg else None
            crop_path = seg.crop_path if seg else None
            
            # Semantic names default to general type name
            if det.category == "label":
                # Find matching OCR result
                ocr_res = self._find_matching_ocr(det.box, ocr_results)
                text_val = ocr_res.text if ocr_res else "Label"
                ocr_conf = ocr_res.confidence if ocr_res else 0.50
                
                det_label = getattr(det, "label", None) or f"{text_val.lower().replace(' ', '_')}_label"
                
                comp = ComponentMetadata(
                    id=det.id,
                    type="text_label",
                    box=det.box,
                    text=text_val,
                    semantic_name=det_label,
                    confidence=float(round(0.6 * det.confidence + 0.4 * ocr_conf, 2)),
                    z_index=10,
                    visible=True
                )
                labels.append(comp)
                
            elif det.category == "arrow":
                det_label = getattr(det, "label", None) or "arrow"
                comp = ComponentMetadata(
                    id=det.id,
                    type="arrow",
                    box=det.box,
                    mask_path=mask_path,
                    crop_path=crop_path,
                    semantic_name=det_label,
                    confidence=det.confidence,
                    z_index=5,
                    visible=True
                )
                arrows.append(comp)
                
            elif det.category == "object":
                det_label = getattr(det, "label", None) or "object"
                comp = ComponentMetadata(
                    id=det.id,
                    type="image_object",
                    box=det.box,
                    mask_path=mask_path,
                    crop_path=crop_path,
                    semantic_name=det_label,
                    confidence=det.confidence,
                    z_index=2,
                    visible=True
                )
                objects.append(comp)
                
        # 2. Match OCR Labels to Arrows
        # Find which arrow starts closest to which text label
        arrow_to_label: Dict[str, str] = {} # maps arrow_id -> label_id
        label_to_arrow: Dict[str, str] = {} # maps label_id -> arrow_id
        
        for arr in arrows:
            nearest_lbl = self._find_nearest_component(arr.box, labels)
            if nearest_lbl:
                dist = self._calculate_box_distance(arr.box, nearest_lbl.box)
                # If arrow starts/is within 150px of a label, associate them
                if dist < 150.0:
                    arrow_to_label[arr.id] = nearest_lbl.id
                    label_to_arrow[nearest_lbl.id] = arr.id
                    arr.associated_label_id = nearest_lbl.id
                    nearest_lbl.associated_object_id = arr.id # Label points to arrow
                    
        # 3. Match Arrows to Segmented Objects
        # Find which physical shape is closest to the arrow but is NOT the label
        for arr in arrows:
            associated_lbl_id = arrow_to_label.get(arr.id)
            candidates = [obj for obj in objects]
            
            nearest_obj = self._find_nearest_component(arr.box, candidates)
            if nearest_obj:
                dist = self._calculate_box_distance(arr.box, nearest_obj.box)
                if dist < 250.0:
                    arr.associated_object_id = nearest_obj.id
                    # Also link the target shape back to the arrow
                    nearest_obj.associated_label_id = arr.id 
                    
        # 4. Direct Label-to-Object Fallback (if no arrow is present/detected)
        for lbl in labels:
            if lbl.id not in label_to_arrow:
                nearest_obj = self._find_nearest_component(lbl.box, objects)
                if nearest_obj:
                    dist = self._calculate_box_distance(lbl.box, nearest_obj.box)
                    if dist < 200.0:
                        lbl.associated_object_id = nearest_obj.id
                        nearest_obj.associated_label_id = lbl.id
                        
        # 5. Assign Semantic Names to Target Objects & Generate Relationships
        # For each label, trace its target shape (either via arrow or direct)
        # Set shape.semantic_name = label.text.lower()
        for lbl in labels:
            target_obj = None
            arrow_id = label_to_arrow.get(lbl.id)
            
            if arrow_id:
                # Find the arrow component
                arr_comp = next((a for a in arrows if a.id == arrow_id), None)
                if arr_comp and arr_comp.associated_object_id:
                    target_obj = next((o for o in objects if o.id == arr_comp.associated_object_id), None)
            else:
                # Direct match
                if lbl.associated_object_id:
                    target_obj = next((o for o in objects if o.id == lbl.associated_object_id), None)
                    
            if target_obj:
                # Clean label text for variable name
                clean_name = lbl.text.lower().strip().replace(" ", "_")
                # Exclude common non-semantic values
                if clean_name not in ["label", "start", "process", "decision", "end"]:
                    lbl.semantic_name = f"{clean_name}_label"
                
                # Add relationship edge
                relationships.append(RelationshipMetadata(
                    label_id=lbl.id,
                    label_text=lbl.text,
                    arrow_id=arrow_id,
                    target_id=target_obj.id
                ))
                
        # Combine all components
        components.extend(objects)
        components.extend(arrows)
        components.extend(labels)
        
        # Ensure all components have a default semantic_name if not assigned
        for c in components:
            if not c.semantic_name:
                c.semantic_name = c.id
                
        # Sort by z-index
        components.sort(key=lambda c: c.z_index)
        
        # Relationship Integrity Validation:
        # Every relationship must reference components that exist inside components[]
        comp_ids = {c.id for c in components}
        valid_relationships = []
        for rel in relationships:
            if rel.label_id not in comp_ids:
                print(f"[Warning] Relationship integrity check failed: label {rel.label_id} not found in components. Skipping.")
                continue
            if rel.arrow_id and rel.arrow_id not in comp_ids:
                print(f"[Warning] Relationship integrity check failed: arrow {rel.arrow_id} not found in components. Skipping.")
                continue
            if rel.target_id not in comp_ids:
                print(f"[Warning] Relationship integrity check failed: target {rel.target_id} not found in components. Skipping.")
                continue
            valid_relationships.append(rel)
        
        relationships = valid_relationships
        return components, relationships

    def _find_matching_ocr(self, box: List[int], ocr_results: List[OCRResult]) -> Optional[OCRResult]:
        best_ocr = None
        best_overlap = -1.0
        for ocr in ocr_results:
            overlap = self._calculate_iou(box, ocr.box)
            if overlap > best_overlap:
                best_overlap = overlap
                best_ocr = ocr
        if best_overlap > 0.1:
            return best_ocr
        if ocr_results:
            closest_ocr = min(ocr_results, key=lambda o: self._calculate_box_distance(box, o.box))
            if self._calculate_box_distance(box, closest_ocr.box) < 40.0:
                return closest_ocr
        return None

    def _find_nearest_component(self, target_box: List[int], candidates: List[ComponentMetadata]) -> Optional[ComponentMetadata]:
        if not candidates:
            return None
        return min(candidates, key=lambda c: self._calculate_box_distance(target_box, c.box))

    def _calculate_box_distance(self, boxA: List[int], boxB: List[int]) -> float:
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
