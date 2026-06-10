from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# Core Datatypes

class DetectionResult(BaseModel):
    id: str = Field(description="Unique ID for the detection")
    category: str = Field(description="Category of object: 'object', 'label', 'arrow', 'shape', 'background'")
    box: List[int] = Field(description="Bounding box [x, y, width, height] in pixel coordinates")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    label: Optional[str] = Field(None, description="Detection label matched (e.g. 'ball', 'earth')")

class SegmentationResult(BaseModel):
    detection_id: str = Field(description="ID of the parent detection")
    mask_path: str = Field(description="Relative path to transparent PNG mask/layer file")
    crop_path: str = Field(description="Relative path to cropped bounding box image")
    box: List[int] = Field(description="Bounding box [x, y, width, height]")
    confidence: float = Field(description="Segmentation confidence score between 0.0 and 1.0")

class OCRResult(BaseModel):
    text: str = Field(description="Extracted text label")
    box: List[int] = Field(description="Bounding box [x, y, width, height] in pixel coordinates")
    confidence: float = Field(description="OCR confidence score between 0.0 and 1.0")

class ComponentMetadata(BaseModel):
    id: str = Field(description="Unique ID of the component")
    type: str = Field(description="Type: 'image_object', 'text_label', 'arrow', 'shape'")
    semantic_name: Optional[str] = Field(None, description="Semantic name (e.g. 'stomach', 'liver')")
    box: List[int] = Field(description="Bounding box [x, y, width, height] in pixel coordinates")
    mask_path: Optional[str] = Field(None, description="Path to transparent PNG layer (for image_objects and shapes)")
    crop_path: Optional[str] = Field(None, description="Path to cropped layer bounding box image")
    text: Optional[str] = Field(None, description="Editable text value (for text_labels)")
    confidence: float = Field(description="Confidence rating of the extracted component")
    visible: bool = Field(True, description="Whether the component is visible/active")
    z_index: int = Field(0, description="Z-index layering order")
    associated_label_id: Optional[str] = Field(None, description="ID of text label associated with this object")
    associated_object_id: Optional[str] = Field(None, description="ID of object this label or arrow points to")

class RelationshipMetadata(BaseModel):
    label_id: str = Field(description="ID of the text label")
    label_text: str = Field(description="Text content of the label")
    arrow_id: Optional[str] = Field(None, description="ID of the arrow connecting them")
    target_id: str = Field(description="ID of the target physical object")

class SlideMetadata(BaseModel):
    slide_index: int = Field(description="0-indexed slide index in batch")
    original_filename: str = Field(description="Name of the original uploaded file")
    width: int = Field(description="Original image width in pixels")
    height: int = Field(description="Original image height in pixels")
    components: List[ComponentMetadata] = Field(default=[], description="Extracted components on this slide")
    routing_status: str = Field("Manual Review", description="Workflow routing: 'Auto Export', 'Warning', 'Manual Review'")
    average_confidence: float = Field(0.0, description="Average confidence across all detections/OCR/segmentation")
    file_type: str = Field("PNG", description="Detected file type")
    content_type: str = Field("EducationalDiagram", description="Detected diagram type")
    relationships: List[RelationshipMetadata] = Field(default=[], description="Structured relationships between label, arrows, and target objects")

class BatchJob(BaseModel):
    batch_id: str = Field(description="Unique batch task ID")
    status: str = Field("queued", description="Status: 'queued', 'processing', 'completed', 'error'")
    slides: List[SlideMetadata] = Field(default=[], description="Metadata for each slide in batch")
    created_at: float = Field(description="Timestamp of batch creation")
    completed_at: Optional[float] = Field(None, description="Timestamp of batch completion")
    error_message: Optional[str] = Field(None, description="Error details if status is 'error'")
    pptx_path: Optional[str] = Field(None, description="Path to generated PPTX file")


# Pluggable Service Interfaces

class IClassifier(ABC):
    @abstractmethod
    def classify(self, file_path: str) -> Dict[str, Any]:
        """
        Determines file type and diagram content type.
        Returns dict containing: 'fileType', 'contentType', 'confidence'
        """
        pass

class IDetector(ABC):
    @abstractmethod
    def detect(self, image_path: str) -> List[DetectionResult]:
        """
        Detects bounding boxes of components (labels, arrows, shapes, main objects).
        """
        pass

class ISegmenter(ABC):
    @abstractmethod
    def segment(self, image_path: str, detections: List[DetectionResult], task_dir: str) -> List[SegmentationResult]:
        """
        Extracts individual transparent PNG masks/layers for detected bounding boxes and writes files to task_dir.
        """
        pass

class IOCR(ABC):
    @abstractmethod
    def extract_text(self, image_path: str, detections: List[DetectionResult]) -> List[OCRResult]:
        """
        Extracts text and positions inside detected text labels (or general text search).
        """
        pass

class IReconstructor(ABC):
    @abstractmethod
    def reconstruct(
        self,
        width: int,
        height: int,
        detections: List[DetectionResult],
        segmentations: List[SegmentationResult],
        ocr_results: List[OCRResult]
    ) -> List[ComponentMetadata]:
        """
        Groups detected shapes, masks, and OCR texts into logical component structures with links and layering.
        """
        pass
