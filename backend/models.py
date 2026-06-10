from sqlalchemy import Column, String, Integer, Float, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from backend.database import Base

class DbBatchJob(Base):
    __tablename__ = "batch_jobs"
    
    batch_id = Column(String, primary_key=True, index=True)
    status = Column(String, default="queued") # queued, processing, completed, error
    created_at = Column(Float, nullable=False)
    completed_at = Column(Float, nullable=True)
    error_message = Column(String, nullable=True)
    pptx_path = Column(String, nullable=True)
    
    slides = relationship("DbSlideMetadata", back_populates="batch", cascade="all, delete-orphan")

class DbSlideMetadata(Base):
    __tablename__ = "slides_metadata"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String, ForeignKey("batch_jobs.batch_id"), nullable=False)
    slide_index = Column(Integer, nullable=False)
    original_filename = Column(String, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    routing_status = Column(String, default="Manual Review") # Auto Export, Warning, Manual Review
    average_confidence = Column(Float, default=0.0)
    file_type = Column(String, default="PNG")
    content_type = Column(String, default="EducationalDiagram")
    metrics_json = Column(JSON, nullable=True) # stores execution times per stage
    
    batch = relationship("DbBatchJob", back_populates="slides")
    components = relationship("DbComponentMetadata", back_populates="slide", cascade="all, delete-orphan")
    relationships = relationship("DbRelationship", back_populates="slide", cascade="all, delete-orphan")

class DbComponentMetadata(Base):
    __tablename__ = "components_metadata"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    slide_id = Column(Integer, ForeignKey("slides_metadata.id"), nullable=False)
    component_id = Column(String, nullable=False)
    type = Column(String, nullable=False) # image_object, text_label, arrow, shape
    semantic_name = Column(String, nullable=True)
    box_x = Column(Integer, nullable=False)
    box_y = Column(Integer, nullable=False)
    box_w = Column(Integer, nullable=False)
    box_h = Column(Integer, nullable=False)
    mask_path = Column(String, nullable=True)
    crop_path = Column(String, nullable=True)
    text = Column(String, nullable=True)
    confidence = Column(Float, default=1.0)
    visible = Column(Boolean, default=True)
    z_index = Column(Integer, default=0)
    associated_label_id = Column(String, nullable=True)
    associated_object_id = Column(String, nullable=True)
    
    slide = relationship("DbSlideMetadata", back_populates="components")

class DbRelationship(Base):
    __tablename__ = "relationships"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    slide_id = Column(Integer, ForeignKey("slides_metadata.id"), nullable=False)
    label_id = Column(String, nullable=False)
    label_text = Column(String, nullable=False)
    arrow_id = Column(String, nullable=True)
    target_id = Column(String, nullable=False)
    
    slide = relationship("DbSlideMetadata", back_populates="relationships")
