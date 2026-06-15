from backend.database import SessionLocal
from backend.models import DbSlideMetadata, DbComponentMetadata

db = SessionLocal()
try:
    print("Fetching Slide 0 components...")
    slide = db.query(DbSlideMetadata).filter(DbSlideMetadata.batch_id == "proof_batch", DbSlideMetadata.slide_index == 0).first()
    if slide:
        print(f"Slide: {slide.original_filename}")
        print("Components in Database:")
        for comp in slide.components:
            if comp.is_occluded or comp.amodal_mask_path:
                print(f" - ID: {comp.component_id}, Semantic Name: {comp.semantic_name}, Box: [{comp.box_x}, {comp.box_y}, {comp.box_w}, {comp.box_h}], Is Occluded: {comp.is_occluded}, Amodal Mask: {comp.amodal_mask_path}")
            elif comp.semantic_name in ["esophagus", "liver", "stomach"]:
                print(f" - ID: {comp.component_id}, Semantic Name: {comp.semantic_name}, Box: [{comp.box_x}, {comp.box_y}, {comp.box_w}, {comp.box_h}], Is Occluded: {comp.is_occluded}")
    else:
        print("Slide not found in DB.")
finally:
    db.close()
