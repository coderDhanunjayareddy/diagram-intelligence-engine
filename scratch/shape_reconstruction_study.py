import os
from sqlalchemy import create_engine, text

DATABASE_URL = "sqlite:///c:/Work/PPT Generation application/backend/storage/diagram_v2.db"
engine = create_engine(DATABASE_URL)

CATEGORY_MAP = {
    "biology_animal_cell.png": "Natural", "biology_digestive_system.png": "Natural",
    "biology_human_heart.png": "Natural", "biology_plant_cell.png": "Natural",
    "difficult_mixed_content_1.png": "Natural", "difficult_textbook_scan_1.png": "Natural",
    "difficult_textbook_scan_2.png": "Natural", "difficult_textbook_scan_3.png": "Natural",
    "difficult_textbook_scan_4.png": "Natural", "electrical_induction.png": "Natural",
    "engineering_shaft.png": "Natural", "gemini_style_schematic.png": "Natural",
    "geography_island.png": "Natural", "gpt_style_diagram.png": "Natural",
    "industrial_tank.png": "Natural", "mechanical_pulley.png": "Natural",
    "physics_electric_circuit.png": "Natural", "physics_pendulum_system.png": "Natural",
    "physics_reflection_of_light.png": "Natural", "physics_solar_system.png": "Natural",
    "physics_water_cycle.png": "Natural",
    "canva_style_slide.png": "Structured", "difficult_mixed_content_2.png": "Structured",
    "difficult_mixed_content_3.png": "Structured", "flowchart_order_processing.png": "Structured",
    "flowchart_software_development_lifecycle.png": "Structured",
    "flowchart_student_admission_process.png": "Structured", "flowchart_temperature_control_loop.png": "Structured",
    "infographic_chemistry_alkali_metals.png": "Infographic", "infographic_learning_process.png": "Infographic",
    "infographic_nitrogen_cycle.png": "Infographic", "infographic_parts_of_speech.png": "Infographic",
    "infographic_photosynthesis.png": "Infographic", "infographic_study_methods.png": "Infographic",
    "infographic_water_conservation.png": "Infographic"
}

def classify_category_a(filename, comp_type, semantic_name):
    filename = filename.lower()
    semantic_name = (semantic_name or "").lower()
    
    # Non-Category A components are skipped
    is_a = False
    if comp_type in ["text_label", "arrow"]:
        is_a = True
    elif "flowchart" in filename or "canva" in filename:
        is_a = True
    elif any(x in filename for x in ["learning_process", "parts_of_speech", "study_methods", "alkali_metals"]):
        is_a = True
        
    if not is_a:
        return None
        
    # Standard text labels are A1
    if comp_type == "text_label":
        return "A1"
        
    # Orthogonal return loop paths in flowcharts or complex timeline connections are A2
    if "flowchart" in filename and ("loop" in semantic_name or "no" in semantic_name or "yes" in semantic_name):
        return "A2"
    if "timeline" in filename or "roadmap" in semantic_name:
        return "A2"
        
    # Curved or wavy arrows are A3
    if "cycle" in filename and "arrow" in semantic_name:
        return "A3"
    if "photosynthesis" in filename and "arrow" in semantic_name:
        return "A3"
        
    # Standard lines, straight arrows, and simple rectangles/circles/diamonds are A1
    return "A1"

def main():
    batch_id = "v4_batch_50"
    
    with engine.connect() as conn:
        slides = conn.execute(
            text("SELECT id, original_filename FROM slides_metadata WHERE batch_id = :batch_id"),
            {"batch_id": batch_id}
        ).fetchall()
        
        slide_ids = [s[0] for s in slides]
        ids_str = ", ".join(str(i) for i in slide_ids)
        
        components = conn.execute(
            text(f"SELECT slide_id, type, semantic_name FROM components_metadata WHERE slide_id IN ({ids_str})")
        ).fetchall()
        
        slide_map = {s[0]: s[1] for s in slides}
        
        counts = {"A1": 0, "A2": 0, "A3": 0}
        
        for c in components:
            slide_id, c_type, semantic_name = c
            filename = slide_map[slide_id]
            subcat = classify_category_a(filename, c_type, semantic_name)
            if subcat:
                counts[subcat] += 1
                
        print("=== SHAPE RECONSTRUCTION STUDY COUNTS ===")
        print(f"A1_count (Exact PPT Primitive): {counts['A1']}")
        print(f"A2_count (Compound Shape): {counts['A2']}")
        print(f"A3_count (Freeform Vector Shape): {counts['A3']}")

if __name__ == "__main__":
    main()
