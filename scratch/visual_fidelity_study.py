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

def classify_component(filename, comp_type, semantic_name):
    filename = filename.lower()
    semantic_name = (semantic_name or "").lower()
    
    # Text labels and arrows/connectors are Category A
    if comp_type == "text_label":
        return "A"
    if comp_type == "arrow":
        return "A"
        
    # Flowchart shapes are Category A
    if "flowchart" in filename:
        return "A"
    if "canva" in filename:
        return "A"
    if "learning_process" in filename or "parts_of_speech" in filename or "study_methods" in filename or "alkali_metals" in filename:
        return "A"
        
    # Hybrid symbols/icons
    if "circuit" in filename or "electrical_induction" in filename:
        if "battery" in semantic_name or "bulb" in semantic_name or "switch" in semantic_name or "coil" in semantic_name:
            return "B"
    if "water_conservation" in filename and "droplet" in semantic_name:
        return "B"
    if "gemini" in filename and "prism" in semantic_name:
        return "B"
    if "pendulum" in filename and "bob" in semantic_name:
        return "B"
        
    # Rich visual objects
    if "biology" in filename or "digestive" in filename or "heart" in filename or "plant" in filename or "animal" in filename:
        return "C"
    if "solar" in filename or "water_cycle" in filename or "photosynthesis" in filename or "nitrogen" in filename:
        return "C"
    if "pulley" in filename or "shaft" in filename or "tank" in filename or "mirror" in filename:
        return "C"
    if "scan" in filename:
        return "C"
        
    return "C" # Default rich object fallback

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
        
        cat_counts = {"A": 0, "B": 0, "C": 0}
        
        for c in components:
            slide_id, c_type, semantic_name = c
            filename = slide_map[slide_id]
            strategy = classify_component(filename, c_type, semantic_name)
            cat_counts[strategy] += 1
            
        print("=== VISUAL FIDELITY EXPORT STRATEGY COUNTS ===")
        print(f"category_a_count (Native PPT Shape): {cat_counts['A']}")
        print(f"category_b_count (Hybrid Symbol): {cat_counts['B']}")
        print(f"category_c_count (Rich Visual Object): {cat_counts['C']}")

if __name__ == "__main__":
    main()
