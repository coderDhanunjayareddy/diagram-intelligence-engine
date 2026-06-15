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
            text(f"SELECT slide_id, type FROM components_metadata WHERE slide_id IN ({ids_str})")
        ).fetchall()
        
        slide_to_cat = {s[0]: CATEGORY_MAP.get(s[1], "Natural") for s in slides}
        
        counts = {
            "Natural": {"text_label": 0, "shape": 0, "arrow": 0, "image_object": 0},
            "Structured": {"text_label": 0, "shape": 0, "arrow": 0, "image_object": 0},
            "Infographic": {"text_label": 0, "shape": 0, "arrow": 0, "image_object": 0}
        }
        
        for c in components:
            slide_id, c_type = c
            cat = slide_to_cat[slide_id]
            # map db type to standard types
            if c_type in counts[cat]:
                counts[cat][c_type] += 1
            else:
                # Group unknown into image_object
                counts[cat]["image_object"] += 1
                
        print("=== COMPONENT TYPE COUNTS BY CATEGORY ===")
        for cat, types in counts.items():
            print(f"{cat}:")
            for t, val in types.items():
                print(f"  {t}: {val}")
                
if __name__ == "__main__":
    main()
