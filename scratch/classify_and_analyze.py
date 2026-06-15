import os
from sqlalchemy import create_engine, text

DATABASE_URL = "sqlite:///c:/Work/PPT Generation application/backend/storage/diagram_v2.db"
engine = create_engine(DATABASE_URL)

# Category mappings for the 35 unique filenames
CATEGORY_MAP = {
    # 1. Natural Diagrams
    "biology_animal_cell.png": "Natural",
    "biology_digestive_system.png": "Natural",
    "biology_human_heart.png": "Natural",
    "biology_plant_cell.png": "Natural",
    "difficult_mixed_content_1.png": "Natural",
    "difficult_textbook_scan_1.png": "Natural",
    "difficult_textbook_scan_2.png": "Natural",
    "difficult_textbook_scan_3.png": "Natural",
    "difficult_textbook_scan_4.png": "Natural",
    "electrical_induction.png": "Natural",
    "engineering_shaft.png": "Natural",
    "gemini_style_schematic.png": "Natural",
    "geography_island.png": "Natural",
    "gpt_style_diagram.png": "Natural",
    "industrial_tank.png": "Natural",
    "mechanical_pulley.png": "Natural",
    "physics_electric_circuit.png": "Natural",
    "physics_pendulum_system.png": "Natural",
    "physics_reflection_of_light.png": "Natural",
    "physics_solar_system.png": "Natural",
    "physics_water_cycle.png": "Natural",
    
    # 2. Structured Diagrams
    "canva_style_slide.png": "Structured",
    "difficult_mixed_content_2.png": "Structured",
    "difficult_mixed_content_3.png": "Structured",
    "flowchart_order_processing.png": "Structured",
    "flowchart_software_development_lifecycle.png": "Structured",
    "flowchart_student_admission_process.png": "Structured",
    "flowchart_temperature_control_loop.png": "Structured",
    
    # 3. Infographics
    "infographic_chemistry_alkali_metals.png": "Infographic",
    "infographic_learning_process.png": "Infographic",
    "infographic_nitrogen_cycle.png": "Infographic",
    "infographic_parts_of_speech.png": "Infographic",
    "infographic_photosynthesis.png": "Infographic",
    "infographic_study_methods.png": "Infographic",
    "infographic_water_conservation.png": "Infographic"
}

def main():
    batch_id = "v4_batch_50"
    
    with engine.connect() as conn:
        slides = conn.execute(
            text("SELECT id, slide_index, original_filename FROM slides_metadata WHERE batch_id = :batch_id"),
            {"batch_id": batch_id}
        ).fetchall()
        
        slide_ids = [s[0] for s in slides]
        if not slide_ids:
            print("No slides found.")
            return
            
        ids_str = ", ".join(str(i) for i in slide_ids)
        components = conn.execute(
            text(f"SELECT slide_id, type FROM components_metadata WHERE slide_id IN ({ids_str})")
        ).fetchall()
        
        # Group components by slide
        slide_comps = {}
        for c in components:
            s_id, c_type = c
            if s_id not in slide_comps:
                slide_comps[s_id] = []
            slide_comps[s_id].append(c_type)
            
        category_counts = {"Natural": 0, "Structured": 0, "Infographic": 0}
        category_data = {
            "Natural": {"total_elements": 0, "total_ocr": 0, "total_editability": 0.0, "slides_count": 0},
            "Structured": {"total_elements": 0, "total_ocr": 0, "total_editability": 0.0, "slides_count": 0},
            "Infographic": {"total_elements": 0, "total_ocr": 0, "total_editability": 0.0, "slides_count": 0}
        }
        
        for s in slides:
            s_id, s_idx, filename = s
            cat = CATEGORY_MAP.get(filename, "Natural") # default fallback
            category_counts[cat] += 1
            
            s_c_types = slide_comps.get(s_id, [])
            total_elements = len(s_c_types)
            ocr_count = sum(1 for t in s_c_types if t == "text_label")
            editability = float(ocr_count) / max(1, total_elements)
            
            category_data[cat]["total_elements"] += total_elements
            category_data[cat]["total_ocr"] += ocr_count
            category_data[cat]["total_editability"] += editability
            category_data[cat]["slides_count"] += 1
            
        print("=== CLASSIFICATION COUNTS ===")
        print(f"Natural diagrams: {category_counts['Natural']}")
        print(f"Structured diagrams: {category_counts['Structured']}")
        print(f"Infographics: {category_counts['Infographic']}")
        
        print("\n=== DETAILED STATS BY CATEGORY ===")
        for cat, data in category_data.items():
            count = data["slides_count"]
            if count > 0:
                avg_elem = data["total_elements"] / count
                avg_ocr = data["total_ocr"] / count
                avg_edit = data["total_editability"] / count
                print(f"{cat}:")
                print(f"  Average Elements: {avg_elem:.2f}")
                print(f"  Average OCR (Text): {avg_ocr:.2f}")
                print(f"  Average Editability Score: {avg_edit * 100:.1f}%")
            else:
                print(f"{cat}: No slides")

if __name__ == "__main__":
    main()
