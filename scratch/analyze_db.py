import os
import sys
import json
from sqlalchemy import create_engine, text

DATABASE_URL = "sqlite:///c:/Work/PPT Generation application/backend/storage/diagram_v2.db"
engine = create_engine(DATABASE_URL)

def main():
    batch_id = "v4_batch_50"
    
    with engine.connect() as conn:
        # Get slide IDs
        slides = conn.execute(
            text("SELECT id, slide_index, original_filename FROM slides_metadata WHERE batch_id = :batch_id"),
            {"batch_id": batch_id}
        ).fetchall()
        
        slide_ids = [s[0] for s in slides]
        
        if not slide_ids:
            print("No slides found for batch:", batch_id)
            return
            
        print(f"Analyzing {len(slide_ids)} slides for batch {batch_id}...")
        
        # Get components using a Python-formatted IN clause for safety in raw SQL execution
        ids_str = ", ".join(str(i) for i in slide_ids)
        query = f"SELECT slide_id, type, visible, is_occluded, amodal_mask_path, reconstruction_source, confidence FROM components_metadata WHERE slide_id IN ({ids_str})"
        
        components = conn.execute(text(query)).fetchall()
        
        # Calculations
        total_detected = len(components)
        exported_count = 0
        left_in_bg = 0
        editable_count = 0
        
        required_amodal = 0
        source_vision_llm = 0
        source_fallback = 0
        
        for c in components:
            slide_id, c_type, visible, is_occluded, amodal_mask, recon_source, confidence = c
            
            # Independent PPT objects are those with visible=True
            if visible:
                exported_count += 1
            else:
                left_in_bg += 1
                
            if c_type == "text_label":
                editable_count += 1
                
            if is_occluded:
                required_amodal += 1
                if recon_source == "vision_llm":
                    source_vision_llm += 1
                elif recon_source == "simulated_fallback":
                    source_fallback += 1
                    
        # Slide-specific editability rankings
        slide_editability = {}
        for s in slides:
            s_id, s_idx, filename = s
            s_comps = [c for c in components if c[0] == s_id]
            total_s = len(s_comps)
            text_s = sum(1 for c in s_comps if c[1] == "text_label")
            score = float(text_s) / max(1, total_s)
            slide_editability[s_idx] = {
                "filename": filename,
                "score": score,
                "total": total_s,
                "text": text_s
            }
            
        sorted_slides = sorted(slide_editability.items(), key=lambda item: item[1]["score"])
        
        print("\n=== SUMMARY METRICS ===")
        print("Total detected:", total_detected)
        print("Exported as independent PPT:", exported_count)
        print("Left in bg:", left_in_bg)
        print("Editable elements (text_labels):", editable_count)
        print("Editability score:", float(editable_count) / max(1, total_detected))
        print("Required amodal:", required_amodal)
        print("Source vision_llm:", source_vision_llm)
        print("Source simulated_fallback:", source_fallback)
        
        print("\n=== WORST 5 SLIDES ===")
        for item in sorted_slides[:5]:
            idx, data = item
            print(f"Slide {idx} ({data['filename']}): {data['score'] * 100:.1f}% (Text: {data['text']}/{data['total']})")
            
        print("\n=== BEST 5 SLIDES ===")
        for item in sorted_slides[-5:]:
            idx, data = item
            print(f"Slide {idx} ({data['filename']}): {data['score'] * 100:.1f}% (Text: {data['text']}/{data['total']})")

if __name__ == "__main__":
    main()
