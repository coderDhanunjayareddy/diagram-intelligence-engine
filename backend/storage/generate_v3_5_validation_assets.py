import os
import cv2
import json
import shutil
import numpy as np
from backend.database import SessionLocal
from backend.models import DbSlideMetadata

def draw_checkerboard(width, height, box_size=10):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(0, height, box_size):
        for x in range(0, width, box_size):
            color = 255 if ((x // box_size) + (y // box_size)) % 2 == 0 else 220
            img[y:y+box_size, x:x+box_size] = color
    return img

def render_polygon_overlay(orig_img_path, polygon_vertices, output_path):
    img = cv2.imread(orig_img_path)
    overlay = img.copy()
    pts = np.array(polygon_vertices, dtype=np.int32)
    # Draw filled red polygon on overlay
    cv2.fillPoly(overlay, [pts], (0, 0, 255))
    # Blend overlay with original
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    # Draw bright red border line
    cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=4)
    cv2.imwrite(output_path, img)

def generate_assets():
    artifact_dir = r"C:\Users\DHANUNJAYA SOMIREDDY\.gemini\antigravity\brain\526bf6ed-0c27-4132-89b3-98573c19bac3"
    batches_dir = r"c:\Work\PPT Generation application\backend\storage\batches\proof_batch\slides"
    
    db = SessionLocal()
    try:
        # 1. Fetch Slides from the latest run (Slide 54 - Digestive, Slide 60 - Scan)
        slide_0 = db.query(DbSlideMetadata).filter(DbSlideMetadata.id == 54).first()
        slide_6 = db.query(DbSlideMetadata).filter(DbSlideMetadata.id == 60).first()
        
        slides = [
            {"meta": slide_0, "folder": "slide_0", "prefix": "biology_digestive_system", "shift_comp_name": "liver", "shift_val": 120},
            {"meta": slide_6, "folder": "slide_6", "prefix": "difficult_textbook_scan_1", "shift_comp_name": "protein_channel", "shift_val": 120}
        ]
        
        for item in slides:
            slide = item["meta"]
            folder_name = item["folder"]
            prefix = item["prefix"]
            shift_comp_name = item["shift_comp_name"]
            shift_val = item["shift_val"]
            
            slide_dir = os.path.join(batches_dir, folder_name)
            if not slide:
                print(f"Slide metadata not found for {folder_name}")
                continue
                
            print(f"Generating validation assets for {slide.original_filename}...")
            
            # Find the occluded/reconstructed component
            reconstructed_comp = next((c for c in slide.components if c.is_occluded), None)
            if not reconstructed_comp:
                print(f"No occluded component found for {slide.original_filename}")
                continue
                
            semantic_name = reconstructed_comp.semantic_name
            print(f"Found reconstructed component: {reconstructed_comp.component_id} ({semantic_name})")
            
            # Paths on disk
            vis_mask_path = os.path.join(slide_dir, reconstructed_comp.mask_path.replace("/", os.sep))
            amodal_mask_path = os.path.join(slide_dir, reconstructed_comp.amodal_mask_path.replace("/", os.sep))
            orig_img_path = os.path.join(slide_dir, "original", slide.original_filename)
            bg_path = os.path.join(slide_dir, "masks", "background.png")
            
            # Destination filenames
            vis_dest = os.path.join(artifact_dir, f"{prefix}_{semantic_name}_visible_mask.png")
            poly_dest = os.path.join(artifact_dir, f"{prefix}_{semantic_name}_predicted_polygon.png")
            amodal_dest = os.path.join(artifact_dir, f"{prefix}_{semantic_name}_amodal_mask.png")
            checker_dest = os.path.join(artifact_dir, f"{prefix}_{semantic_name}_amodal_checkerboard.png")
            ppt_before = os.path.join(artifact_dir, f"{prefix}_{semantic_name}_ppt_before.png")
            ppt_after = os.path.join(artifact_dir, f"{prefix}_{semantic_name}_ppt_after.png")
            
            # Output 1: Copy Visible Mask
            if os.path.exists(vis_mask_path):
                shutil.copy(vis_mask_path, vis_dest)
                print(f" -> Copied visible mask to {vis_dest}")
                
            # Output 2: Generate Predicted Polygon Overlay
            polygon_vertices = json.loads(reconstructed_comp.polygon_vertices_json)
            render_polygon_overlay(orig_img_path, polygon_vertices, poly_dest)
            print(f" -> Rendered polygon overlay to {poly_dest}")
            
            # Output 3: Copy Amodal Mask & Render on Checkerboard
            if os.path.exists(amodal_mask_path):
                shutil.copy(amodal_mask_path, amodal_dest)
                print(f" -> Copied amodal mask to {amodal_dest}")
                
                # Render amodal mask on checkerboard
                amodal_rgba = cv2.imread(amodal_mask_path, cv2.IMREAD_UNCHANGED)
                h_am, w_am = amodal_rgba.shape[:2]
                checker = draw_checkerboard(w_am, h_am, box_size=8)
                alpha = amodal_rgba[:, :, 3] / 255.0
                alpha = np.expand_dims(alpha, axis=2)
                checker_composite = (amodal_rgba[:, :, :3] * alpha + checker * (1.0 - alpha)).astype(np.uint8)
                cv2.imwrite(checker_dest, checker_composite)
                print(f" -> Rendered checkerboard composite to {checker_dest}")
                
            # Output 4: Render PPT Composites (Before & After reveal)
            bg_img = cv2.imread(bg_path)
            h_bg, w_bg = bg_img.shape[:2]
            
            # Query components in Z-index order
            components = []
            db_comps = sorted(slide.components, key=lambda c: c.z_index)
            for c in db_comps:
                if c.type in ["image_object", "shape"]:
                    if c.amodal_mask_path:
                        c_path = os.path.join(slide_dir, c.amodal_mask_path.replace("/", os.sep))
                    else:
                        c_path = os.path.join(slide_dir, c.mask_path.replace("/", os.sep)) if c.mask_path else ""
                        
                    if c_path and os.path.exists(c_path):
                        components.append({
                            "name": c.semantic_name,
                            "box": [c.box_x, c.box_y, c.box_w, c.box_h],
                            "path": c_path
                        })
                        
            def render_layout(shift_x=0):
                canvas = bg_img.copy()
                for comp in components:
                    mask_rgba = cv2.imread(comp["path"], cv2.IMREAD_UNCHANGED)
                    x, y, w, h = comp["box"]
                    
                    if comp["name"] == shift_comp_name:
                        x += shift_x
                        
                    # Resize mask to match database box size
                    if mask_rgba.shape[1] != w or mask_rgba.shape[0] != h:
                        mask_rgba = cv2.resize(mask_rgba, (w, h), interpolation=cv2.INTER_AREA)
                        
                    x_start = max(0, min(x, w_bg - 1))
                    y_start = max(0, min(y, h_bg - 1))
                    x_end = max(1, min(x + w, w_bg))
                    y_end = max(1, min(y + h, h_bg))
                    
                    cw = x_end - x_start
                    ch = y_end - y_start
                    
                    if cw <= 0 or ch <= 0:
                        continue
                        
                    mask_y_start = y_start - y
                    mask_x_start = x_start - x
                    mask_y_end = mask_y_start + ch
                    mask_x_end = mask_x_start + cw
                    
                    crop_rgba = mask_rgba[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
                    alpha = crop_rgba[:, :, 3] / 255.0
                    alpha = np.expand_dims(alpha, axis=2)
                    
                    canvas[y_start:y_end, x_start:x_end] = (
                        crop_rgba[:, :, :3] * alpha + canvas[y_start:y_end, x_start:x_end] * (1.0 - alpha)
                    ).astype(np.uint8)
                return canvas
                
            # Render Before Reveal (0 shift)
            before_img = render_layout(shift_x=0)
            cv2.imwrite(ppt_before, before_img)
            print(f" -> Rendered PPT before reveal to {ppt_before}")
            
            # Render After Reveal
            after_img = render_layout(shift_x=shift_val)
            cv2.imwrite(ppt_after, after_img)
            print(f" -> Rendered PPT after reveal to {ppt_after}")
            
    finally:
        db.close()
        
if __name__ == "__main__":
    generate_assets()
