import os
import cv2
import numpy as np
import shutil
from PIL import Image
from backend.database import SessionLocal
from backend.models import DbSlideMetadata

def draw_checkerboard(width, height, box_size=10):
    """Generates a gray and white checkerboard pattern."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(0, height, box_size):
        for x in range(0, width, box_size):
            color = 255 if ((x // box_size) + (y // box_size)) % 2 == 0 else 220
            img[y:y+box_size, x:x+box_size] = color
    return img

def main():
    print("Starting visual verification data extraction...")
    
    artifact_dir = r"C:\Users\DHANUNJAYA SOMIREDDY\.gemini\antigravity\brain\526bf6ed-0c27-4132-89b3-98573c19bac3"
    slide_dir = r"c:\Work\PPT Generation application\backend\storage\batches\proof_batch\slides\slide_0"
    
    # 1. Copy required files to artifacts directory
    shutil.copy(os.path.join(slide_dir, "original", "biology_digestive_system.png"), 
                os.path.join(artifact_dir, "digestive_original.png"))
    shutil.copy(os.path.join(slide_dir, "masks", "artifacts", "mask_det_obj_2.png"), 
                os.path.join(artifact_dir, "esophagus_visible_mask.png"))
    shutil.copy(os.path.join(slide_dir, "masks", "amodal_mask_det_obj_2.png"), 
                os.path.join(artifact_dir, "esophagus_amodal_mask.png"))
    shutil.copy(os.path.join(slide_dir, "masks", "background.png"), 
                os.path.join(artifact_dir, "inpainted_background.png"))
                
    print("[1/3] Base files copied to artifacts directory.")
    
    # 2. Render checkerboard esophagus amodal mask
    amodal_mask_path = os.path.join(slide_dir, "masks", "amodal_mask_det_obj_2.png")
    amodal_rgba = cv2.imread(amodal_mask_path, cv2.IMREAD_UNCHANGED)
    h_amodal, w_amodal = amodal_rgba.shape[:2]
    
    checkerboard = draw_checkerboard(w_amodal, h_amodal, box_size=8)
    alpha = amodal_rgba[:, :, 3] / 255.0
    alpha = np.expand_dims(alpha, axis=2)
    
    checker_composite = (amodal_rgba[:, :, :3] * alpha + checkerboard * (1.0 - alpha)).astype(np.uint8)
    cv2.imwrite(os.path.join(artifact_dir, "esophagus_amodal_checkerboard.png"), checker_composite)
    print("[2/3] Checkerboard visualization saved.")
    
    # 3. PPT Simulation Composition
    # Load background
    bg_img = cv2.imread(os.path.join(slide_dir, "masks", "background.png"))
    h_bg, w_bg = bg_img.shape[:2]
    
    # Query database for actual slide 0 components
    db = SessionLocal()
    try:
        slide = db.query(DbSlideMetadata).filter(
            DbSlideMetadata.batch_id == "proof_batch",
            DbSlideMetadata.slide_index == 0
        ).order_by(DbSlideMetadata.id.desc()).first()
        
        if not slide:
            print("Error: Slide 0 not found in database.")
            return
            
        print(f"Loaded slide {slide.id} from database: {slide.original_filename}")
        
        components = []
        db_comps = sorted(slide.components, key=lambda c: c.z_index)
        for c in db_comps:
            # Render image objects (shapes)
            if c.type in ["image_object", "shape"]:
                # Determine paths
                if c.amodal_mask_path:
                    path = os.path.join(slide_dir, c.amodal_mask_path.replace("/", os.sep))
                else:
                    path = os.path.join(slide_dir, c.mask_path.replace("/", os.sep)) if c.mask_path else ""
                
                if path and os.path.exists(path):
                    components.append({
                        "id": c.component_id,
                        "name": c.semantic_name,
                        "box": [c.box_x, c.box_y, c.box_w, c.box_h],
                        "path": path
                    })
                    print(f"Component: {c.component_id} ({c.semantic_name}), Box: {[c.box_x, c.box_y, c.box_w, c.box_h]}, Path: {path}")
    finally:
        db.close()
        
    def render_ppt_layout(liver_shift_x=0):
        canvas = bg_img.copy()
        for comp in components:
            mask_path = comp["path"]
            mask_rgba = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            x, y, w, h = comp["box"]
            
            # Apply shift for liver
            if comp["name"] == "liver":
                x += liver_shift_x
                
            # Resize mask crop to match the database box dimensions
            if mask_rgba.shape[1] != w or mask_rgba.shape[0] != h:
                mask_rgba = cv2.resize(mask_rgba, (w, h), interpolation=cv2.INTER_AREA)
                
            # Constrain to canvas boundaries
            x_start = max(0, min(x, w_bg - 1))
            y_start = max(0, min(y, h_bg - 1))
            x_end = max(1, min(x + w, w_bg))
            y_end = max(1, min(y + h, h_bg))
            
            cw = x_end - x_start
            ch = y_end - y_start
            
            if cw <= 0 or ch <= 0:
                continue
                
            # Compute sub-crop coordinates relative to the resized mask
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
        
    # Render layout 1: Liver in original position
    original_layout = render_ppt_layout(liver_shift_x=0)
    cv2.imwrite(os.path.join(artifact_dir, "ppt_original_layout.png"), original_layout)
    
    # Render layout 2: Liver moved (shifted X by +120 pixels)
    moved_layout = render_ppt_layout(liver_shift_x=120)
    cv2.imwrite(os.path.join(artifact_dir, "ppt_moved_layout.png"), moved_layout)
    
    print("[3/3] PowerPoint layout simulations rendered successfully.")

if __name__ == "__main__":
    main()
