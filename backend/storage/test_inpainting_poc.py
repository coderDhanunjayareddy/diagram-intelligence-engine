import os
import cv2
import numpy as np
from PIL import Image

def run_grabcut_mask(img, bbox):
    """
    Runs GrabCut inside the bounding box of the target object
    to extract a clean shape mask.
    """
    x, y, w, h = bbox
    crop = img[y:y+h, x:x+w]
    
    mask_gc = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # We use a small border margin for GrabCut background estimation
    border = min(4, max(1, min(w, h) // 10))
    rect = (border, border, w - 2 * border, h - 2 * border)
    
    try:
        cv2.grabCut(crop, mask_gc, rect, bgdModel, fgdModel, 7, cv2.GC_INIT_WITH_RECT)
        mask_crop = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        
        # Also clean up any bright background pixels (white text or paper borders)
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        white_mask = np.where(gray_crop > 245, 0, 255).astype(np.uint8)
        mask_crop = cv2.bitwise_and(mask_crop, white_mask)
        return mask_crop
    except Exception as e:
        print(f"GrabCut failed, using bounding box mask fallback. Error: {e}")
        return np.ones((h, w), dtype=np.uint8) * 255

def build_full_mask(image_shape, bbox, mask_crop):
    """
    Builds a full-sized binary mask (black with white shape).
    """
    h_img, w_img = image_shape[:2]
    full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    
    x, y, w, h = bbox
    # Paste the crop mask
    full_mask[y:y+h, x:x+w] = mask_crop
    return full_mask

def main():
    print("Initializing V3 Inpainting Proof of Concept (POC) with LaMa...")
    
    # Import SimpleLama here to ensure dependencies are loaded
    from simple_lama_inpainting import SimpleLama
    
    # Initialize the model
    print("Loading LaMa model weights (will download if running for the first time)...")
    simple_lama = SimpleLama()
    print("LaMa model loaded successfully.")
    
    output_dir = r"c:\Work\PPT Generation application\backend\storage\poc_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the two test cases
    test_cases = [
        {
            "name": "Standard Digestive System (V2 Validation)",
            "image_path": r"c:\Work\PPT Generation application\test_data\biology_digestive_system.png",
            "liver_bbox": [315, 205, 60, 45] # det_obj_3
        },
        {
            "name": "High-Res 3D Labeled Digestive System (Upload)",
            "image_path": r"C:\Users\DHANUNJAYA SOMIREDDY\Downloads\the-human-digestive-system-labeled.jpg",
            "liver_bbox": [224, 48, 512, 320] # det_obj_2 from detector_provider
        }
    ]
    
    for case in test_cases:
        print(f"\n--- Processing Case: {case['name']} ---")
        img_path = case["image_path"]
        if not os.path.exists(img_path):
            print(f"Error: Target image does not exist at {img_path}")
            continue
            
        # 1. Load image
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        print(f"Loaded image size: {w}x{h}")
        
        # 2. Segment Liver
        bbox = case["liver_bbox"]
        print(f"Extracting Liver mask inside box: {bbox}")
        mask_crop = run_grabcut_mask(img, bbox)
        
        # Save segmented liver crop for reference
        liver_crop = cv2.bitwise_and(img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]], 
                                     cv2.cvtColor(mask_crop, cv2.COLOR_GRAY2BGR))
        cv2.imwrite(os.path.join(output_dir, f"{case['name'].replace(' ', '_').lower()}_liver_cutout.png"), liver_crop)
        
        # 3. Create full-size binary mask for inpainting
        full_mask = build_full_mask(img.shape, bbox, mask_crop)
        cv2.imwrite(os.path.join(output_dir, f"{case['name'].replace(' ', '_').lower()}_inpainting_mask.png"), full_mask)
        
        # 4. Run LaMa Inpainting
        print("Running LaMa Inpainting on CPU... (This may take several seconds)")
        # Convert BGR to RGB for PIL/LaMa
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_mask = Image.fromarray(full_mask).convert("L")
        
        import time
        start_time = time.time()
        result_pil = simple_lama(pil_img, pil_mask)
        elapsed = time.time() - start_time
        print(f"Inpainting complete in {elapsed:.2f} seconds.")
        
        # 5. Save Results
        result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
        result_path = os.path.join(output_dir, f"{case['name'].replace(' ', '_').lower()}_inpainted.png")
        cv2.imwrite(result_path, result_bgr)
        print(f"Saved inpainted result to: {result_path}")
        
        # Create a side-by-side comparison visualization
        # Resize to fit side-by-side easily if high-res
        disp_w = 400
        disp_h = int((disp_w / w) * h)
        
        img_resized = cv2.resize(img, (disp_w, disp_h))
        result_resized = cv2.resize(result_bgr, (disp_w, disp_h))
        
        # Show mask overlay on original image
        mask_resized = cv2.resize(full_mask, (disp_w, disp_h))
        mask_overlay = img_resized.copy()
        mask_overlay[mask_resized > 0] = [0, 0, 255] # Red overlay
        
        comparison = np.hstack((img_resized, mask_overlay, result_resized))
        comparison_path = os.path.join(output_dir, f"{case['name'].replace(' ', '_').lower()}_comparison.png")
        cv2.imwrite(comparison_path, comparison)
        print(f"Saved side-by-side comparison to: {comparison_path}")

if __name__ == "__main__":
    main()
