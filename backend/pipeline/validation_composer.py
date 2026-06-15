import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Tuple
from backend.pipeline.interfaces import ComponentMetadata
from backend.pipeline.shape_classifier import ShapeGeometryClassifier, TextStyleExtractor

class ValidationComposer:
    def __init__(self):
        self.shape_classifier = ShapeGeometryClassifier()
        self.text_extractor = TextStyleExtractor()
        
    def compose_validation_screenshots(
        self,
        original_img_path: str,
        erased_bg_path: str,
        components: List[ComponentMetadata],
        slide_dir: str,
        output_dir: str,
        slide_index: int,
        original_filename: str
    ) -> Dict[str, str]:
        """
        Generates and saves visual verification composites:
        - ppt_before: Original composite (clean template + shapes + text)
        - ppt_after: Composite with major objects shifted by 100px (reveal occlusions)
        - Both have a PowerPoint-style Selection Pane sidebar rendered on the right.
        """
        # 1. Load background
        bg_bgr = cv2.imread(erased_bg_path)
        if bg_bgr is None:
            bg_bgr = cv2.imread(original_img_path)
        if bg_bgr is None:
            # Create blank backup canvas
            bg_bgr = np.ones((600, 800, 3), dtype=np.uint8) * 248
            
        h_bg, w_bg = bg_bgr.shape[:2]
        
        # 2. Extract shape styling and text styling for all components
        comp_styles = {}
        for comp in components:
            mask_rel = comp.amodal_mask_path or comp.mask_path
            mask_abs = ""
            if mask_rel:
                mask_abs = os.path.join(slide_dir, mask_rel.replace("/", os.sep))
                if not os.path.exists(mask_abs):
                    mask_abs = os.path.join(slide_dir, "masks", os.path.basename(mask_rel))
            
            # Run Shape Classification
            style = self.shape_classifier.classify_shape_and_style(
                original_img_path=original_img_path,
                mask_png_path=mask_abs,
                box=comp.box,
                comp_type=comp.type,
                semantic_name=comp.semantic_name or ""
            )
            
            # Run Text style extraction if text_label
            text_style = {}
            if comp.type == "text_label" and comp.text:
                text_style = self.text_extractor.extract_text_style(
                    original_img_path=original_img_path,
                    box=comp.box,
                    text_content=comp.text
                )
                
            comp_styles[comp.id] = {
                "style": style,
                "text_style": text_style,
                "mask_abs_path": mask_abs
            }

        # Identify major elements to shift
        shift_ids = set()
        filename_lower = original_filename.lower()
        
        # Specific shift targets based on diagram type
        if "digestive" in filename_lower:
            # Shift liver
            for c in components:
                if c.semantic_name and "liver" in c.semantic_name.lower():
                    shift_ids.add(c.id)
        elif "solar" in filename_lower:
            # Shift earth or jupiter
            for c in components:
                if c.semantic_name and ("earth" in c.semantic_name.lower() or "jupiter" in c.semantic_name.lower()):
                    shift_ids.add(c.id)
        elif "admission" in filename_lower:
            # Shift submit and eligible (decision)
            for c in components:
                if c.semantic_name and ("submit" in c.semantic_name.lower() or "eligible" in c.semantic_name.lower()):
                    shift_ids.add(c.id)
        elif "parts_of_speech" in filename_lower or "learning" in filename_lower:
            # Shift first card
            for c in components:
                if c.type == "shape" or c.type == "image_object":
                    shift_ids.add(c.id)
                    break
        elif "canva" in filename_lower:
            # Shift card_1
            for c in components:
                if c.semantic_name and "card_1" in c.semantic_name.lower():
                    shift_ids.add(c.id)
                    
        # Fallback shift if no target found
        if not shift_ids:
            for c in components:
                if c.type in ["image_object", "shape"]:
                    shift_ids.add(c.id)
                    break

        # Render composite functions
        def render_composite(shift_x: int = 0) -> Image.Image:
            # Convert BGR background to PIL
            bg_pil = Image.fromarray(cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(bg_pil)
            
            # Draw components in order of Z-index
            sorted_comps = sorted(components, key=lambda c: c.z_index)
            for comp in sorted_comps:
                if not comp.visible:
                    continue
                    
                cs = comp_styles[comp.id]
                style = cs["style"]
                text_style = cs["text_style"]
                mask_path = cs["mask_abs_path"]
                
                # Bounding box
                x, y, w, h = comp.box
                if comp.id in shift_ids:
                    x += shift_x
                    
                if w <= 0 or h <= 0:
                    continue
                    
                # 1. Native Shapes (Category A)
                if style["export_strategy"] == "Native":
                    shape_type = style["shape_type"]
                    fill_col = style["fill_color"]
                    border_col = style["border_color"]
                    thickness = int(style["border_thickness"])
                    
                    if shape_type == "Rectangle":
                        draw.rectangle([x, y, x + w, y + h], fill=fill_col, outline=border_col, width=thickness)
                    elif shape_type == "Rounded Rectangle":
                        draw.rounded_rectangle([x, y, x + w, y + h], radius=10, fill=fill_col, outline=border_col, width=thickness)
                    elif shape_type == "Circle":
                        draw.ellipse([x, y, x + w, y + h], fill=fill_col, outline=border_col, width=thickness)
                    elif shape_type == "Diamond":
                        # Diamond polygon
                        pts = [(x + w//2, y), (x + w, y + h//2), (x + w//2, y + h), (x, y + h//2)]
                        draw.polygon(pts, fill=fill_col, outline=border_col)
                    elif shape_type in ["Line", "Arrow"]:
                        # Draw connector line
                        draw.line([(x, y + h//2), (x + w, y + h//2)], fill=border_col, width=max(2, thickness))
                        if shape_type == "Arrow":
                            # Draw arrowhead polygon at the end of the line
                            arrow_pts = [(x + w, y + h//2), (x + w - 8, y + h//2 - 5), (x + w - 8, y + h//2 + 5)]
                            draw.polygon(arrow_pts, fill=border_col)
                            
                # 2. Transparent organic PNG cutouts (Category C)
                else:
                    if mask_path and os.path.exists(mask_path):
                        try:
                            comp_img = Image.open(mask_path).convert("RGBA")
                            # Resize to target box
                            comp_img = comp_img.resize((w, h), Image.Resampling.LANCZOS)
                            bg_pil.paste(comp_img, (x, y), comp_img)
                        except Exception as e:
                            print(f"Error drawing organic PNG {comp.id}: {e}")
                            
                # 3. Text label overlays
                if comp.type == "text_label" and comp.text:
                    f_col = text_style.get("font_color", (0, 0, 0))
                    f_size = text_style.get("estimated_font_size", 12)
                    bold = text_style.get("bold_estimate", False)
                    align = text_style.get("alignment", "center")
                    
                    font_names = ["arialbd.ttf" if bold else "arial.ttf", "arial.ttf"]
                    font = None
                    for fn in font_names:
                        try:
                            font = ImageFont.truetype(fn, f_size)
                            break
                        except OSError:
                            pass
                    if font is None:
                        font = ImageFont.load_default()
                        
                    lines = comp.text.split("\n")
                    y_text = y + 4
                    for line in lines:
                        # Draw line with alignment padding
                        try:
                            left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
                            tw = right - left
                        except AttributeError:
                            tw = draw.textsize(line, font=font)[0]
                            
                        if align == "center":
                            x_text = x + (w - tw) // 2
                        elif align == "right":
                            x_text = x + w - tw - 6
                        else:
                            x_text = x + 6
                            
                        draw.text((x_text, y_text), line, fill=f_col, font=font)
                        y_text += f_size + 3
                        
            return bg_pil

        # 3. Create PowerPoint Selection Pane Sidebar
        def draw_selection_pane(w_sidebar: int, h_sidebar: int) -> Image.Image:
            sidebar = Image.new("RGB", (w_sidebar, h_sidebar), (243, 244, 246)) # slate-100 bg
            draw = ImageDraw.Draw(sidebar)
            
            # Title
            font_title = ImageFont.load_default()
            try:
                font_title = ImageFont.truetype("arialbd.ttf", 14)
                font_item = ImageFont.truetype("arial.ttf", 11)
            except OSError:
                font_item = ImageFont.load_default()
                
            draw.text((15, 15), "Selection", fill=(15, 23, 42), font=font_title)
            draw.text((15, 38), "Show All   Hide All", fill=(59, 130, 246), font=font_item)
            draw.line([(0, 58), (w_sidebar, 58)], fill=(203, 213, 225), width=1)
            
            # List components in reverse Z-order (top layers first)
            sorted_comps = sorted(components, key=lambda c: c.z_index, reverse=True)
            y_offset = 68
            
            for comp in sorted_comps:
                if not comp.visible:
                    continue
                    
                cs = comp_styles[comp.id]
                style = cs["style"]
                shape_type = style["shape_type"]
                strategy = style["export_strategy"]
                
                # Create friendly PowerPoint-style layer name
                if comp.type == "text_label":
                    layer_name = f"TextBox: {comp.text.split(chr(10))[0][:16]}"
                elif shape_type == "Rich Object":
                    layer_name = f"Picture ({comp.semantic_name or 'Organic'})"
                else:
                    layer_name = f"{shape_type}: {comp.semantic_name or 'Element'}"
                    
                # Item background card
                draw.rectangle([10, y_offset, w_sidebar - 10, y_offset + 28], fill=(255, 255, 255), outline=(226, 232, 240), width=1)
                
                # Draw layer name
                draw.text((20, y_offset + 7), layer_name, fill=(51, 65, 85), font=font_item)
                
                # Draw small simulated eye icon (visible check mark / circle)
                draw.ellipse([w_sidebar - 30, y_offset + 9, w_sidebar - 20, y_offset + 19], fill=(187, 247, 208), outline=(34, 197, 94))
                
                y_offset += 34
                if y_offset > h_sidebar - 35:
                    # Draw indicator for overflow
                    draw.text((15, h_sidebar - 25), "... more shapes", fill=(100, 100, 100), font=font_item)
                    break
                    
            return sidebar

        # 4. Generate Slide Composites
        slide_before = render_composite(shift_x=0)
        slide_after = render_composite(shift_x=100)
        
        # Draw text indicator on Moved composite
        draw_after = ImageDraw.Draw(slide_after)
        try:
            f_warn = ImageFont.truetype("arialbd.ttf", 13)
        except OSError:
            f_warn = ImageFont.load_default()
            
        for comp in components:
            if comp.id in shift_ids:
                x, y, w, h = comp.box
                draw_after.rectangle([x + 100, y, x + 100 + w, y + h], outline=(239, 68, 68), width=2)
                draw_after.text((x + 100, y - 18), "Moved +100px", fill=(239, 68, 68), font=f_warn)
                
        # 5. Create final side-by-side images with Selection Pane sidebar
        w_sidebar = 240
        h_sidebar = h_bg
        
        pane_img = draw_selection_pane(w_sidebar, h_sidebar)
        
        # Combine slide_before + Selection Pane
        combined_before = Image.new("RGB", (w_bg + w_sidebar, h_bg))
        combined_before.paste(slide_before, (0, 0))
        combined_before.paste(pane_img, (w_bg, 0))
        
        # Combine slide_after + Selection Pane
        combined_after = Image.new("RGB", (w_bg + w_sidebar, h_bg))
        combined_after.paste(slide_after, (0, 0))
        combined_after.paste(pane_img, (w_bg, 0))
        
        # Save files
        before_filename = f"slide_{slide_index}_ppt_before.png"
        after_filename = f"slide_{slide_index}_ppt_after.png"
        
        os.makedirs(output_dir, exist_ok=True)
        before_abs_path = os.path.join(output_dir, before_filename)
        after_abs_path = os.path.join(output_dir, after_filename)
        
        combined_before.save(before_abs_path)
        combined_after.save(after_abs_path)
        
        # Write individual amodal checkerboard assets for organic occlusions (e.g. esophagus)
        for comp in components:
            if comp.is_occluded and comp.amodal_mask_path:
                try:
                    mask_abs = os.path.join(slide_dir, comp.amodal_mask_path.replace("/", os.sep))
                    if os.path.exists(mask_abs):
                        rgba = cv2.imread(mask_abs, cv2.IMREAD_UNCHANGED)
                        if rgba is not None and rgba.shape[2] == 4:
                            h_m, w_m = rgba.shape[:2]
                            checker = np.zeros((h_m, w_m, 3), dtype=np.uint8)
                            box_size = 8
                            for y_c in range(0, h_m, box_size):
                                for x_c in range(0, w_m, box_size):
                                    color = 255 if ((x_c // box_size) + (y_c // box_size)) % 2 == 0 else 220
                                    checker[y_c:y_c+box_size, x_c:x_c+box_size] = color
                                    
                            alpha = rgba[:, :, 3] / 255.0
                            alpha = np.expand_dims(alpha, axis=2)
                            composite = (rgba[:, :, :3] * alpha + checker * (1.0 - alpha)).astype(np.uint8)
                            
                            # Save organic amodal checkerboard verification
                            checker_name = f"slide_{slide_index}_{comp.semantic_name or 'shape'}_amodal_checkerboard.png"
                            cv2.imwrite(os.path.join(output_dir, checker_name), composite)
                except Exception as e_check:
                    print(f"Error generating checkerboard for {comp.id}: {e_check}")
                    
        return {
            "ppt_before": before_abs_path,
            "ppt_after": after_abs_path
        }
