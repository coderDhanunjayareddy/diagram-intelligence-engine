import os
import cv2
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from backend.pipeline.interfaces import BatchJob, SlideMetadata

class PPTGenerator:
    def generate_batch_pptx(self, job: BatchJob, batch_dir: str, output_pptx_path: str):
        """
        Compiles all processed slides in a BatchJob into a single PowerPoint (.pptx) file.
        Uses a layered fallback strategy with background erasure:
        - Layer 1 (Bottom): Clean background template with extracted elements erased.
        - Layer 2 (Middle): Individual transparent segmented shapes/arrows.
        - Layer 3 (Top): Editable text shapes containing labels.
        """
        prs = Presentation()
        
        # Configure slide dimensions dynamically based on first slide aspect ratio
        if job.slides:
            first_slide = job.slides[0]
            img_w = max(1, first_slide.width)
            img_h = max(1, first_slide.height)
            aspect_ratio = img_w / img_h
            if abs(aspect_ratio - 1.0) < 0.05: # Square image (1:1 ratio)
                prs.slide_width = Inches(10)
                prs.slide_height = Inches(10)
            elif aspect_ratio > 1.4: # Widescreen 16:9
                prs.slide_width = Inches(13.33)
                prs.slide_height = Inches(7.5)
            elif aspect_ratio > 1.2: # Standard 4:3
                prs.slide_width = Inches(10)
                prs.slide_height = Inches(7.5)
            else: # Portrait or other custom shape
                prs.slide_height = Inches(10)
                prs.slide_width = Inches(10 * aspect_ratio)
        else:
            prs.slide_width = Inches(13.33)
            prs.slide_height = Inches(7.5)
        
        blank_slide_layout = prs.slide_layouts[6] # blank layout
        
        for slide_meta in job.slides:
            # 1. Add slide
            slide = prs.slides.add_slide(blank_slide_layout)
            
            # Find the original image path for this slide
            slide_dir = os.path.join(batch_dir, "slides", f"slide_{slide_meta.slide_index}")
            
            # Check if background template exists, otherwise fall back to original
            background_image_path = os.path.join(slide_dir, "masks", "background.png")
            if not os.path.exists(background_image_path):
                background_image_path = os.path.join(slide_dir, "background.png")
                
            if not os.path.exists(background_image_path):
                original_dir = os.path.join(slide_dir, "original")
                if os.path.exists(original_dir) and os.listdir(original_dir):
                    background_image_path = os.path.join(original_dir, os.listdir(original_dir)[0])
                else:
                    original_file = None
                    if os.path.exists(slide_dir):
                        for f in os.listdir(slide_dir):
                            if f.startswith("original."):
                                original_file = f
                                break
                    if original_file:
                        background_image_path = os.path.join(slide_dir, original_file)
                    else:
                        continue
            
            # 2. Add Layer 1: Bottom Clean Background Image
            slide.shapes.add_picture(
                background_image_path, 
                Inches(0), Inches(0), 
                prs.slide_width, prs.slide_height
            )
            
            # Coordinate scaling factors: image pixels to Inches
            img_w = max(1, slide_meta.width)
            img_h = max(1, slide_meta.height)
            
            scale_x = float(prs.slide_width) / img_w
            scale_y = float(prs.slide_height) / img_h
            
            # 3. Add Layer 2 and 3: Extracted transparent components and text labels
            sorted_components = sorted(slide_meta.components, key=lambda c: c.z_index)
            
            for comp in sorted_components:
                if not comp.visible:
                    continue
                    
                # Compute scaling coordinates (already in EMUs)
                left = int(comp.box[0] * scale_x)
                top = int(comp.box[1] * scale_y)
                width = int(comp.box[2] * scale_x)
                height = int(comp.box[3] * scale_y)
                
                # Safeguard dimensions (avoid zero/negative sizes)
                if width <= 0 or height <= 0:
                    continue
                    
                # 3a. Segmented Transparent PNG Layers (shapes, arrows)
                if comp.type in ["image_object", "arrow"] and comp.mask_path:
                    mask_abs_path = os.path.abspath(os.path.join(slide_dir, comp.mask_path))
                    if not os.path.exists(mask_abs_path):
                        # Fallback: check inside the masks/ subdirectory
                        mask_abs_path = os.path.abspath(os.path.join(slide_dir, "masks", comp.mask_path))
                    if os.path.exists(mask_abs_path):
                        try:
                            slide.shapes.add_picture(mask_abs_path, left, top, width, height)
                        except Exception as e:
                            print(f"Error embedding transparent crop {comp.id}: {e}")
                            
                # 3b. Text Labels (using standard Rectangle shapes to prevent PPTX XML corruption)
                elif comp.type == "text_label" and comp.text:
                    try:
                        # Use a standard Rectangle shape instead of TextBox to allow safe border/fill styling
                        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
                        
                        # Style background fill as transparent (no fill)
                        fill = shape.fill
                        fill.background()
                        
                        # Style line border as transparent (no border)
                        line = shape.line
                        line.fill.background()
                        
                        # Configure text container padding
                        tf = shape.text_frame
                        tf.word_wrap = True
                        tf.margin_left = Inches(0.05)
                        tf.margin_right = Inches(0.05)
                        tf.margin_top = Inches(0.02)
                        tf.margin_bottom = Inches(0.02)
                        
                        # Set paragraph run parameters (text color must be black)
                        p = tf.paragraphs[0]
                        p.text = comp.text
                        p.font.name = 'Arial'
                        p.font.size = Pt(13)
                        p.font.bold = True
                        p.font.color.rgb = RGBColor(0, 0, 0) # black text
                        
                    except Exception as e:
                        print(f"Error building editable text shape {comp.id}: {e}")
                        
        # Save presentation to output path
        prs.save(output_pptx_path)
        print(f"PowerPoint Presentation exported to: {output_pptx_path}")
