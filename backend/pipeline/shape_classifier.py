import os
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional

class ShapeGeometryClassifier:
    def __init__(self):
        pass

    def classify_shape_and_style(
        self,
        original_img_path: str,
        mask_png_path: str,
        box: Tuple[int, int, int, int],
        comp_type: str,
        semantic_name: str
    ) -> Dict[str, Any]:
        """
        Runs contour analysis, visual corner detection, and color extraction on the crop.
        Determines shape type using ONLY contour geometry and Shi-Tomasi corner detection,
        WITHOUT relying on semantic name or filename keywords.
        Returns a style dictionary:
        {
           "shape_type": "Rectangle" | "Rounded Rectangle" | "Circle" | "Diamond" | "Line" | "Arrow" | "Rich Object",
           "confidence": float,
           "export_strategy": "Native" | "PNG",
           "fill_color": (r, g, b),
           "border_color": (r, g, b),
           "border_thickness": float,
           "corner_radius": float
        }
        """
        bx, by, bw, bh = box
        default_fill = (240, 240, 240)  # Light gray default
        default_border = (100, 100, 100) # dark gray border
        
        default_style = {
            "shape_type": "Rich Object",
            "confidence": 0.90,
            "export_strategy": "PNG",
            "fill_color": default_fill,
            "border_color": default_border,
            "border_thickness": 1.5,
            "corner_radius": 0.0
        }
        
        # Load the transparent mask crop to get the exact shape boundaries
        if not os.path.exists(mask_png_path):
            return self._fallback_by_type(comp_type, default_fill, default_border)
            
        mask_rgba = cv2.imread(mask_png_path, cv2.IMREAD_UNCHANGED)
        if mask_rgba is None or mask_rgba.shape[2] < 4:
            return self._fallback_by_type(comp_type, default_fill, default_border)
            
        # Get BGR crop from original image for color extraction
        original_img = cv2.imread(original_img_path)
        if original_img is None:
            return default_style
            
        h_orig, w_orig = original_img.shape[:2]
        x_start = max(0, bx)
        y_start = max(0, by)
        x_end = min(w_orig, bx + bw)
        y_end = min(h_orig, by + bh)
        
        # If box is out of bounds or empty
        if (x_end - x_start) <= 0 or (y_end - y_start) <= 0:
            return default_style
            
        crop_bgr = original_img[y_start:y_end, x_start:x_end]
        
        # Extract alpha mask and ensure it matches the crop size
        alpha = mask_rgba[:, :, 3]
        if crop_bgr.shape[0] != alpha.shape[0] or crop_bgr.shape[1] != alpha.shape[1]:
            alpha = cv2.resize(alpha, (crop_bgr.shape[1], crop_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        # 1. Contour Analysis
        contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._fallback_by_type(comp_type, default_fill, default_border)
            
        # Find the largest contour
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        
        if perimeter == 0 or area == 0:
            return self._fallback_by_type(comp_type, default_fill, default_border)
            
        # Approximate polygon vertices
        approx = cv2.approxPolyDP(c, 0.035 * perimeter, True)
        num_vertices = len(approx)
        
        # Circularity score (4 * pi * area / perimeter^2)
        circularity = (4.0 * np.pi * area) / (perimeter ** 2)
        
        # Solidity (contour area / bounding box area)
        bbox_area = (x_end - x_start) * (y_end - y_start)
        solidity = float(area) / bbox_area if bbox_area > 0 else 0.0
        
        # Convexity (contour area / convex hull area)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0.0
        
        # 2. Visual Corner Analysis (Shi-Tomasi corner count inside crop, masked to shape)
        gray_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray_crop, gray_crop, mask=alpha)
        corners = cv2.goodFeaturesToTrack(masked_gray, maxCorners=12, qualityLevel=0.08, minDistance=6)
        corner_count = len(corners) if corners is not None else 0
        
        # 3. Shape Decision Logic based strictly on geometry & corner properties
        shape_type = "Rich Object"
        confidence = 0.85
        export_strategy = "PNG"
        corner_radius = 0.0
        
        aspect_ratio = float(max(bw, bh)) / max(1, min(bw, bh))
        
        # Rules for categorization:
        # A1/A2 primitives: Rectangle, Rounded Rectangle, Circle, Diamond, Line, Arrow
        
        # Rule 1: Thin elongated items are Lines/Connectors/Arrows
        if aspect_ratio > 3.8 or bw < 12 or bh < 12:
            if comp_type == "arrow":
                shape_type = "Arrow"
            else:
                shape_type = "Line"
            export_strategy = "Native"
            confidence = 0.95
        # Rule 2: Circle / Ellipse (no sharp corners, high circularity)
        elif circularity > 0.74 and corner_count <= 2:
            shape_type = "Circle"
            export_strategy = "Native"
            confidence = 0.90
        # Rule 3: Diamond (4 sharp corners, solidity is ~0.5 relative to bounding box)
        elif 0.40 <= solidity <= 0.68 and corner_count == 4 and num_vertices == 4 and aspect_ratio < 1.4:
            shape_type = "Diamond"
            export_strategy = "Native"
            confidence = 0.92
        # Rule 4: Rectangle (4 corners, high solidity, vertices is ~4)
        elif solidity > 0.93 and corner_count == 4 and num_vertices == 4:
            shape_type = "Rectangle"
            export_strategy = "Native"
            confidence = 0.95
        # Rule 5: Rounded Rectangle (solidity slightly lower than rectangle, curved corners -> higher vertex count, 4+ visual corners)
        elif 0.78 <= solidity <= 0.93 and (corner_count >= 4 or num_vertices >= 5) and aspect_ratio < 2.5:
            shape_type = "Rounded Rectangle"
            export_strategy = "Native"
            corner_radius = 0.15 # curve ratio
            confidence = 0.92
        else:
            # Fallback for text boxes which are always native rectangles
            if comp_type == "text_label":
                shape_type = "Rectangle"
                export_strategy = "Native"
                confidence = 0.95
            elif comp_type == "arrow":
                shape_type = "Arrow"
                export_strategy = "Native"
                confidence = 0.90
            else:
                # Organic objects (e.g. liver, stomach, planet details)
                shape_type = "Rich Object"
                export_strategy = "PNG"
                confidence = 0.95
                
        # 4. Color & Border Styling parameter extraction
        fill_rgb = default_fill
        border_rgb = default_border
        border_thickness = 1.5
        
        if crop_bgr.size > 0:
            # Create eroded mask for interior fill sampling (avoid borders)
            kernel = np.ones((5, 5), dtype=np.uint8)
            eroded_alpha = cv2.erode(alpha, kernel, iterations=1)
            
            # Median of interior pixels
            interior_pixels = crop_bgr[eroded_alpha > 0]
            if len(interior_pixels) > 0:
                median_bgr = np.median(interior_pixels, axis=0)
                # Convert BGR to RGB
                fill_rgb = (int(median_bgr[2]), int(median_bgr[1]), int(median_bgr[0]))
                
            # Sample border pixels (subtract eroded mask from full boundary)
            border_mask = cv2.subtract(alpha, eroded_alpha)
            border_pixels = crop_bgr[border_mask > 0]
            if len(border_pixels) > 0:
                median_border_bgr = np.median(border_pixels, axis=0)
                border_rgb = (int(median_border_bgr[2]), int(median_border_bgr[1]), int(median_border_bgr[0]))
                
            # Border thickness estimate based on boundary mask width
            border_pixels_count = np.sum(border_mask > 0)
            if perimeter > 0:
                thickness = float(border_pixels_count) / perimeter
                border_thickness = max(1.0, min(float(round(thickness, 1)), 4.0))

        # Ensure the fill color is not identical to border to keep outline clear
        if fill_rgb == border_rgb:
            # Shift border color slightly darker
            border_rgb = (max(0, border_rgb[0] - 40), max(0, border_rgb[1] - 40), max(0, border_rgb[2] - 40))

        return {
            "shape_type": shape_type,
            "confidence": float(round(confidence, 2)),
            "export_strategy": export_strategy,
            "fill_color": fill_rgb,
            "border_color": border_rgb,
            "border_thickness": border_thickness,
            "corner_radius": corner_radius
        }

    def _fallback_by_type(self, comp_type: str, fill: Tuple[int, int, int], border: Tuple[int, int, int]) -> Dict[str, Any]:
        """Simple fallback strategy when files do not exist."""
        if comp_type == "text_label":
            return {
                "shape_type": "Rectangle", "confidence": 0.95, "export_strategy": "Native",
                "fill_color": fill, "border_color": border, "border_thickness": 1.5, "corner_radius": 0.0
            }
        elif comp_type == "arrow":
            return {
                "shape_type": "Arrow", "confidence": 0.95, "export_strategy": "Native",
                "fill_color": fill, "border_color": border, "border_thickness": 2.0, "corner_radius": 0.0
            }
        else:
            return {
                "shape_type": "Rich Object", "confidence": 0.90, "export_strategy": "PNG",
                "fill_color": fill, "border_color": border, "border_thickness": 1.5, "corner_radius": 0.0
            }


class TextStyleExtractor:
    def __init__(self):
        pass

    def extract_text_style(
        self,
        original_img_path: str,
        box: Tuple[int, int, int, int],
        text_content: str
    ) -> Dict[str, Any]:
        """
        Extracts font_color, estimated_font_size, bold_estimate, and alignment from textbox region.
        """
        bx, by, bw, bh = box
        default_color = (0, 0, 0) # black default
        
        # Load crop
        original_img = cv2.imread(original_img_path)
        if original_img is None:
            return {
                "font_color": default_color,
                "estimated_font_size": 13,
                "bold_estimate": True,
                "alignment": "center"
            }
            
        h_orig, w_orig = original_img.shape[:2]
        x_start = max(0, bx)
        y_start = max(0, by)
        x_end = min(w_orig, bx + bw)
        y_end = min(h_orig, by + bh)
        
        if (x_end - x_start) <= 0 or (y_end - y_start) <= 0:
            return {
                "font_color": default_color,
                "estimated_font_size": 13,
                "bold_estimate": True,
                "alignment": "center"
            }
            
        crop_bgr = original_img[y_start:y_end, x_start:x_end]
        
        # Detect text pixels (typically dark strokes)
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        # Apply Otsu threshold to separate dark strokes from light background
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Get color of stroke pixels
        stroke_pixels = crop_bgr[thresh > 0]
        font_color = default_color
        if len(stroke_pixels) > 0:
            median_stroke = np.median(stroke_pixels, axis=0)
            # Convert BGR to RGB
            font_color = (int(median_stroke[2]), int(median_stroke[1]), int(median_stroke[0]))
            
        # Estimate lines of text
        text_str = text_content or ""
        lines_count = max(1, len(text_str.split("\n")))
        
        # Font size estimate
        size_est = int((y_end - y_start) / (lines_count * 1.5))
        estimated_font_size = max(10, min(size_est, 20))
        
        # Bold estimate based on density of text stroke pixels
        stroke_ratio = float(np.sum(thresh > 0)) / thresh.size if thresh.size > 0 else 0.0
        bold_estimate = True if stroke_ratio > 0.08 else False
        
        # Text alignment extraction: check horizontal center of mass
        text_pixels_y, text_pixels_x = np.where(thresh > 0)
        alignment = "center"
        if len(text_pixels_x) > 0:
            mean_x = np.mean(text_pixels_x)
            col_ratio = mean_x / (x_end - x_start)
            if col_ratio < 0.40:
                alignment = "left"
            elif col_ratio > 0.60:
                alignment = "right"
            else:
                alignment = "center"
        
        return {
            "font_color": font_color,
            "estimated_font_size": estimated_font_size,
            "bold_estimate": bold_estimate,
            "alignment": alignment
        }
