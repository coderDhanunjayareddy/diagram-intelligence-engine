import os
import json
from typing import List, Dict, Any

class PresentationReadinessAnalyzer:
    def __init__(self):
        pass

    def analyze(self, slides: List[Any], batch_dir: str = None) -> Dict[str, Any]:
        """
        Analyzes the presentation quality across the given list of slides.
        Supports both Pydantic models (SlideMetadata) and SQLAlchemy DB instances (DbSlideMetadata).
        """
        total_slides = len(slides)
        if total_slides == 0:
            return {
                "editability_score": 1.0,
                "animation_ready_score": 1.0,
                "teacher_ready_score": 1.0,
                "avg_elements_per_slide": 0.0,
                "reconstructed_elements_per_slide": 0.0,
                "total_elements": 0,
                "total_text_labels": 0,
                "total_occluded": 0,
                "total_reconstructed": 0
            }

        total_elements = 0
        total_text_labels = 0
        total_occluded = 0
        total_reconstructed = 0
        
        sum_ocr_confidence = 0.0
        count_ocr_labels = 0
        sum_mapped_labels = 0
        count_bg_erasure_success = 0
        
        slide_details = []

        for idx, slide in enumerate(slides):
            # Resolve properties whether slide is dict, Pydantic model, or DB object
            slide_idx = getattr(slide, "slide_index", idx)
            orig_filename = getattr(slide, "original_filename", f"slide_{idx}")
            components = getattr(slide, "components", [])
            
            # Check for background.png success
            bg_success = False
            if batch_dir:
                slide_task_dir = os.path.join(batch_dir, "slides", f"slide_{slide_idx}")
                bg_path = os.path.join(slide_task_dir, "masks", "background.png")
                if os.path.exists(bg_path) and os.path.getsize(bg_path) > 0:
                    bg_success = True
            else:
                # Default to true if not checking path
                bg_success = True

            if bg_success:
                count_bg_erasure_success += 1

            slide_elements = 0
            slide_text_labels = 0
            slide_occluded = 0
            slide_reconstructed = 0
            slide_ocr_conf_sum = 0.0
            slide_ocr_conf_count = 0
            slide_mapped_labels = 0

            for comp in components:
                # Resolve attributes from Pydantic or DB object
                comp_type = getattr(comp, "type", "")
                comp_text = getattr(comp, "text", None)
                comp_confidence = getattr(comp, "confidence", 1.0)
                comp_is_occluded = getattr(comp, "is_occluded", False)
                comp_amodal_mask_path = getattr(comp, "amodal_mask_path", None)
                comp_assoc_obj = getattr(comp, "associated_object_id", None)
                
                slide_elements += 1
                
                if comp_type == "text_label":
                    slide_text_labels += 1
                    slide_ocr_conf_sum += comp_confidence
                    slide_ocr_conf_count += 1
                    if comp_assoc_obj is not None or comp_text:
                        slide_mapped_labels += 1
                
                if comp_is_occluded:
                    slide_occluded += 1
                    if comp_amodal_mask_path is not None:
                        slide_reconstructed += 1

            # Update globals
            total_elements += slide_elements
            total_text_labels += slide_text_labels
            total_occluded += slide_occluded
            total_reconstructed += slide_reconstructed
            
            sum_ocr_confidence += slide_ocr_conf_sum
            count_ocr_labels += slide_ocr_conf_count
            sum_mapped_labels += slide_mapped_labels
            
            # Slide-level scores
            slide_editability = float(slide_text_labels) / max(1, slide_elements)
            slide_animation = float(slide_reconstructed) / max(1, slide_occluded) if slide_occluded > 0 else 1.0
            
            slide_avg_ocr_conf = float(slide_ocr_conf_sum) / max(1, slide_ocr_conf_count) if slide_ocr_conf_count > 0 else 1.0
            slide_label_coverage = float(slide_mapped_labels) / max(1, slide_text_labels) if slide_text_labels > 0 else 1.0
            slide_bg_val = 1.0 if bg_success else 0.0
            slide_teacher = (slide_avg_ocr_conf * 0.4) + (slide_label_coverage * 0.3) + (slide_bg_val * 0.3)

            slide_details.append({
                "slide_index": slide_idx,
                "original_filename": orig_filename,
                "elements_count": slide_elements,
                "text_labels_count": slide_text_labels,
                "occluded_count": slide_occluded,
                "reconstructed_count": slide_reconstructed,
                "editability_score": float(round(slide_editability, 3)),
                "animation_ready_score": float(round(slide_animation, 3)),
                "teacher_ready_score": float(round(slide_teacher, 3))
            })

        # Calculate global presentation metrics
        editability_score = float(total_text_labels) / max(1, total_elements)
        animation_ready_score = float(total_reconstructed) / max(1, total_occluded) if total_occluded > 0 else 1.0
        
        global_avg_ocr_conf = float(sum_ocr_confidence) / max(1, count_ocr_labels) if count_ocr_labels > 0 else 1.0
        global_label_coverage = float(sum_mapped_labels) / max(1, total_text_labels) if total_text_labels > 0 else 1.0
        global_bg_val = float(count_bg_erasure_success) / total_slides
        
        teacher_ready_score = (global_avg_ocr_conf * 0.4) + (global_label_coverage * 0.3) + (global_bg_val * 0.3)
        
        avg_elements_per_slide = float(total_elements) / total_slides
        reconstructed_elements_per_slide = float(total_reconstructed) / total_slides

        results = {
            "editability_score": float(round(editability_score, 3)),
            "animation_ready_score": float(round(animation_ready_score, 3)),
            "teacher_ready_score": float(round(teacher_ready_score, 3)),
            "avg_elements_per_slide": float(round(avg_elements_per_slide, 2)),
            "reconstructed_elements_per_slide": float(round(reconstructed_elements_per_slide, 2)),
            "total_elements": total_elements,
            "total_text_labels": total_text_labels,
            "total_occluded": total_occluded,
            "total_reconstructed": total_reconstructed,
            "slide_details": slide_details
        }

        return results

    def save_reports(self, dest_dir: str, results: Dict[str, Any]):
        """
        Saves presentation_quality_report.json and presentation_quality_report.md
        to the specified destination directory.
        """
        os.makedirs(dest_dir, exist_ok=True)
        
        # Save JSON
        json_path = os.path.join(dest_dir, "presentation_quality_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            
        # Save MD
        md_path = os.path.join(dest_dir, "presentation_quality_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# V4 Presentation Quality & Classroom Readiness Report\n\n")
            f.write("This report evaluates the final PowerPoint presentation against classroom teaching requirements, focus layouts, and layer editability criteria.\n\n")
            f.write("---\n\n")
            
            f.write("## 🏆 Core Diagram Readiness Scores\n\n")
            f.write(f"- **Editability Score**: `{results['editability_score'] * 100:.1f}%` (ratio of text labels to total elements)\n")
            f.write(f"- **Animation Readiness Score**: `{results['animation_ready_score'] * 100:.1f}%` (layered occluded elements reconstructed amodally)\n")
            f.write(f"- **Teacher Classroom Readiness Score**: `{results['teacher_ready_score'] * 100:.1f}%` (synthesizing OCR quality, connections, and clean erasure template)\n\n")
            
            f.write("---\n\n")
            
            f.write("## 📊 Element Density Metrics\n\n")
            f.write(f"- **Average Elements per Slide**: `{results['avg_elements_per_slide']}`\n")
            f.write(f"- **Average Reconstructed Amodal Elements per Slide**: `{results['reconstructed_elements_per_slide']}`\n")
            f.write(f"- **Total Components Processed**: `{results['total_elements']}`\n")
            f.write(f"- **Total Reconstructed Shapes**: `{results['total_reconstructed']}`\n\n")
            
            f.write("---\n\n")
            
            f.write("## 🎞️ Slide-by-Slide Quality Breakdown\n\n")
            f.write("| Slide # | Original Filename | Elements | Occluded | Reconstructed | Editability | Animation | Teacher Ready |\n")
            f.write("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for slide in results["slide_details"]:
                f.write(
                    f"| Slide {slide['slide_index']} | {slide['original_filename']} | "
                    f"{slide['elements_count']} | {slide['occluded_count']} | {slide['reconstructed_count']} | "
                    f"{slide['editability_score'] * 100:.0f}% | {slide['animation_ready_score'] * 100:.0f}% | "
                    f"{slide['teacher_ready_score'] * 100:.0f}% |\n"
                )
            f.write("\n")
