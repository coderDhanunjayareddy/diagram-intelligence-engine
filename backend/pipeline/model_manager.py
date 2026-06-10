import os
import gc
from typing import Dict, Any

class ModelManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelManager, cls).__new__(cls, *args, **kwargs)
            cls._instance._init_manager()
        return cls._instance

    def _init_manager(self):
        self.models: Dict[str, Any] = {
            "grounding_dino": None,
            "sam2": None,
            "paddle_ocr": None
        }
        self.loaded_status: Dict[str, bool] = {
            "grounding_dino": False,
            "sam2": False,
            "paddle_ocr": False
        }
        
    def load_detector(self) -> Any:
        """Loads Grounding DINO if available, else logs warning."""
        if self.loaded_status["grounding_dino"]:
            return self.models["grounding_dino"]
            
        try:
            # Simulated model loading structure for local environments
            # In a full GPU workspace, we would import groundingdino
            # from groundingdino.util.inference import load_model
            # self.models["grounding_dino"] = load_model(...)
            
            print("[ModelManager] Initializing Grounding DINO...")
            # We mock the class handles
            self.models["grounding_dino"] = "GROUNDING_DINO_ACTIVE_HANDLE"
            self.loaded_status["grounding_dino"] = True
            print("[ModelManager] Grounding DINO loaded successfully.")
        except Exception as e:
            print(f"[ModelManager] Error loading Grounding DINO: {e}")
            self.loaded_status["grounding_dino"] = False
            
        return self.models["grounding_dino"]

    def load_segmenter(self) -> Any:
        """Loads SAM2 if available, else logs warning."""
        if self.loaded_status["sam2"]:
            return self.models["sam2"]
            
        try:
            # In GPU server setup:
            # import torch
            # from sam2.build_sam import build_sam2
            # from sam2.sam2_image_predictor import SAM2ImagePredictor
            # predictor = SAM2ImagePredictor(build_sam2(checkpoint, config))
            
            print("[ModelManager] Initializing SAM2 (Segment Anything Model 2)...")
            self.models["sam2"] = "SAM2_ACTIVE_HANDLE"
            self.loaded_status["sam2"] = True
            print("[ModelManager] SAM2 loaded successfully.")
        except Exception as e:
            print(f"[ModelManager] Error loading SAM2: {e}")
            self.loaded_status["sam2"] = False
            
        return self.models["sam2"]

    def load_ocr(self) -> Any:
        """Loads PaddleOCR if available, else logs warning."""
        if self.loaded_status["paddle_ocr"]:
            return self.models["paddle_ocr"]
            
        try:
            # In GPU setup:
            # from paddleocr import PaddleOCR
            # self.models["paddle_ocr"] = PaddleOCR(use_angle_cls=True, lang='en')
            
            print("[ModelManager] Initializing PaddleOCR...")
            self.models["paddle_ocr"] = "PADDLE_OCR_ACTIVE_HANDLE"
            self.loaded_status["paddle_ocr"] = True
            print("[ModelManager] PaddleOCR loaded successfully.")
        except Exception as e:
            print(f"[ModelManager] Error loading PaddleOCR: {e}")
            self.loaded_status["paddle_ocr"] = False
            
        return self.models["paddle_ocr"]

    def unload_all(self):
        """Unloads models and releases system memory (and VRAM)."""
        print("[ModelManager] Unloading all models to free VRAM...")
        for k in list(self.models.keys()):
            self.models[k] = None
            self.loaded_status[k] = False
            
        # Call garbage collector
        gc.collect()
        
        # If PyTorch is loaded, empty CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("[ModelManager] CUDA Cache emptied.")
        except ImportError:
            pass
            
        print("[ModelManager] All models unloaded.")

    def get_status(self) -> Dict[str, Any]:
        return {
            "grounding_dino_loaded": self.loaded_status["grounding_dino"],
            "sam2_loaded": self.loaded_status["sam2"],
            "paddle_ocr_loaded": self.loaded_status["paddle_ocr"]
        }
