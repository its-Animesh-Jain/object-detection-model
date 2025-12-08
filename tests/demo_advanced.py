import sys
import os
from pathlib import Path # Import pathlib

# --- PATH CORRECTION ---
# Find the project root directory (Object_Detector)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Construct the path to the models directory
MODELS_DIR = PROJECT_ROOT / "models"
# Construct the path to the classes file
CLASSES_PATH = MODELS_DIR / "coco.names"

# Add the project root to sys.path if it's not already there
# This ensures imports from 'src' work when running this script directly
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
# Now this import works correctly because PROJECT_ROOT is in sys.path
from src.advanced_ensemble import AdvancedEnsemble

def main():
    print("Advanced Ensemble Demo with Analytics")
    print("=" * 50)
    
    # Model configurations using absolute paths for robustness
    configs = {
        'yolov3': (str(MODELS_DIR / 'yolov3.cfg'), str(MODELS_DIR / 'yolov3.weights')),
        'yolov4-tiny': (str(MODELS_DIR / 'yolov4-tiny.cfg'), str(MODELS_DIR / 'yolov4-tiny.weights'))
    }
    
    # Initialize advanced ensemble
    print("Loading advanced ensemble with analytics...")
    # --- FIX: Pass the classes_path to the constructor ---
    try:
        ensemble = AdvancedEnsemble(configs, classes_path=str(CLASSES_PATH))
        print("Advanced ensemble loaded!")
    except FileNotFoundError:
        print(f"❌ Error: Could not find classes file at {CLASSES_PATH}")
        return
    except Exception as e:
        print(f"❌ Error initializing AdvancedEnsemble: {e}")
        return

    # Create test image (using relative path from project root for output)
    print("Creating test image...")
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    cv2.rectangle(image, (100, 100), (200, 200), (0, 0, 255), -1)
    cv2.rectangle(image, (300, 150), (400, 250), (0, 255, 0), -1)
    cv2.circle(image, (500, 300), 50, (255, 0, 0), -1)
    
    # Create outputs directory if it doesn't exist (relative to project root)
    output_dir = PROJECT_ROOT / 'outputs'
    output_dir.mkdir(exist_ok=True)
    test_image_path = output_dir / 'test_image_advanced_demo.jpg'
    cv2.imwrite(str(test_image_path), image)
    print(f"Test image saved to: {test_image_path.relative_to(PROJECT_ROOT)}")
    
    # Run detection
    print("Running detection with analytics...")
    try:
        boxes, scores, labels, tracks = ensemble.smart_ensemble_detect(image)
        print(f"Found {len(boxes)} objects")
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        return

    # Show analytics
    try:
        ensemble.show_analytics()
    except Exception as e:
        print(f"❌ Error showing analytics: {e}")

    print("Demo completed!")

if __name__ == "__main__":
    main()