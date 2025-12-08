import os

def fix_advanced_ensemble():
    """Fix src/advanced_ensemble.py imports correctly"""
    
    # Read the current file
    with open('src/advanced_ensemble.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the problematic import section
    new_imports = '''import cv2
import numpy as np
import time
import json
import pandas as pd
from collections import defaultdict, deque
import sys
import os

# Add parent directory to path for cross-directory imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import from same directory (src/)
from model_ensemble import ModelEnsemble

# Import from utils directory
from utils.performance_analytics import DetectionAnalytics
from utils.object_tracker import ObjectTracker

'''
    
    # Find where the actual class content starts (after the broken imports)
    # Look for the class definition
    lines = content.split('\n')
    class_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('class AdvancedEnsemble'):
            class_start = i
            break
    
    if class_start is None:
        # If we can't find the class, the file might be empty or different
        print("Error: Could not find AdvancedEnsemble class in the file")
        return
    
    # Get the class content (everything from class definition onward)
    class_content = '\n'.join(lines[class_start:])
    
    # Write the fixed file
    new_content = new_imports + class_content
    with open('src/advanced_ensemble.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("Fixed src/advanced_ensemble.py imports")

def fix_demo_advanced():
    """Fix tests/demo_advanced.py to use correct import path"""
    content = '''import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2
import numpy as np
from advanced_ensemble import AdvancedEnsemble

def main():
    print("Advanced Ensemble Demo with Analytics")
    print("=" * 50)
    
    # Model configurations
    configs = {
        'yolov3': ('../models/yolov3.cfg', '../models/yolov3.weights'),
        'yolov4-tiny': ('../models/yolov4-tiny.cfg', '../models/yolov4-tiny.weights')
    }
    
    # Initialize advanced ensemble
    print("Loading advanced ensemble with analytics...")
    ensemble = AdvancedEnsemble(configs)
    print("Advanced ensemble loaded!")
    
    # Create test image
    print("Creating test image...")
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    cv2.rectangle(image, (100, 100), (200, 200), (0, 0, 255), -1)
    cv2.rectangle(image, (300, 150), (400, 250), (0, 255, 0), -1)
    cv2.circle(image, (500, 300), 50, (255, 0, 0), -1)
    
    # Create outputs directory if it doesn't exist
    os.makedirs('../outputs', exist_ok=True)
    cv2.imwrite('../outputs/test_image.jpg', image)
    
    # Run detection
    print("Running detection with analytics...")
    boxes, scores, labels, tracks = ensemble.smart_ensemble_detect(image)
    print(f"Found {len(boxes)} objects")
    
    # Show analytics
    ensemble.show_analytics()
    print("Demo completed!")

if __name__ == "__main__":
    main()
'''
    
    with open('tests/demo_advanced.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Fixed tests/demo_advanced.py")

def main():
    print("Fixing import paths...")
    fix_advanced_ensemble()
    fix_demo_advanced()
    print("All imports fixed!")

if __name__ == "__main__":
    main()