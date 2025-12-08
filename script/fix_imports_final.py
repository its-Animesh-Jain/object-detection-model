import os
import sys

def fix_demo_advanced():
    """Fix tests/demo_advanced.py"""
    content = '''import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
from src.advanced_ensemble import AdvancedEnsemble

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

def fix_advanced_ensemble():
    """Fix src/advanced_ensemble.py imports"""
    # Read the original file
    with open('src/advanced_ensemble.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the import section
    new_imports = '''import cv2
import numpy as np
import time
import json
import pandas as pd
from collections import defaultdict, deque
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from model_ensemble import ModelEnsemble
from performance_analytics import DetectionAnalytics
from object_tracker import ObjectTracker

'''
    
    # Find where the class definition starts (after imports)
    lines = content.split('\n')
    class_start = None
    for i, line in enumerate(lines):
        if line.startswith('class AdvancedEnsemble'):
            class_start = i
            break
    
    if class_start is not None:
        # Keep everything from the class definition onward
        class_content = '\n'.join(lines[class_start:])
        new_content = new_imports + class_content
        
        with open('src/advanced_ensemble.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("Fixed src/advanced_ensemble.py imports")
    else:
        print("Could not find class definition in advanced_ensemble.py")

def fix_dashboard():
    """Fix src/dashboard.py imports"""
    with open('src/dashboard.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the import
    content = content.replace('from src.advanced_ensemble import AdvancedEnsemble', '''import sys
import os
sys.path.append(os.path.dirname(__file__))
from advanced_ensemble import AdvancedEnsemble''')
    
    with open('src/dashboard.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Fixed src/dashboard.py imports")

def main():
    print("Fixing all import issues...")
    fix_demo_advanced()
    fix_advanced_ensemble() 
    fix_dashboard()
    print("All imports fixed!")

if __name__ == "__main__":
    main()