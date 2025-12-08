import os

def fix_advanced_ensemble_imports():
    """Fix the utils imports in advanced_ensemble.py"""
    
    # Read the current file
    with open('src/advanced_ensemble.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the utils imports with correct relative imports
    new_imports = '''import cv2
import numpy as np
import time
import json
import pandas as pd
from collections import defaultdict, deque
import sys
import os

# Import from same directory (src/)
from model_ensemble import ModelEnsemble

# Import from utils using relative path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.performance_analytics import DetectionAnalytics
from utils.object_tracker import ObjectTracker

'''
    
    # Find where the class content starts
    lines = content.split('\n')
    class_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('class AdvancedEnsemble'):
            class_start = i
            break
    
    if class_start is None:
        print("Error: Could not find AdvancedEnsemble class")
        return
    
    # Get the class content
    class_content = '\n'.join(lines[class_start:])
    
    # Write the fixed file
    new_content = new_imports + class_content
    with open('src/advanced_ensemble.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("Fixed utils imports in src/advanced_ensemble.py")

def main():
    print("Fixing utils imports...")
    fix_advanced_ensemble_imports()
    print("Utils imports fixed!")

if __name__ == "__main__":
    main()