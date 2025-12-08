python -c "
content = '''import cv2
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

class AdvancedEnsemble(ModelEnsemble):
    def __init__(self, configs):
        super().__init__(configs)
        self.analytics = DetectionAnalytics(self.classes)
        self.tracker = ObjectTracker()
    
    def smart_ensemble_detect(self, image, confidence_threshold=0.5):
        '''Enhanced detection with analytics and tracking'''
        start_time = time.time()
        
        # Get ensemble results
        boxes, scores, labels = self.ensemble_detect(image, confidence_threshold)
        
        # Update analytics
        image_info = {'shape': image.shape}
        processing_time = time.time() - start_time
        self.analytics.log_detection(image_info, boxes, scores, labels, processing_time)
        
        # Update object tracking
        track_summary = self.tracker.update_tracks((boxes, scores, labels))
        
        return boxes, scores, labels, track_summary
    
    def adaptive_confidence_threshold(self):
        '''Dynamically adjust confidence threshold based on scene complexity'''
        if not self.analytics.detection_history:
            return 0.5
        
        recent_detections = self.analytics.detection_history[-5:]  # Last 5 detections
        if not recent_detections:
            return 0.5
            
        avg_confidence = np.mean([d['confidence_avg'] for d in recent_detections])
        
        # Simple adaptive logic
        if avg_confidence > 0.7:  # High confidence scenes
            return 0.4  # Be more sensitive
        elif avg_confidence < 0.3:  # Low confidence scenes  
            return 0.6  # Be more strict
        else:
            return 0.5  # Default
    
    def export_detection_data(self, format='json'):
        '''Export detection data for external analysis'''
        report = self.analytics.generate_performance_report()
        
        if format == 'json':
            with open('../outputs/detection_report.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print('Report exported as outputs/detection_report.json')
        elif format == 'csv':
            df = pd.DataFrame(self.analytics.detection_history)
            df.to_csv('../outputs/detection_history.csv', index=False)
            print('Data exported as outputs/detection_history.csv')
        
        return report
    
    def show_analytics(self):
        '''Display analytics and visualizations'''
        report = self.export_detection_data()
        print('\\nPERFORMANCE REPORT:')
        print('=' * 40)
        for key, value in report.items():
            print(f'{key}: {value}')
        
        # Create visualizations
        self.analytics.create_visualizations()

'''

with open('src/advanced_ensemble.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Restored complete advanced_ensemble.py')
"