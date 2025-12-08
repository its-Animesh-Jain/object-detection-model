import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

class DetectionAnalytics:
    def __init__(self, classes):
        self.detection_history = []
        self.performance_metrics = {}
        self.classes = classes
    
    def log_detection(self, image_info, boxes, scores, labels, processing_time):
        """Log detection results for analysis"""
        detection_entry = {
            'timestamp': datetime.now(),
            'image_size': image_info['shape'],
            'objects_detected': len(boxes),
            'confidence_avg': np.mean(scores) if scores else 0,
            'confidence_std': np.std(scores) if scores else 0,
            'processing_time': processing_time,
            'object_types': {},
            'model_weights': getattr(self, 'current_weights', {})
        }
        
        # Count object types
        for label in labels:
            class_name = self.classes[label]
            detection_entry['object_types'][class_name] = detection_entry['object_types'].get(class_name, 0) + 1
        
        self.detection_history.append(detection_entry)
        return detection_entry
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        if not self.detection_history:
            return {"status": "No data available"}
        
        df = pd.DataFrame(self.detection_history)
        
        report = {
            # --- BUG FIX ---
            # Correctly sum the total objects detected, not just the number of entries.
            'total_detections': int(df['objects_detected'].sum()),
            'avg_objects_per_image': float(df['objects_detected'].mean()),
            'avg_confidence': float(df['confidence_avg'].mean()),
            'avg_processing_time': float(df['processing_time'].mean()),
            'most_common_objects': self.get_most_common_objects(),
            'performance_trend': self.analyze_performance_trend(df)
        }
        
        return report
    
    def get_most_common_objects(self):
        """Get most frequently detected objects"""
        object_counts = {}
        for detection in self.detection_history:
            for obj_type, count in detection['object_types'].items():
                object_counts[obj_type] = object_counts.get(obj_type, 0) + count
        
        # Return top 10 most common objects
        return dict(sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def analyze_performance_trend(self, df):
        """Analyze performance trends over time"""
        if len(df) < 2:
            return "Insufficient data for trend analysis"
        
        # Simple trend analysis
        confidence_trend = "stable"
        if len(df) >= 3:
            recent_avg = df['confidence_avg'].tail(3).mean()
            older_avg = df['confidence_avg'].head(len(df)-3).mean() if len(df) > 3 else df['confidence_avg'].iloc[0]
            if recent_avg > older_avg + 0.1:
                confidence_trend = "improving"
            elif recent_avg < older_avg - 0.1:
                confidence_trend = "declining"
        
        return f"Confidence trend: {confidence_trend}"
    
    def create_visualizations(self):
        """Create various charts and graphs for performance analysis"""
        if not self.detection_history:
            print("No data available for visualization")
            return
        
        df = pd.DataFrame(self.detection_history)
        
        plt.figure(figsize=(15, 12))
        
        plt.subplot(2, 2, 1)
        plt.plot(range(len(df)), df['objects_detected'], marker='o', linewidth=2, markersize=4)
        plt.title('Objects Detected Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Detection Sequence')
        plt.ylabel('Number of Objects')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.hist(df['confidence_avg'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Confidence Score Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Average Confidence Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.plot(range(len(df)), df['processing_time'], marker='s', color='red', linewidth=2, markersize=4)
        plt.title('Processing Time Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Detection Sequence')
        plt.ylabel('Processing Time (seconds)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        common_objects = self.get_most_common_objects()
        if common_objects:
            objects = list(common_objects.keys())[:8]
            counts = list(common_objects.values())[:8]
            colors = plt.cm.Set3(np.linspace(0, 1, len(objects)))
            bars = plt.bar(objects, counts, color=colors, edgecolor='black', alpha=0.8)
            plt.title('Most Frequently Detected Objects', fontsize=14, fontweight='bold')
            plt.xlabel('Object Types')
            plt.ylabel('Detection Count')
            plt.xticks(rotation=45, ha='right')
            
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                         str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout(pad=3.0)
        plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show(block=False) # Use block=False to prevent script from halting in some environments
        
        print("✅ Performance visualization saved as 'performance_analysis.png'")
    
    def export_detection_data(self, format='json'):
        """Export detection data for external analysis"""
        report = self.generate_performance_report()
        
        if format == 'json':
            with open('detection_report.json', 'w') as f:
                def convert_types(obj):
                    if isinstance(obj, (np.integer, np.floating)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, datetime):
                        return obj.isoformat()
                    return obj
                
                json.dump(report, f, indent=2, default=convert_types)
            print("✅ Report exported as 'detection_report.json'")
            
        elif format == 'csv':
            df = pd.DataFrame(self.detection_history)
            
            for col in ['object_types', 'model_weights']:
                if col in df.columns:
                    df[col] = df[col].apply(json.dumps)
            
            df.to_csv('detection_history.csv', index=False)
            print("✅ Data exported as 'detection_history.csv'")
        
        elif format == 'excel':
            df = pd.DataFrame(self.detection_history)
            with pd.ExcelWriter('detection_analysis.xlsx') as writer:
                df.to_excel(writer, sheet_name='Detection History', index=False)
                
                summary_data = {
                    'Metric': list(report.keys()),
                    'Value': [str(v) for v in report.values()]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            print("✅ Data exported as 'detection_analysis.xlsx'")
        
        return report
    
    def print_summary_statistics(self):
        """Print summary statistics to console"""
        if not self.detection_history:
            print("No detection data available")
            return
        
        report = self.generate_performance_report()
        
        print("\n" + "="*60)
        print("📊 DETECTION ANALYTICS SUMMARY")
        print("="*60)
        
        print(f"Total Detections: {report['total_detections']}")
        print(f"Average Objects per Image: {report['avg_objects_per_image']:.2f}")
        print(f"Average Confidence: {report['avg_confidence']:.2%}")
        print(f"Average Processing Time: {report['avg_processing_time']:.3f} seconds")
        print(f"Performance Trend: {report['performance_trend']}")
        
        print("\n🏆 MOST COMMON OBJECTS:")
        common_objects = report.get('most_common_objects', {})
        for obj, count in common_objects.items():
            print(f"  • {obj}: {count} detections")
        
        print("="*60)

if __name__ == "__main__":
    sample_classes = ['person', 'car', 'chair', 'bottle', 'book']
    analytics = DetectionAnalytics(sample_classes)
    
    sample_data = [
        {'boxes': 3, 'scores': [0.8, 0.7, 0.9], 'labels': [0, 1, 0]},
        {'boxes': 2, 'scores': [0.6, 0.85], 'labels': [2, 3]},
        {'boxes': 4, 'scores': [0.9, 0.75, 0.8, 0.65], 'labels': [0, 1, 0, 4]},
    ]
    
    for i, data in enumerate(sample_data):
        image_info = {'shape': (480, 640, 3)}
        analytics.log_detection(
            image_info, 
            [None] * data['boxes'],
            data['scores'], 
            data['labels'], 
            processing_time=0.1 + i * 0.05
        )
    
    analytics.print_summary_statistics()
    analytics.create_visualizations()
    analytics.export_detection_data('json')
    
    print("\n✅ Performance analytics module is working correctly!")