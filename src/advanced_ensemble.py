import os
import sys
import time
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Ensure project root is on sys.path so "src.*" and "utils.*" imports work reliably
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project root (one above src/)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import from src / utils packages
from src.model_ensemble import ModelEnsemble
from src.utils.performance_analytics import DetectionAnalytics
from src.utils.object_tracker import ObjectTracker


class AdvancedEnsemble(ModelEnsemble):
    # --- CHANGED: Added 'classes_path' to accept the file path ---
    def __init__(self, configs, classes_path):
        # --- CHANGED: Pass 'classes_path' to the parent class constructor ---
        super().__init__(configs, classes_path)
        
        # Now self.classes is correctly set by the parent class
        self.analytics = DetectionAnalytics(self.classes)
        self.tracker = ObjectTracker()

    def smart_ensemble_detect(self, image, confidence_threshold=0.5):
        """Enhanced detection with analytics and tracking"""
        start_time = time.time()

        # Get ensemble results (boxes, scores, labels)
        boxes, scores, labels = self.ensemble_detect(image, confidence_threshold)

        # Update analytics
        image_info = {"shape": image.shape}
        processing_time = time.time() - start_time
        try:
            self.analytics.log_detection(image_info, boxes, scores, labels, processing_time)
        except Exception:
            # If the analytics API differs, don't crash here — fail gracefully
            print("Warning: analytics.log_detection failed (check signature).")

        # Update object tracking
        try:
            # pass lists; ObjectTracker implementation will decide the expected format
            track_summary = self.tracker.update_tracks(boxes, scores, labels)
        except TypeError:
            # older implementation may expect a single tuple
            track_summary = self.tracker.update_tracks((boxes, scores, labels))
        except Exception:
            print("Warning: tracker.update_tracks failed (check implementation).")
            track_summary = None

        return boxes, scores, labels, track_summary

    def adaptive_confidence_threshold(self):
        """Dynamically adjust confidence threshold based on scene complexity"""
        hist = getattr(self.analytics, "detection_history", None)
        if not hist:
            return 0.5

        recent = hist[-5:]
        if not recent:
            return 0.5

        # Defensive: ensure key exists
        confidences = [d.get("confidence_avg", None) for d in recent]
        confidences = [c for c in confidences if c is not None]
        if not confidences:
            return 0.5

        avg_confidence = float(np.mean(confidences))

        # Simple adaptive logic
        if avg_confidence > 0.7:
            return 0.4
        elif avg_confidence < 0.3:
            return 0.6
        else:
            return 0.5

    def export_detection_data(self, out_format="json"):
        """Export detection data for external analysis"""
        # ensure outputs dir exists
        outputs_dir = PROJECT_ROOT / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # generate report (assuming this returns a dict)
        report = self.analytics.generate_performance_report()

        if out_format == "json":
            target = outputs_dir / "detection_report.json"
            with target.open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Report exported as {target}")
        elif out_format == "csv":
            df = pd.DataFrame(self.analytics.detection_history)
            target = outputs_dir / "detection_history.csv"
            df.to_csv(target, index=False)
            print(f"Data exported as {target}")
        else:
            raise ValueError(f"Unsupported export format: {out_format}")

        return report

    def show_analytics(self):
        """Display analytics and visualizations"""
        report = self.export_detection_data()
        print("\nPERFORMANCE REPORT:")
        print("=" * 40)
        for key, value in report.items():
            print(f"{key}: {value}")

        # Create visualizations (guarded)
        try:
            self.analytics.create_visualizations()
        except Exception:
            print("Warning: analytics.create_visualizations failed (check implementation).")