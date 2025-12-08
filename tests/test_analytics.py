import sys
from pathlib import Path
import numpy as np

# --- PATH CORRECTION ---
# Add the project's root directory to the Python path.
# This allows us to import from the 'src' package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now this import will work correctly
from src.utils.performance_analytics import DetectionAnalytics

def run_tests():
    """A simple test function for the DetectionAnalytics class."""
    print("🧪 Running Analytics Tests...")

    # 1. Test Initialization
    print("   - Testing initialization...")
    class_names = ['person', 'car', 'dog']
    try:
        analytics = DetectionAnalytics(class_names)
        print("     ✅ Initialization successful.")
    except Exception as e:
        print(f"     ❌ Initialization FAILED: {e}")
        return

    # 2. Test Logging a Detection
    print("   - Testing detection logging...")
    try:
        # Create some fake detection data
        image_info = {"shape": (480, 640, 3)}
        boxes = [[10, 10, 50, 50], [100, 100, 80, 80]]
        scores = [0.95, 0.88]
        labels = [0, 1] # person, car
        processing_time = 0.123
        analytics.log_detection(image_info, boxes, scores, labels, processing_time)
        print("     ✅ Logging successful.")
    except Exception as e:
        print(f"     ❌ Logging FAILED: {e}")
        return

    # 3. Test Report Generation (with better error reporting)
    print("   - Testing report generation...")
    report = {} # Define report here to make it available in the except block
    try:
        report = analytics.generate_performance_report()
        assert report['total_detections'] == 2
        assert report['most_common_objects']['person'] == 1
        print("     ✅ Report generation successful.")
    except Exception as e:
        # This part is new - it gives us more debug info
        print(f"     ❌ Report generation FAILED.")
        print(f"     - The test failed with this error: {e} ({type(e).__name__})")
        print("     - Here is the actual report the test received:")
        import json
        print(json.dumps(report, indent=4)) # Print the full report for debugging
        return
        
    print("\n🎉 All Analytics Tests Passed!")


if __name__ == "__main__":
    run_tests()