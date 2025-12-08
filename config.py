"""
⚙️ Configuration Settings for Object Detection System
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATHS = {
    "yolov3": {"cfg": BASE_DIR / "models" / "yolov3.cfg", "weights": BASE_DIR / "models" / "yolov3.weights"},
    "yolov4_tiny": {"cfg": BASE_DIR / "models" / "yolov4-tiny.cfg", "weights": BASE_DIR / "models" / "yolov4-tiny.weights"},
}

DETECTION_CONFIG = {
    "confidence_threshold": 0.5,
    "nms_threshold": 0.4,
    "input_size": (416, 416),
    "ensemble_weights": {"yolov3": 0.6, "yolov4_tiny": 0.4},
}

ANALYTICS_CONFIG = {
    "track_history_length": 50,
    "save_visualizations": True,
    "export_formats": ["json", "csv"],
}

PATHS = {
    "output_detections": BASE_DIR / "outputs" / "detections",
    "output_analytics": BASE_DIR / "outputs" / "analytics",
    "output_logs": BASE_DIR / "outputs" / "logs",
    "data_images": BASE_DIR / "data" / "images",
    "data_videos": BASE_DIR / "data" / "videos",
}

# Ensure directories exist
for path in PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

PERFORMANCE_CONFIG = {"use_gpu": False, "max_detections": 100, "processing_fps": 30}
