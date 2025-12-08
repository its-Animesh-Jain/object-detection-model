import cv2
import argparse
from pathlib import Path
from src.model_ensemble import ModelEnsemble


def run_demo(image_path=None, video_path=None, use_webcam=False, **kwargs):
    """Runs YOLO ensemble detection on image, video, or webcam input."""

    # This goes up two levels (from tests/ensemble_demo.py to Object_Detector/) and then into models/
    base_dir = Path(__file__).resolve().parents[1] / "models"

    configs = {
        'yolov3': (
            str(base_dir / 'yolov3.cfg'),
            str(base_dir / 'yolov3.weights')
        ),
        'yolov4-tiny': (
            str(base_dir / 'yolov4-tiny.cfg'),
            str(base_dir / 'yolov4-tiny.weights')
        )
    }
    
    # --- ADDED: Define the path to the coco.names file ---
    classes_path = base_dir / 'coco.names'

    for model, (cfg, weights) in configs.items():
        if not Path(cfg).exists() or not Path(weights).exists():
            print(f"❌ Missing files for {model}:")
            print(f"   - {cfg if not Path(cfg).exists() else '✅ Found cfg'}")
            print(f"   - {weights if not Path(weights).exists() else '✅ Found weights'}")
            return

    print("🧠 Initializing model ensemble...")
    # --- CHANGED: Pass the classes_path to the constructor ---
    ensemble = ModelEnsemble(configs, classes_path=str(classes_path))
    print("✅ Ensemble ready.")

    if image_path:
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Could not load image {image_path}")
            return

        boxes, scores, labels = ensemble.ensemble_detect(image)
        result_image = ensemble.visualize_detections(image, boxes, scores, labels)

        output_path = Path("outputs/detections/ensemble_result.jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result_image)
        print(f"✅ Results saved to {output_path}")

        cv2.imshow('Ensemble Detection', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif video_path or use_webcam:
        if use_webcam:
            source = 0
            print("📹 Starting webcam...")
        else:
            source = video_path
            print(f"📹 Processing video: {source}")
            
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video source.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            boxes, scores, labels = ensemble.ensemble_detect(frame)
            result_frame = ensemble.visualize_detections(frame, boxes, scores, labels)

            cv2.imshow('Ensemble Detection', result_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Processing finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run ensemble object detection demo.")
    parser.add_argument("--image", type=str, help="Path to the input image.")
    parser.add_argument("--video", type=str, help="Path to the input video.")
    parser.add_argument("--webcam", action="store_true", help="Use webcam as input.")
    args = parser.parse_args()

    if not any([args.image, args.video, args.webcam]):
        print("💡 No input specified. Use --image, --video, or --webcam.")
    else:
        run_demo(image_path=args.image, video_path=args.video, use_webcam=args.webcam)