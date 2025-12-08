#!/usr/bin/env python3
"""
project_setup.py

Create an organized project structure for your Object_Detector repo,
move existing files into appropriate folders, and create starter files
(main.py, config.py, requirements.txt, README.md).
This version is more robust:
 - Resolves paths relative to this script file (not current working dir)
 - Searches recursively for files if they're not directly in the root
 - Uses pathlib everywhere and avoids fragile sys.path hacks
"""

import shutil
import sys
from pathlib import Path
from textwrap import dedent

# Use the script's directory as the repository root (safer than cwd)
ROOT = Path(__file__).resolve().parent


def find_file_in_repo(filename):
    """
    Find a file by name anywhere under ROOT (first match).
    Returns Path or None.
    """
    # prefer files directly under ROOT
    candidate = ROOT.joinpath(filename)
    if candidate.exists():
        return candidate
    # otherwise search recursively, ignoring virtual env folder
    for p in ROOT.rglob(filename):
        # skip inside .venv, .git, __pycache__
        if any(part in (".venv", ".git", "__pycache__") for part in p.parts):
            continue
        return p
    return None


def create_project_structure():
    """Create the organized folder structure"""
    folders = [
        "models",
        "src",
        "utils",
        "tests",
        "outputs",
        "docs",
        "data/images",
        "data/videos",
        "outputs/detections",
        "outputs/analytics",
        "outputs/logs",
    ]

    print("📁 Creating project structure...")
    for folder in folders:
        path = ROOT.joinpath(folder)
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {folder}/")
    print("🎯 Project structure created successfully!\n")


def move_files_to_organized_structure():
    """Move files to their proper locations (if they exist). Copy Image_snapshot contents."""
    file_movements = [
        # Model files
        ("yolov3.cfg", "models", "Model config"),
        ("yolov3.weights", "models", "Model weights"),
        ("yolov4-tiny.cfg", "models", "Model config"),
        ("yolov4-tiny.weights", "models", "Model weights"),
        ("coco.names", "models", "Class names"),
        # Core source files
        ("detector.py", "src", "Core detector"),
        ("model_ensemble.py", "src", "Ensemble core"),
        ("advanced_ensemble.py", "src", "Enhanced ensemble"),
        # Utility files
        ("performance_analytics.py", "utils", "Analytics"),
        ("object_tracker.py", "utils", "Tracking"),
        ("create_better_test_image.py", "utils", "Image utils"),
        # Demo and test files
        ("ensemble_demo.py", "tests", "Demo script"),
        ("demo_advanced.py", "tests", "Advanced demo"),
        ("test_analytics.py", "tests", "Testing"),
        # Dashboard
        ("dashboard.py", "src", "Web dashboard"),
    ]

    print("📦 Moving files to organized structure...")
    moved_count = 0
    for source_name, dest_folder, file_type in file_movements:
        # Try to find file in repo
        src_path = find_file_in_repo(source_name)
        dest_dir = ROOT.joinpath(dest_folder)
        dest_dir.mkdir(parents=True, exist_ok=True)
        if src_path and src_path.exists():
            dest_path = dest_dir.joinpath(src_path.name)
            try:
                # Avoid overwriting existing destination
                if dest_path.exists():
                    print(f"  ⚠️  Already exists, skipping: {dest_path}")
                else:
                    # If file is already in desired folder but path differs (same file), still skip move
                    if src_path.resolve() == dest_path.resolve():
                        print(f"  ℹ️  Already at destination: {dest_path}")
                    else:
                        shutil.move(str(src_path), str(dest_path))
                        print(f"  ✅ Moved: {src_path.relative_to(ROOT)} → {dest_folder}/ ({file_type})")
                        moved_count += 1
            except Exception as e:
                print(f"  ❌ Failed to move {source_name}: {e}")
        else:
            print(f"  ⚠️  Not found in repo: {source_name}")

    # Copy Image_snapshot folder contents (preserve originals if present)
    image_snapshot = find_file_in_repo("Image_snapshot")
    if image_snapshot and image_snapshot.exists() and image_snapshot.is_dir():
        dest_images = ROOT.joinpath("data/images")
        dest_images.mkdir(parents=True, exist_ok=True)
        copied = 0
        for item in image_snapshot.iterdir():
            src_item = item
            dest_item = dest_images.joinpath(item.name)
            try:
                if item.is_file():
                    if dest_item.exists():
                        print(f"  ⚠️  Image already exists, skipping: {dest_item.name}")
                    else:
                        shutil.copy2(src_item, dest_item)
                        copied += 1
                elif item.is_dir():
                    if not dest_item.exists():
                        shutil.copytree(src_item, dest_item)
                        copied += 1
                    else:
                        for sub in src_item.iterdir():
                            sub_dest = dest_item.joinpath(sub.name)
                            if sub.is_file() and not sub_dest.exists():
                                shutil.copy2(sub, sub_dest)
                                copied += 1
            except Exception as e:
                print(f"  ❌ Error copying {src_item}: {e}")

        print(f"  ✅ Copied Image_snapshot contents to data/images/ ({copied} items)")
    else:
        print("  ⚠️  No Image_snapshot folder found to copy (or not a directory).")

    print(f"\n🎉 Finished moving files. Total moved: {moved_count}\n")


def create_main_launcher():
    """Create main.py as the project entry point (robust imports using pathlib)"""
    main_py_content = dedent(
        r'''
        #!/usr/bin/env python3
        """
        🎯 Object Detection System - Main Launcher
        """
        import argparse
        import os
        import sys
        import traceback
        from pathlib import Path
        import importlib

        # Ensure repo root and src/utils are on sys.path (resolve relative to this file)
        REPO_ROOT = Path(__file__).resolve().parent
        SRC_DIR = REPO_ROOT.joinpath("src")
        UTILS_DIR = REPO_ROOT.joinpath("utils")

        for p in (str(REPO_ROOT), str(SRC_DIR), str(UTILS_DIR)):
            if p not in sys.path:
                sys.path.insert(0, p)

        def setup_environment():
            """Setup project environment and check required models"""
            print("🚀 Initializing Object Detection System...")
            # Use config if available to discover model paths
            try:
                import config
                model_candidates = []
                for m in getattr(config, "MODEL_PATHS", {}).values():
                    # m may be a dict with 'cfg' and 'weights' entries
                    if isinstance(m, dict):
                        model_candidates.extend([m.get("cfg"), m.get("weights")])
                    else:
                        model_candidates.append(m)
                # fallback models to check if config empty
                if not model_candidates:
                    model_candidates = [
                        os.path.join("models", "yolov3.cfg"),
                        os.path.join("models", "yolov3.weights"),
                        os.path.join("models", "yolov4-tiny.cfg"),
                        os.path.join("models", "yolov4-tiny.weights"),
                        os.path.join("models", "coco.names"),
                    ]
            except Exception:
                model_candidates = [
                    os.path.join("models", "yolov3.cfg"),
                    os.path.join("models", "yolov3.weights"),
                    os.path.join("models", "yolov4-tiny.cfg"),
                    os.path.join("models", "yolov4-tiny.weights"),
                    os.path.join("models", "coco.names"),
                ]

            missing = []
            for m in model_candidates:
                if m and not Path(m).exists():
                    # try relative to repo root
                    if not REPO_ROOT.joinpath(m).exists():
                        missing.append(m)
            if missing:
                print("❌ Missing model files (relative paths):")
                for m in missing:
                    print("   -", m)
                return False
            print("✅ All required model files found (or paths configured).")
            return True

        def run_demo_mode(args):
            """Run the demo (tries multiple locations for the demo entry)"""
            try:
                # Prefer src.ensemble_demo then tests.ensemble_demo
                demo_main = None
                try:
                    demo_mod = importlib.import_module("ensemble_demo")
                    demo_main = getattr(demo_mod, "main", None)
                except Exception:
                    # try package style import
                    try:
                        demo_mod = importlib.import_module("src.ensemble_demo")
                        demo_main = getattr(demo_mod, "main", None)
                    except Exception:
                        pass

                if not demo_main:
                    # fallback to tests
                    try:
                        demo_mod = importlib.import_module("tests.ensemble_demo")
                        demo_main = getattr(demo_mod, "main", None)
                    except Exception:
                        pass

                if not demo_main:
                    raise ImportError("Could not locate an 'ensemble_demo' module with a main() function.")

                demo_kw = {}
                if args.image:
                    demo_kw["image"] = args.image
                if args.webcam:
                    demo_kw["webcam"] = True

                demo_main(**demo_kw)
            except ImportError as e:
                print("❌ Error importing demo:", e)
                print("💡 Make sure ensemble_demo.py is in src/ or tests/ and exposes a main(args...) function")
            except Exception as e:
                print("❌ Error running demo:", e)
                traceback.print_exc()

        def run_advanced_mode():
            try:
                adv = importlib.import_module("demo_advanced")
                advanced_main = getattr(adv, "main", None)
                if advanced_main:
                    advanced_main()
                else:
                    print("❌ demo_advanced.main() not found.")
            except Exception as e:
                print("❌ Error running advanced mode:", e)

        def run_dashboard_mode():
            try:
                print("🌐 Starting Streamlit Dashboard...")
                # run streamlit from the repo's src/dashboard.py
                dash = REPO_ROOT.joinpath("src", "dashboard.py")
                if dash.exists():
                    os.system(f"streamlit run \"{str(dash)}\"")
                else:
                    print("❌ Dashboard script not found at src/dashboard.py")
            except Exception as e:
                print("❌ Error starting dashboard:", e)

        def main():
            parser = argparse.ArgumentParser(description="Object Detection System Launcher")
            parser.add_argument("--mode", choices=["demo", "dashboard", "advanced", "webcam"], default="demo")
            parser.add_argument("--image", help="Path to input image file")
            parser.add_argument("--webcam", action="store_true", help="Use webcam for real-time detection")
            parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold (0-1)")
            args = parser.parse_args()

            print("=" * 60)
            print("🎯 OBJECT DETECTION SYSTEM")
            print("=" * 60)

            if not setup_environment():
                print("Please add missing model files to the models/ directory and try again.")
                return

            try:
                if args.mode == "demo":
                    run_demo_mode(args)
                elif args.mode == "dashboard":
                    run_dashboard_mode()
                elif args.mode == "advanced":
                    run_advanced_mode()
                elif args.mode == "webcam":
                    args.webcam = True
                    run_demo_mode(args)
            except KeyboardInterrupt:
                print("\\n👋 Session ended by user")
            except Exception as e:
                print(f"❌ Error in {args.mode} mode: {e}")
                traceback.print_exc()

        if __name__ == "__main__":
            main()
        '''
    )

    target = ROOT.joinpath("main.py")
    target.write_text(main_py_content, encoding="utf-8")
    print("✅ Created main.py - Project entry point")


def create_config_file():
    """Create configuration file (pathlib-based)"""
    config_py_content = dedent(
        '''
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

        # Make sure directories exist
        for path in PATHS.values():
            path.mkdir(parents=True, exist_ok=True)

        PERFORMANCE_CONFIG = {"use_gpu": False, "max_detections": 100, "processing_fps": 30}
        '''
    )

    (ROOT.joinpath("config.py")).write_text(config_py_content, encoding="utf-8")
    print("✅ Created config.py - Configuration settings")


def create_requirements_file():
    """Create requirements.txt"""
    requirements_content = dedent(
        """
        # Object Detection System Dependencies

        # Core computer vision
        opencv-python>=4.5.0
        numpy>=1.21.0

        # Data analysis and visualization
        pandas>=1.3.0
        matplotlib>=3.5.0

        # Web dashboard
        streamlit>=1.12.0

        # Utilities
        Pillow>=8.3.0
        scipy>=1.7.0

        # Development tools (optional)
        black>=21.0.0
        flake8>=3.9.0
        pytest>=6.2.0
        """
    )

    (ROOT.joinpath("requirements.txt")).write_text(requirements_content.strip() + "\n", encoding="utf-8")
    print("✅ Created requirements.txt - Dependencies list")


def create_readme():
    """Create README.md with example project structure and usage"""
    readme_content = dedent(
        """
        # 🎯 Object Detection System
        ...
        """
    )

    (ROOT.joinpath("README.md")).write_text(readme_content.strip() + "\n", encoding="utf-8")
    print("✅ Created README.md")


def main():
    print("\n🔧 Project Setup starting...\n")
    create_project_structure()
    move_files_to_organized_structure()
    create_main_launcher()
    create_config_file()
    create_requirements_file()
    create_readme()
    print("\n✅ All done! Check the folders and files created/updated above.")
    print("Tip: review main.py and src/ files to wire actual demo/advanced functions.\n")


if __name__ == "__main__":
    main()
