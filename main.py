#!/usr/bin/env python3
"""
🎯 Object Detection System - Main Launcher
"""
import subprocess
import argparse
import os
import sys
import traceback
from pathlib import Path
import importlib
import inspect
from types import ModuleType

# Add the project's root directory to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPO_ROOT = PROJECT_ROOT # Use PROJECT_ROOT consistently

def _safe_import_module(module_name: str) -> ModuleType | None:
    """Safely import a module."""
    old_argv = sys.argv.copy()
    try:
        sys.argv[:] = [old_argv[0]]
        return importlib.import_module(module_name)
    except Exception:
        return None
    finally:
        sys.argv[:] = old_argv

def setup_environment():
    """Check if model files exist."""
    print("🚀 Initializing Object Detection System...")
    # Simplified check focusing on essential files in the 'models' directory
    models_dir = REPO_ROOT / "models"
    required_files = [
        models_dir / "yolov3.cfg",
        models_dir / "yolov3.weights",
        models_dir / "yolov4-tiny.cfg",
        models_dir / "yolov4-tiny.weights",
        models_dir / "coco.names",
    ]
    missing = [f for f in required_files if not f.exists()]
    if missing:
        print("❌ Missing required model files:")
        for f in missing:
            print(f"   - {f.relative_to(REPO_ROOT)}")
        return False
    print("✅ All required model files found.")
    return True

def _locate_callable(module_names, attr="main"):
    """Find a callable function within specified modules."""
    for mod_name in module_names:
        mod = _safe_import_module(mod_name)
        if mod:
            target = getattr(mod, attr, None)
            if callable(target):
                return target
    return None

def run_demo_mode(image_path=None, use_webcam=False):
    """Runs the demo by directly calling the function."""
    try:
        demo_main = _locate_callable(["tests.ensemble_demo"], "run_demo")
        if demo_main is None:
            print("❌ Could not find 'run_demo' in 'tests/ensemble_demo.py'.")
            return

        if image_path:
            print(f"📷 Processing image: {image_path}")
            demo_main(image_path=image_path)
        elif use_webcam:
            print("📹 Firing up the webcam...")
            demo_main(use_webcam=True)
        else:
             print("💡 No input (image/webcam) specified for demo mode.")

    except Exception as e:
        print(f"❌ Error running demo: {e}")
        traceback.print_exc()

def run_advanced_mode():
    """Runs the advanced demo."""
    try:
        advanced_main = _locate_callable(["tests.demo_advanced"], "main")
        if advanced_main is None:
            print("❌ 'demo_advanced.py' not found or has no 'main' function in 'tests/'.")
            return
        # Call the function (assuming it takes no arguments or handles None)
        try:
             advanced_main()
        except TypeError:
             print("Note: Trying to call advanced demo main() without arguments.")
             advanced_main(None) # Attempt fallback for functions expecting args
    except Exception as e:
        print(f"❌ Error running advanced mode: {e}")
        traceback.print_exc()

def run_dashboard_mode():
    """Starts the Streamlit dashboard using the robust subprocess module."""
    try:
        print("🌐 Starting Streamlit Dashboard...")
        dash_script = REPO_ROOT / "src" / "dashboard.py"

        if not dash_script.exists():
            print("❌ Dashboard script not found at src/dashboard.py")
            return

        # Create the command as a list of arguments
        command = [
            sys.executable,  # The full path to your venv's python.exe
            "-m",            # The '-m' argument
            "streamlit",     # The module to run
            "run",           # The command to streamlit
            str(dash_script) # The path to the script to run
        ]

        # Run the command
        subprocess.run(command)

    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")

# --- NEW FUNCTION ADDED ---
def run_detector_mode():
    """Runs the standalone security monitor."""
    try:
        print("📹 Starting Live Detection & Alert System...")
        detector_script = REPO_ROOT / "src" / "detector.py"
        
        if not detector_script.exists():
            print("❌ Detector script not found at src/detector.py")
            return

        # Use subprocess.run to execute the script in its own process
        command = [
            sys.executable,      # The full path to your venv's python.exe
            str(detector_script) # The script to run
        ]
        
        subprocess.run(command)
        
    except Exception as e:
        print(f"❌ Error starting detector: {e}")

# --- INTERACTIVE MENU UPDATED ---
def interactive_menu():
    """Displays an interactive menu for the user to choose a mode."""
    print("\n" + "=" * 60)
    print("🤖 OBJECT DETECTION SYSTEM - INTERACTIVE MENU")
    print("=" * 60)
    print("Please choose an option to run:")
    print("  1. Run Demo (Image/Webcam)")
    print("  2. Launch Web Dashboard")
    print("  3. Run Advanced Demo")
    print("  4. Run Live Detection & Alert System") # <-- NEW OPTION
    print("  0. Exit")
    print("-" * 60)

    while True:
        choice = input("Enter the number of your choice: ").strip()
        if choice == '1':
            print("\n--- Demo Mode ---")
            while True:
                sub_choice = input("Choose input: (1) Webcam, (2) Image File, (0) Back: ").strip()
                if sub_choice == '1':
                    run_demo_mode(use_webcam=True)
                    return # Exit after running
                elif sub_choice == '2':
                    img_path_str = input("Enter the path to your image file (e.g., data/images/sample.jpg): ").strip()
                    img_path = Path(img_path_str)
                    if img_path.exists() and img_path.is_file():
                         run_demo_mode(image_path=str(img_path))
                         return # Exit after running
                    else:
                         print(f"❌ Error: File not found or is not a file: '{img_path_str}'")
                elif sub_choice == '0':
                    break # Go back to main menu
                else:
                    print("Invalid choice. Please enter 1, 2, or 0.")
            break # Re-show main menu after going back
        elif choice == '2':
            run_dashboard_mode()
            return # Exit after launching
        elif choice == '3':
            run_advanced_mode()
            return # Exit after running
        elif choice == '4': # <-- NEW BLOCK
            run_detector_mode()
            return # Exit after running
        elif choice == '0':
            print("👋 Exiting.")
            return # Exit program
        else:
            print("Invalid choice. Please enter a number from 0 to 4.") # <-- UPDATED RANGE

def main():
    parser = argparse.ArgumentParser(description="Object Detection System Launcher", add_help=False)
    # Make arguments optional for interactive mode
    # --- UPDATED CHOICES ---
    parser.add_argument("--mode", choices=["demo", "dashboard", "advanced", "webcam", "detector"], default=None, help="Specify the mode to run directly.")
    parser.add_argument("--image", help="Path to input image file (used with --mode demo or implicitly).")
    parser.add_argument("--webcam", action="store_true", help="Use webcam (shortcut for --mode demo with webcam).")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold (0-1). Note: Not all modes use this directly.")
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit.")

    args = parser.parse_args()

    print("=" * 60)
    print("🎯 OBJECT DETECTION SYSTEM")
    print("=" * 60)

    if not setup_environment():
        print("Please fix the missing model files and try again.")
        return

    # --- NEW: Check if arguments were provided or run interactive menu ---
    run_interactively = args.mode is None and not args.webcam and args.image is None

    if run_interactively:
        interactive_menu()
    else:
        # --- Run based on command-line arguments ---
        mode_to_run = args.mode
        if args.webcam:
            mode_to_run = "demo" # --webcam implies demo mode

        try:
            if mode_to_run == "demo":
                # Pass image path or webcam flag based on args
                run_demo_mode(image_path=args.image, use_webcam=args.webcam)
            elif mode_to_run == "dashboard":
                run_dashboard_mode()
            elif mode_to_run == "advanced":
                run_advanced_mode()
            elif mode_to_run == "detector": # <-- NEW BLOCK
                run_detector_mode()
            else:
                # Should not happen if arguments are parsed correctly, but good to handle
                print(f"Error: Unknown mode '{mode_to_run}' specified via arguments.")
                interactive_menu() # Fallback to interactive menu

        except KeyboardInterrupt:
            print("\n👋 Session ended by user")
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()