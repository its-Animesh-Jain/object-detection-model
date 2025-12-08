import os
import sys

def fix_imports_simple():
    """Fix imports without Unicode issues"""
    print("Fixing import issues...")
    
    # Files to fix
    files_to_fix = [
        'src/advanced_ensemble.py',
        'src/dashboard.py', 
        'tests/demo_advanced.py',
        'tests/test_analytics.py'
    ]
    
    # Import fixes
    fixes = [
        ('from model_ensemble', 'from src.model_ensemble'),
        ('import model_ensemble', 'from src import model_ensemble'),
        ('from advanced_ensemble', 'from src.advanced_ensemble'),
        ('import advanced_ensemble', 'from src import advanced_ensemble'),
        ('from performance_analytics', 'from utils.performance_analytics'),
        ('import performance_analytics', 'from utils import performance_analytics'),
        ('from object_tracker', 'from utils.object_tracker'),
        ('import object_tracker', 'from utils import object_tracker'),
    ]
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original = content
                for old, new in fixes:
                    content = content.replace(old, new)
                
                if content != original:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"Fixed: {file_path}")
            except Exception as e:
                print(f"Error with {file_path}: {e}")

def create_simple_main():
    """Create main.py without Unicode characters"""
    main_content = '''#!/usr/bin/env python3
"""
Object Detection System - Main Launcher
"""

import os
import sys
import subprocess
import argparse

def setup_environment():
    """Setup the environment"""
    print("Object Detection System")
    print("=" * 50)
    
    required_files = [
        'models/yolov3.cfg',
        'models/yolov3.weights', 
        'models/yolov4-tiny.cfg',
        'models/yolov4-tiny.weights',
        'models/coco.names'
    ]
    
    print("Checking for model files...")
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Missing model files:")
        for f in missing_files:
            print(f"   - {f}")
        return False
    
    print("All model files found")
    return True

def run_demo(image_path=None, webcam=False):
    """Run the ensemble demo"""
    try:
        cmd = [sys.executable, 'src/ensemble_demo.py']
        
        if image_path:
            if os.path.exists(image_path):
                cmd.extend(['--image', image_path])
                print(f"Processing: {image_path}")
            else:
                data_path = os.path.join('data', 'images', image_path)
                if os.path.exists(data_path):
                    cmd.extend(['--image', data_path])
                    print(f"Processing: {data_path}")
                else:
                    print(f"Image not found: {image_path}")
                    return
        elif webcam:
            cmd.append('--webcam')
            print("Starting webcam...")
        else:
            print("Starting interactive demo...")
        
        subprocess.run(cmd)
        
    except Exception as e:
        print(f"Error running demo: {e}")

def run_advanced():
    """Run advanced analytics"""
    try:
        print("Starting Advanced Analytics...")
        subprocess.run([sys.executable, 'tests/demo_advanced.py'])
    except Exception as e:
        print(f"Error running advanced demo: {e}")

def run_dashboard():
    """Run web dashboard"""
    try:
        print("Starting Streamlit Dashboard...")
        print("The dashboard will open in your web browser")
        print("Press Ctrl+C to stop the dashboard")
        subprocess.run(['streamlit', 'run', 'src/dashboard.py'])
    except FileNotFoundError:
        print("Streamlit not found. Install with: pip install streamlit")
    except KeyboardInterrupt:
        print("Dashboard stopped by user")
    except Exception as e:
        print(f"Error running dashboard: {e}")

def list_images():
    """List available images"""
    image_dirs = ['data/images/', 'Image_snapshot/', './']
    
    images_found = []
    for img_dir in image_dirs:
        if os.path.exists(img_dir):
            images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if images:
                print(f"Images in {img_dir}:")
                for img in images[:10]:
                    full_path = os.path.join(img_dir, img)
                    print(f"   - {img}")
                    images_found.append(full_path)
                if len(images) > 10:
                    print(f"   ... and {len(images) - 10} more")
    
    if not images_found:
        print("No images found in common directories")
    
    return images_found

def interactive_mode():
    """Interactive menu"""
    while True:
        print("INTERACTIVE MENU")
        print("=" * 40)
        print("1. Detect objects in image")
        print("2. Webcam detection")
        print("3. Advanced analytics")
        print("4. Web dashboard")
        print("5. List available images")
        print("6. Exit")
        
        try:
            choice = input("Select option (1-6): ").strip()
            
            if choice == '1':
                images = list_images()
                if images:
                    print("Enter image filename or full path:")
                    img_input = input("Image: ").strip()
                    run_demo(image_path=img_input)
            elif choice == '2':
                run_demo(webcam=True)
            elif choice == '3':
                run_advanced()
            elif choice == '4':
                run_dashboard()
            elif choice == '5':
                list_images()
            elif choice == '6':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please select 1-6.")
                
        except KeyboardInterrupt:
            print("Session ended by user")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Object Detection System')
    parser.add_argument('--mode', choices=['demo', 'webcam', 'advanced', 'dashboard', 'interactive'],
                       default='interactive', help='Run mode')
    parser.add_argument('--image', help='Image file path')
    
    args = parser.parse_args()
    
    if not setup_environment():
        return
    
    if args.mode == 'demo':
        run_demo(image_path=args.image)
    elif args.mode == 'webcam':
        run_demo(webcam=True)
    elif args.mode == 'advanced':
        run_advanced()
    elif args.mode == 'dashboard':
        run_dashboard()
    elif args.mode == 'interactive':
        interactive_mode()

if __name__ == "__main__":
    main()
'''
    
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(main_content)
    
    print("Created main.py")

if __name__ == "__main__":
    fix_imports_simple()
    create_simple_main()
    print("All fixes applied!")
    print("Now run: python main.py --mode interactive")