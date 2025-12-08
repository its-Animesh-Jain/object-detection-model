import os
import sys

def fix_import_issues():
    """Fix all import issues in the project"""
    
    print("🔧 Fixing import issues...")
    
    # First, let's check what files exist
    python_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"📁 Found {len(python_files)} Python files")
    
    # Common import patterns to fix
    import_fixes = [
        # From model_ensemble imports
        ('from src.model_ensemble', 'from src.model_ensemble'),
        ('from src import model_ensemble', 'from src from src import model_ensemble'),
        
        # From advanced_ensemble imports  
        ('from src.advanced_ensemble', 'from src.advanced_ensemble'),
        ('from src import advanced_ensemble', 'from src from src import advanced_ensemble'),
        
        # From performance_analytics imports
        ('from utils.performance_analytics', 'from utils.performance_analytics'),
        ('from utils import performance_analytics', 'from utils from utils import performance_analytics'),
        
        # From object_tracker imports
        ('from utils.object_tracker', 'from utils.object_tracker'),
        ('from utils import object_tracker', 'from utils from utils import object_tracker'),
        
        # From ensemble_demo imports
        ('from src.ensemble_demo', 'from src.ensemble_demo'),
        ('from src import ensemble_demo', 'from src from src import ensemble_demo'),
        
        # From demo_advanced imports
        ('from tests.demo_advanced', 'from tests.demo_advanced'),
        ('from tests import demo_advanced', 'from tests from tests import demo_advanced'),
    ]
    
    fixed_files = 0
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply fixes
            for old_import, new_import in import_fixes:
                content = content.replace(old_import, new_import)
            
            # If content changed, write it back
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ Fixed imports in: {file_path}")
                fixed_files += 1
                
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    print(f"\n🎉 Fixed imports in {fixed_files} files")
    
    # Also create __init__.py files to make directories importable
    init_dirs = ['src', 'utils', 'tests']
    for dir_name in init_dirs:
        init_file = os.path.join(dir_name, '__init__.py')
        if os.path.exists(dir_name) and not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# Package initialization\n')
            print(f"✅ Created: {init_file}")

def create_universal_main():
    """Create a main.py that works regardless of structure"""
    
    main_content = '''#!/usr/bin/env python3
"""
🎯 Object Detection System - Universal Launcher
"""

import os
import sys
import subprocess
import argparse

def setup_environment():
    """Setup the environment"""
    print("🚀 Object Detection System")
    print("=" * 50)
    
    # Check for model files in various locations
    possible_model_locations = [
        'models/',
        './',  # Current directory
        'src/models/'
    ]
    
    required_files = ['yolov3.cfg', 'yolov3.weights', 'yolov4-tiny.cfg', 'yolov4-tiny.weights', 'coco.names']
    
    print("🔍 Checking for model files...")
    for location in possible_model_locations:
        if all(os.path.exists(os.path.join(location, f)) for f in required_files):
            print(f"✅ Models found in: {location}")
            break
    else:
        print("❌ Model files not found in common locations")
        print("💡 Make sure you have:")
        for f in required_files:
            print(f"   - {f}")
        return False
    
    return True

def run_demo(image_path=None, webcam=False):
    """Run the ensemble demo"""
    try:
        # Try different possible locations for the demo
        demo_scripts = [
            'src/ensemble_demo.py',
            'ensemble_demo.py',
            'tests/ensemble_demo.py'
        ]
        
        demo_script = None
        for script in demo_scripts:
            if os.path.exists(script):
                demo_script = script
                break
        
        if not demo_script:
            print("❌ Could not find ensemble_demo.py")
            return
        
        cmd = [sys.executable, demo_script]
        
        if image_path:
            if os.path.exists(image_path):
                cmd.extend(['--image', image_path])
                print(f"📷 Processing: {image_path}")
            else:
                print(f"❌ Image not found: {image_path}")
                return
        elif webcam:
            cmd.append('--webcam')
            print("📹 Starting webcam...")
        
        print(f"🔄 Running: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    except Exception as e:
        print(f"❌ Error running demo: {e}")

def run_advanced():
    """Run advanced analytics"""
    try:
        advanced_scripts = [
            'tests/demo_advanced.py',
            'demo_advanced.py',
            'src/demo_advanced.py'
        ]
        
        advanced_script = None
        for script in advanced_scripts:
            if os.path.exists(script):
                advanced_script = script
                break
        
        if not advanced_script:
            print("❌ Could not find demo_advanced.py")
            return
        
        print("🔬 Starting Advanced Analytics...")
        subprocess.run([sys.executable, advanced_script])
        
    except Exception as e:
        print(f"❌ Error running advanced demo: {e}")

def run_dashboard():
    """Run web dashboard"""
    try:
        dashboard_scripts = [
            'src/dashboard.py',
            'dashboard.py'
        ]
        
        dashboard_script = None
        for script in dashboard_scripts:
            if os.path.exists(script):
                dashboard_script = script
                break
        
        if not dashboard_script:
            print("❌ Could not find dashboard.py")
            return
        
        print("🌐 Starting Streamlit Dashboard...")
        subprocess.run(['streamlit', 'run', dashboard_script])
        
    except FileNotFoundError:
        print("❌ Streamlit not found. Install with: pip install streamlit")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

def list_images():
    """List available images"""
    image_dirs = ['data/images/', 'Image_snapshot/', './']
    
    for img_dir in image_dirs:
        if os.path.exists(img_dir):
            images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if images:
                print(f"\\n📁 Images in {img_dir}:")
                for img in images[:10]:  # Show first 10
                    print(f"   - {img}")
                if len(images) > 10:
                    print(f"   ... and {len(images) - 10} more")
                return images
    
    print("❌ No images found in common directories")
    return []

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

def interactive_mode():
    """Interactive menu"""
    while True:
        print("\\n🎮 INTERACTIVE MENU")
        print("=" * 40)
        print("1. Detect objects in image")
        print("2. Webcam detection")
        print("3. Advanced analytics")
        print("4. Web dashboard")
        print("5. List available images")
        print("6. Exit")
        
        choice = input("\\nSelect option (1-6): ").strip()
        
        if choice == '1':
            images = list_images()
            if images:
                print("\\nEnter image filename (or full path):")
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
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
'''
    
    with open('main.py', 'w') as f:
        f.write(main_content)
    
    print("✅ Created universal main.py")

if __name__ == "__main__":
    fix_import_issues()
    create_universal_main()
    print("\\n🎉 All fixes applied!")
    print("\\n🚀 Now you can run:")
    print("   python main.py --mode interactive")