# Monkey Detection AI Training - Starter Script
# Run this to begin your AI model training journey!

import os
from pathlib import Path

def setup_project_structure():
    """Create project folder structure"""
    folders = [
        'monkey_dataset/images/train',
        'monkey_dataset/images/val', 
        'monkey_dataset/labels/train',
        'monkey_dataset/labels/val',
        'models',
        'results',
        'test_images'
    ]

    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {folder}")

    print("\n🎯 Project structure ready!")

def create_dataset_config():
    """Create dataset YAML configuration"""
    config = '''# Monkey Detection Dataset Configuration
path: monkey_dataset
train: images/train
val: images/val

# Number of classes  
nc: 1

# Class names
names: ['monkey']
'''

    with open('monkey_dataset/data.yaml', 'w') as f:
        f.write(config)

    print("✅ Dataset configuration created: monkey_dataset/data.yaml")

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'ultralytics',
        'opencv-python', 
        'torch',
        'numpy',
        'matplotlib',
        'pillow'
    ]

    print("🔍 Checking requirements...")
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n📦 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("\n🎉 All requirements satisfied!")
        return True

def main():
    """Main setup function"""
    print("🚀 MONKEY DETECTION AI PROJECT SETUP")
    print("=" * 50)

    # Setup project
    setup_project_structure()
    create_dataset_config()

    # Check requirements
    if not check_requirements():
        print("\n⚠️  Please install missing packages before continuing")
        print("Run: pip install ultralytics opencv-python torch numpy matplotlib pillow")
        return

    print("\n" + "=" * 60)
    print("🎉 PROJECT SETUP COMPLETE!")
    print("=" * 60)
    print("\n📋 NEXT STEPS:")
    print("1. Collect monkey images → Put in monkey_dataset/images/train/")
    print("2. Annotate images → Use LabelImg or Roboflow")
    print("3. Train model → Follow the guide")
    print("4. Test model → Integrate with your Arduino system")
    print("\n🐒 Your AI-powered monkey detection system awaits!")

if __name__ == "__main__":
    main()
