#!/usr/bin/env python3
"""
Example usage of the frontalization video processor

This script demonstrates how to use the test_frontalization_video.py script
to process videos with the frontalization model.
"""

import os
import subprocess
import sys

def run_frontalization_example():
    """
    Run example frontalization on a video file
    """
    print("🎬 Face Frontalization Video Processor - Example Usage")
    print("=" * 60)
    
    # Check if the main script exists
    script_path = "test_frontalization_video.py"
    if not os.path.exists(script_path):
        print(f"❌ Error: {script_path} not found in current directory")
        return
    
    # Example 1: Basic frontalization
    print("\n📹 Example 1: Basic frontalization")
    print("   Command: python test_frontalization_video.py input_video.mp4")
    print("   This will create: input_video_frontalized.mp4")
    
    # Example 2: Side-by-side comparison
    print("\n📹 Example 2: Side-by-side comparison")
    print("   Command: python test_frontalization_video.py input_video.mp4 --side-by-side")
    print("   This will create: input_video_frontalized_comparison.mp4")
    
    # Example 3: Custom output path
    print("\n📹 Example 3: Custom output path")
    print("   Command: python test_frontalization_video.py input_video.mp4 -o custom_output.mp4")
    
    # Example 4: Process specific frame range
    print("\n📹 Example 4: Process specific frame range")
    print("   Command: python test_frontalization_video.py input_video.mp4 --start-frame 100 --end-frame 200")
    
    # Example 5: Use different model
    print("\n📹 Example 5: Use different model")
    print("   Command: python test_frontalization_video.py input_video.mp4 -m ./my_model.pt")
    
    # Example 6: Hide progress bar
    print("\n📹 Example 6: Hide progress bar")
    print("   Command: python test_frontalization_video.py input_video.mp4 --no-progress")
    
    print("\n🔧 All available options:")
    print("   --help, -h          : Show help message")
    print("   --output, -o        : Specify output video path")
    print("   --model, -m         : Specify model file path")
    print("   --start-frame       : Starting frame number (0-indexed)")
    print("   --end-frame         : Ending frame number")
    print("   --side-by-side, -s  : Create side-by-side comparison video")
    print("   --no-progress       : Hide progress bar")
    
    print("\n💡 Tips:")
    print("   • The model expects 128x128 input images and outputs values in [-1, 1] range")
    print("   • Output videos maintain the original resolution and FPS")
    print("   • Processing time depends on video length and model complexity")
    print("   • Use --side-by-side to easily compare original vs frontalized results")
    print("   • Frame range options are useful for testing on short segments")

def check_dependencies():
    """
    Check if required dependencies are available
    """
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        'torch',
        'torchvision', 
        'numpy',
        'PIL',
        'cv2'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            else:
                __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Install them with: pip install -r requirements_frontalization.txt")
        return False
    else:
        print("\n✅ All dependencies are available!")
        return True

def main():
    """
    Main function
    """
    print("🚀 Face Frontalization Video Processor")
    print("=" * 60)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Show usage examples
    run_frontalization_example()
    
    if not deps_ok:
        print("\n❌ Please install missing dependencies before running the processor")
    else:
        print("\n🎯 Ready to process videos!")
        print("   Make sure you have:")
        print("   1. A frontalization model file (e.g., generator_v0.pt)")
        print("   2. An input video file")
        print("   3. Sufficient disk space for output videos")

if __name__ == "__main__":
    main()
