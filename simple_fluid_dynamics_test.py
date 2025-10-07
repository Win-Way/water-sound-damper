# 🌊 SIMPLE FLUID DYNAMICS TEST
# This is a simplified test to verify everything works before running the full analysis

print("🌊 SIMPLE FLUID DYNAMICS TEST")
print("=" * 40)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Test 1: Check if simulation directory exists
simulation_dir = Path("simulation/ansys-fluent")
print(f"📁 Simulation directory exists: {simulation_dir.exists()}")

if simulation_dir.exists():
    video_files = list(simulation_dir.glob("*.mpeg"))
    print(f"📹 Found {len(video_files)} video files:")
    for video_file in video_files:
        print(f"   - {video_file.name}")
else:
    print("❌ Simulation directory not found!")

# Test 2: Try to load a video (if available)
if simulation_dir.exists() and video_files:
    test_video = str(video_files[0])
    print(f"\n🔬 Testing video loading: {Path(test_video).name}")
    
    try:
        cap = cv2.VideoCapture(test_video)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Successfully loaded frame: {frame.shape}")
                cap.release()
            else:
                print("❌ Could not read frame from video")
                cap.release()
        else:
            print("❌ Could not open video file")
    except Exception as e:
        print(f"❌ Error loading video: {e}")

# Test 3: Create a simple plot
print("\n📊 Testing matplotlib:")
try:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Create sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    ax.plot(x, y, 'b-', linewidth=2, label='Test Signal')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.set_title('Fluid Dynamics Test Plot')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Matplotlib plotting successful!")
    
except Exception as e:
    print(f"❌ Matplotlib error: {e}")

print("\n✅ SIMPLE FLUID DYNAMICS TEST COMPLETE!")
print("If all tests passed, you can now run the full analysis.")



