# 🔧 JUPYTER KERNEL VERIFICATION TEST
# Run this cell to verify your kernel is working correctly

print("🔧 JUPYTER KERNEL VERIFICATION TEST")
print("=" * 50)

import sys
import os

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

# Check if we're in the correct environment
if "water_analysis_env" in sys.executable:
    print("✅ Correct environment detected!")
else:
    print("❌ Wrong environment! Please change kernel to 'Python (water_analysis_env)'")

print("\n🔍 Testing critical imports:")
print("-" * 30)

try:
    import cv2
    print("✅ OpenCV (cv2):", cv2.__version__)
except ImportError as e:
    print("❌ OpenCV (cv2):", e)

try:
    import numpy as np
    print("✅ NumPy:", np.__version__)
except ImportError as e:
    print("❌ NumPy:", e)

try:
    import matplotlib.pyplot as plt
    print("✅ Matplotlib:", plt.matplotlib.__version__)
except ImportError as e:
    print("❌ Matplotlib:", e)

try:
    import seaborn as sns
    print("✅ Seaborn:", sns.__version__)
except ImportError as e:
    print("❌ Seaborn:", e)

try:
    from skimage import measure, filters
    print("✅ Scikit-image: Available")
except ImportError as e:
    print("❌ Scikit-image:", e)

print("\n🚀 IF ALL IMPORTS WORK:")
print("=" * 25)
print("You can now run the fluid dynamics analysis!")
print("If any imports fail, please:")
print("1. Restart your Jupyter kernel")
print("2. Make sure you selected 'Python (water_analysis_env)' kernel")
print("3. Try running this cell again")



