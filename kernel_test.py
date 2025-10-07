# ✅ KERNEL TEST: Verify OpenCV Installation
# This cell should work now with the correct kernel

print("✅ KERNEL TEST: Verifying OpenCV Installation")
print("=" * 50)

import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

print("\n🔍 Testing imports:")
print("-" * 20)

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
    print("❌ Seaborn:", sns.__version__)

try:
    from skimage import measure, filters
    print("✅ Scikit-image: Available")
except ImportError as e:
    print("❌ Scikit-image:", e)

print("\n🚀 NEXT STEPS:")
print("=" * 15)
print("1. If all imports work ✅, you can now use the full fluid dynamics analysis")
print("2. Change kernel to 'Python (water_analysis_env)' if not already selected")
print("3. Run the comprehensive fluid dynamics analysis below")

print("\n📋 TO CHANGE KERNEL:")
print("-" * 20)
print("In Jupyter: Kernel → Change Kernel → Python (water_analysis_env)")



