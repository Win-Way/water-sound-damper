# 🔧 JUPYTER KERNEL DIAGNOSTIC & FIX
# This cell will help identify and fix the kernel issue

print("🔧 JUPYTER KERNEL DIAGNOSTIC & FIX")
print("=" * 50)

import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path[:3]}...")

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
    print("❌ Seaborn:", e)

try:
    from skimage import measure, filters
    print("✅ Scikit-image: Available")
except ImportError as e:
    print("❌ Scikit-image:", e)

print("\n🚀 SOLUTION OPTIONS:")
print("=" * 30)
print("1. Change Jupyter kernel to Python 3.11.0")
print("2. Install packages in current kernel environment")
print("3. Use the corrected fluid dynamics analysis below")

print("\n📋 TO FIX KERNEL ISSUE:")
print("-" * 25)
print("1. In Jupyter: Kernel → Change Kernel → Python 3.11.0")
print("2. Or run: python -m pip install opencv-python matplotlib seaborn scikit-image")
print("3. Restart kernel after installation")




