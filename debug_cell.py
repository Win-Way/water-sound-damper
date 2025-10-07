# Add this as the FIRST cell in your notebook to diagnose the issue:

import sys
print("🔍 PYTHON DIAGNOSTIC REPORT")
print("=" * 50)
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Conda environment: {sys.prefix}")
print(f"Virtual env: {hasattr(sys, 'real_prefix') or (sys.base_prefix != sys.prefix)}")
print("\n🔬 Testing imports:")
print("-" * 20)

try:
    import numpy as np
    print("✅ Numpy version:", np.__version__)
except ImportError as e:
    print("❌ Numpy import failed:", e)

try:
    import pandas as pd
    print("✅ Pandas version:", pd.__version__)
except ImportError as e:
    print("❌ Pandas import failed:", e)

print(f"\n📁 Python paths:")
for i, path in enumerate(sys.path[:5]):  # Show first 5 paths
    print(f"  {i+1}. {path}")

print("\n🚀 FORCE FIX - Adding virtual environment to path:")
sys.path.insert(0, '/Users/jeffrey/sources/WinWay/water-sound-damper/water_analysis_env/lib/python3.13/site-packages')
print("✅ Virtual environment path added to sys.path")
print("Now try importing numpy again in the next cell!")
