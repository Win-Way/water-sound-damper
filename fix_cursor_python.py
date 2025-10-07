#!/usr/bin/env python3
"""
Fix Cursor IDE Python Interpreter
This script helps configure Cursor to use the correct virtual environment
"""

import sys
import os
import subprocess

# Get the current Python path
current_python = sys.executable
print(f"Current Python: {current_python}")

# Virtual environment path
venv_python = os.path.join(os.getcwd(), "water_analysis_env", "bin", "python")
print(f"Virtual env Python: {venv_python}")

# Test imports in virtual environment
try:
    result = subprocess.run([
        venv_python, "-c", 
        "import numpy as np; import pandas as pd; import matplotlib.pyplot as plt; print('✅ All packages working!')"
    ], capture_output=True, text=True, timeout=10)
    
    if result.returncode == 0:
        print("✅ Virtual environment has all packages!")
        print("\n📋 TO FIX CURSOR:")
        print("1. Press Cmd + Shift + P")
        print("2. Search: 'Python: Select Interpreter'")
        print(f"3. Choose: {venv_python}")
        print("4. Restart the Jupyter kernel: Cmd + Shift + P → 'Jupyter: Restart Kernel'")
        print("5. Run your cell again")
    else:
        print("❌ Virtual environment issue:")
        print(result.stderr)
        
except Exception as e:
    print(f"❌ Error testing virtual environment: {e}")
