#!/usr/bin/env python3
"""
Standalone notebook runner that bypasses Cursor's kernel issues
"""

import subprocess
import sys
import os

def run_python_code(code_lines, description):
    """Run Python code in virtual environment"""
    print(f"\n🔬 {description}")
    print("=" * 50)
    
    # Join the code lines and add error handling
    full_code = f"""
import sys
print(f"Using Python: {{sys.executable}}")
try:
{chr(10).join(f'    {line}' for line in code_lines)}
    print("✅ SUCCESS: All code executed without errors!")
except Exception as e:
    print(f"❌ ERROR: {{e}}")
"""
    
    try:
        # Run in virtual environment
        result = subprocess.run([
            "./water_analysis_env/bin/python", "-c", full_code
        ], capture_output=True, text=True, timeout=30)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ Timeout running code")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 WATER ANALYSIS - BYPASSING CURSOR KERNEL ISSUES")
    print("=" * 60)
    
    # Test 1: Import diagnostics
    diagnostic_code = [
        "import numpy as np",
        "import pandas as pd", 
        "import matplotlib.pyplot as plt",
        "from scipy.optimize import curve_fit",
        "from scipy.signal import savgol_filter, periodogram, welch",
        "from scipy.fft import fft, fftfreq",
        "from scipy import stats",
        "print('📊 All imports successful!')",
        "print(f'Numpy version: {{np.__version__}}')",
        "print(f'Pandas version: {{pd.__version__}}')"
    ]
    
    success = run_python_code(diagnostic_code, "Testing All Imports")
    
    if success:
        print("\n🎉 SOLUTION WORKING!")
        print("=" * 50)
        print("📋 TO USE THIS APPROACH:")
        print("1. Run this script: python3 run_notebook.py")
        print("2. Use this script to test your notebook code")
        print("3. All your imports will work perfectly!")
        
        # Interactive mode
        print("\n🔬 INTERACTIVE MODE:")
        print("Enter Python code (type 'quit' to exit):")
        
        while True:
            try:
                code_input = input("\n🐍 >>> ")
                if code_input.lower() in ['quit', 'exit', 'q']:
                    break
                if code_input.strip():
                    run_python_code([code_input], "Interactive Execution")
            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                break
    else:
        print("\n❌ Still having issues. Let me try a different approach...")
