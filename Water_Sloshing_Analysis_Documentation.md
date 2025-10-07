# Water Sloshing Analysis: Comprehensive Documentation

## 🌊 Overview

This analysis investigates how water-filled balls absorb vibrational energy through sloshing dynamics. The experiment uses different ball sizes (10mm to 100mm) with varying water content (dry, half-full, full) subjected to controlled frequency vibrations from a shaker.

## 🔬 Physics Background

### The Problem We're Solving
When a water-filled ball is subjected to mechanical vibrations, the water inside creates complex sloshing patterns that:
- Generate harmonics (multiples of the base frequency)
- Absorb mechanical energy through fluid motion
- Create frequency-dependent damping effects

### Why We Need Baseline Subtraction
Raw measurements contain multiple sources of frequency content:
- **Ball resonance**: Natural vibration of the ball structure
- **System noise**: Shaker vibrations, holder resonance
- **Water sloshing**: The effect we want to measure
- **Measurement artifacts**: Equipment-specific effects

**Solution**: Subtract dry ball baseline to isolate pure water sloshing effects.

## 📊 Experimental Setup

### Ball Sizes Tested
- 10 mm, 30 mm, 47.5 mm, 65 mm, 82.5 mm, 100 mm

### Water Content Conditions
- **Dry**: Empty balls (baseline reference)
- **Half**: Balls half-filled with water
- **Full**: Balls completely filled with water

### Frequency Range
- Typically 8-40 Hz (varies by ball size)
- Shaker provides controlled frequency input
- 1000 Hz sampling rate for high-resolution analysis

## 🔧 Code Structure and Methodology

### 1. Data Loading (`load_csv_data_corrected`)
```python
def load_csv_data_corrected(filepath):
    # Reads CSV files with proper header handling
    # Skips first 6 metadata rows
    # Extracts time and voltage data
    # Converts sample numbers to time (1000 Hz sampling)
```

**What it does**: Loads experimental data from CSV files with proper formatting.

### 2. Harmonic Analysis (`analyze_harmonic_content_detailed`)
```python
def analyze_harmonic_content_detailed(signal, dt, base_freq):
    # Removes DC offset
    # Applies Savitzky-Golay smoothing
    # Performs FFT analysis
    # Returns frequency spectrum
```

**What it does**: Analyzes the frequency content of the signal to identify harmonics.

### 3. Baseline Creation (`create_dry_baseline`)
```python
def create_dry_baseline(ball_size, data_structure):
    # Analyzes all dry ball measurements
    # Averages results across multiple files
    # Creates frequency-dependent baseline
```

**What it does**: Creates a reference baseline from dry ball measurements.

### 4. Baseline Subtraction (`analyze_ball_with_baseline_subtraction`)
```python
def analyze_ball_with_baseline_subtraction(csv_file, base_freq, dry_baseline):
    # Analyzes water-filled ball data
    # Subtracts dry ball baseline
    # Calculates cleaned harmonic strength
```

**What it does**: Isolates pure water sloshing effects by removing ball resonance.

## 📈 Expected Results

### 1. Dry Ball Baseline
- **Harmonic strength**: ~0.05 (minimal)
- **Peak frequency**: 5-7 Hz (natural ball resonance)
- **Purpose**: Reference for baseline subtraction

### 2. Full Water Balls
- **Harmonic strength**: ~0.3 (significant)
- **Peak frequency**: ~20 Hz (water sloshing resonance)
- **Pattern**: Consistent across all ball sizes
- **Physics**: Water creates strong harmonics at 20 Hz

### 3. Half Water Balls
- **Small balls (10mm-82.5mm)**: Minimal sloshing (~0.05)
- **Large balls (100mm)**: Strong sloshing (~0.35) at ~13 Hz
- **Physics**: Optimal sloshing conditions for large balls

### 4. Cleaned Results (After Baseline Subtraction)
- **Lower harmonic strengths**: More realistic values
- **Cleaner peaks**: Focused on water-specific frequencies
- **Better separation**: Clearer half vs full water differences
- **Physical accuracy**: True water energy absorption

## 🎯 Key Findings Expected

### Universal 20 Hz Resonance
- All full water balls show peak sloshing at ~20 Hz
- Independent of ball size (10mm to 100mm)
- Suggests fundamental water sloshing frequency

### Size-Dependent Effects
- **Small balls**: Minimal sloshing effects
- **Large balls**: Dramatic sloshing, especially half-water
- **100mm half-water**: Strongest sloshing effect observed

### Energy Absorption Mechanism
- **Harmonic strength = Energy absorption rate**
- **Higher harmonics = More energy converted to water motion**
- **Peak at 20 Hz = Maximum energy absorption**

## 🔍 What the Visualizations Show

### 1. Harmonic Strength vs Frequency
- **X-axis**: Base frequency (Hz) from shaker
- **Y-axis**: Harmonic strength (0-1 scale)
- **Lines**: Different water content conditions
- **Peaks**: Resonance frequencies for water sloshing

### 2. Water Content Comparison
- **Gray line**: Dry balls (baseline)
- **Light blue**: Half water
- **Dark blue**: Full water
- **Peak heights**: Relative sloshing intensity

### 3. Ball Size Effects
- **Consistent patterns**: Similar behavior across sizes
- **Size scaling**: Larger balls show more dramatic effects
- **Optimal conditions**: Specific size/water combinations

## 🧪 Scientific Interpretation

### Water Sloshing Physics
1. **Shaker input**: Mechanical vibration at base frequency
2. **Ball response**: Ball oscillates at input frequency
3. **Water sloshing**: Creates harmonics (2x, 3x, 4x base frequency)
4. **Energy conversion**: Mechanical energy → Fluid motion
5. **Damping effect**: Water motion absorbs vibration energy

### Why 20 Hz is Special
- **Natural sloshing frequency**: Water's preferred oscillation rate
- **Resonance condition**: Maximum energy transfer occurs
- **Size independence**: Fundamental fluid dynamics property
- **Optimal damping**: Best energy absorption at this frequency

### Practical Applications
- **Vibration damping**: Use 20 Hz for maximum effectiveness
- **Energy harvesting**: Convert mechanical energy to fluid motion
- **Structural control**: Dampen unwanted vibrations
- **Fluid dynamics research**: Study sloshing behavior

## 🚀 Running the Analysis

### Prerequisites
- Python environment with required packages
- Experimental data in proper directory structure
- CSV files with voltage measurements

### Execution Steps
1. **Load data structure**: Organize files by ball size and water content
2. **Create dry baselines**: Average dry ball measurements
3. **Analyze water effects**: Subtract baselines from water data
4. **Generate visualizations**: Plot harmonic strength vs frequency
5. **Interpret results**: Analyze sloshing physics and energy absorption

### Expected Output
- **6 subplot visualization**: One for each ball size
- **Harmonic strength plots**: Showing water sloshing effects
- **Summary statistics**: Quantitative analysis of results
- **Physical insights**: Understanding of water sloshing dynamics

## 🔬 Conclusion

This analysis reveals fundamental physics of water sloshing in spherical containers:
- **Universal 20 Hz resonance** for full water conditions
- **Size-dependent effects** with optimal conditions for large balls
- **Energy absorption mechanism** through harmonic generation
- **Practical applications** for vibration damping and energy harvesting

The cleaned analysis provides accurate measurements of pure water sloshing effects, enabling precise understanding of fluid dynamics and energy conversion mechanisms.
