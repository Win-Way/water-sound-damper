# Data Processing Lessons: From Raw CSV to Scientific Analysis

## Data Pipeline Architecture

### 1. Data Loading and Validation
```python
# Always verify data source
df = pd.read_csv(csv_filename, header=None)
print(f"Loaded {len(time)} data points from {csv_filename}")
print(f"Time range: {time[0]:.3f}s to {time[-1]:.3f}s")

# Verify data integrity
assert len(df.columns) == 2, "Expected 2 columns: time, displacement"
assert not df.isnull().any().any(), "Data contains NaN values"
```

### 2. Signal Processing Chain
```python
# Step 1: Calculate derivatives (displacement → velocity → acceleration)
dt = time[1] - time[0]  # Time step
velocity = np.gradient(displacement, dt)
acceleration = np.gradient(velocity, dt)

# Step 2: Noise reduction
window_length = min(51, len(acceleration) // 10)
if window_length % 2 == 0:
    window_length += 1  # Ensure odd number for Savitzky-Golay
smoothed_acceleration = savgol_filter(acceleration, window_length, 3)
```

### 3. Window Selection Strategy
```python
# Use first 2 seconds for detailed analysis
end_time = 2.0
mask = time <= end_time
time_window = time[mask]
accel_window = smoothed_acceleration[mask]

# Rationale: First 2 seconds contain most dynamic behavior
# before system settles into steady-state
```

## Critical Technical Decisions

### 1. Differentiation Method
**Choice**: `np.gradient()` over finite differences
**Rationale**: 
- Handles variable time steps
- Better numerical stability
- Consistent with scipy ecosystem

### 2. Smoothing Strategy
**Choice**: Savitzky-Golay filter
**Rationale**:
- Preserves peak shapes
- Polynomial-based smoothing
- Adjustable window size and polynomial order

### 3. Window Selection
**Choice**: First 2 seconds
**Rationale**:
- Contains transient behavior
- Shows system response dynamics
- Avoids steady-state artifacts

## Visualization Best Practices

### 1. Inline Display in Jupyter
```python
# CRITICAL: Configure for inline display
%matplotlib inline
plt.rcParams['figure.figsize'] = (20, 12)

# Always use plt.show() for inline display
plt.tight_layout()
plt.show()
```

### 2. Comprehensive Multi-Panel Layout
```python
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
# 6 subplots provide complete analysis view:
# - Individual components
# - Progressive superposition  
# - Experimental vs model
# - Power contributions
# - Parameter table
# - Energy dissipation
```

### 3. Physical Interpretation in Labels
```python
# Instead of generic labels:
labels = ['Component 1', 'Component 2', 'Component 3']

# Use physically meaningful labels:
labels = ['Shaker Drive', 'Water Sloshing', 'Ball-Surface Friction']
```

## Common Pitfalls and Solutions

### 1. Mock Data Trap
**Problem**: Using synthetic data for testing
**Solution**: Always use real experimental data
```python
# BAD
test_signal = np.sin(2 * np.pi * 16 * t) + noise

# GOOD
df = pd.read_csv('10mm16Hz2Adry.csv', header=None)
real_signal = process_experimental_data(df)
```

### 2. Inadequate Model Complexity
**Problem**: Single-frequency models for complex systems
**Solution**: Multi-component superposition
```python
# BAD: Oversimplified
model = A * np.sin(omega * t + phi)

# GOOD: Physically complete
model = A0*sin(ω0*t) + A1*e^(-ζ1*ω1*t)*sin(ω1*t) + A2*e^(-ζ2*ω2*t)*sin(ω2*t)
```

### 3. Poor Optimization Strategy
**Problem**: Local optimization for multi-parameter fitting
**Solution**: Global optimization with proper bounds
```python
# BAD: curve_fit with poor initial guesses
popt, _ = curve_fit(model, t, data, p0=[1, 1, 1])

# GOOD: Differential evolution with bounds
bounds = [(0, 1000), (10, 30), (0, 10)]  # Physical constraints
result = differential_evolution(objective, bounds, seed=42)
```

### 4. Missing Error Handling
**Problem**: Crashes on fitting failures
**Solution**: Robust error handling
```python
try:
    result = differential_evolution(objective, bounds, maxiter=1000)
    if result.success:
        return result.x, fitted_signal
    else:
        print(f"❌ Fitting failed for {freq_name}")
        return None, None
except Exception as e:
    print(f"❌ Error: {str(e)}")
    return None, None
```

## Performance Optimization

### 1. Efficient Data Processing
```python
# Vectorized operations
acceleration = np.gradient(velocity, dt)  # Fast

# Avoid loops for large datasets
# BAD: for i in range(len(time)): ...
# GOOD: np.vectorize() or array operations
```

### 2. Memory Management
```python
# Process data in windows for large datasets
window_size = 2000  # 2 seconds at 1000 Hz
for start_idx in range(0, len(data), window_size):
    window_data = data[start_idx:start_idx + window_size]
    process_window(window_data)
```

### 3. Caching Results
```python
# Cache expensive computations
@lru_cache(maxsize=128)
def expensive_computation(params):
    return compute_result(params)
```

## Data Quality Assurance

### 1. Signal-to-Noise Ratio
```python
signal_power = np.var(smoothed_signal)
noise_power = np.var(smoothed_signal - raw_signal)
snr_db = 10 * np.log10(signal_power / noise_power)
print(f"Signal-to-Noise Ratio: {snr_db:.1f} dB")
```

### 2. Sampling Rate Validation
```python
dt = time[1] - time[0]
fs = 1 / dt
nyquist_freq = fs / 2
print(f"Sampling rate: {fs:.1f} Hz, Nyquist: {nyquist_freq:.1f} Hz")
assert fs >= 2 * max_expected_freq, "Sampling rate too low"
```

### 3. Data Completeness Check
```python
# Check for missing data
missing_data = df.isnull().sum()
if missing_data.any():
    print(f"Warning: Missing data detected: {missing_data}")
    
# Check for consistent time steps
dt_variations = np.diff(time)
dt_std = np.std(dt_variations)
if dt_std > 1e-6:
    print(f"Warning: Inconsistent time steps (std: {dt_std:.2e})")
```

## Lessons for Future Projects

### 1. Start with Data Quality
- Validate data source and integrity
- Check sampling rates and completeness
- Verify physical units and ranges

### 2. Build Incremental Complexity
- Start with simple models
- Add complexity only when needed
- Validate each step against physical principles

### 3. Comprehensive Visualization
- Show multiple perspectives
- Include physical interpretation
- Display inline in notebooks

### 4. Robust Error Handling
- Plan for fitting failures
- Provide informative error messages
- Graceful degradation when possible

### 5. Document Everything
- Explain technical choices
- Provide physical interpretation
- Include parameter bounds and constraints

---
*Technical lessons for robust scientific data processing*
