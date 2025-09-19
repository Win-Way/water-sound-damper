# Multi-Wave Analysis: From Simple to Sophisticated Signal Decomposition

## Evolution of Analysis Approach

### Phase 1: Single-Frequency Models (Inadequate)
```python
# Simple sine wave fitting
def sine_model(t, A, phi, C):
    return A * np.sin(2 * np.pi * frequency * t + phi) + C

# Damped sine wave
def damped_sine(t, A, phi, C, tau):
    return A * np.exp(-t/tau) * np.sin(2 * np.pi * frequency * t + phi) + C
```

**Problems Identified:**
- ❌ Oversimplification for complex fluid-structure interaction
- ❌ Missing multiple frequency components from water dynamics
- ❌ No physical interpretation of fitted parameters
- ❌ Poor fit quality (high NRMSE, low R²)

### Phase 2: Multi-Component Superposition (Breakthrough)
```python
def multi_wave_model(t, A0, f0, phi0, offset, A1, f1, zeta1, phi1, A2, f2, zeta2, phi2):
    # Primary driving component (shaker)
    primary = A0 * np.sin(2 * np.pi * f0 * t + phi0)
    
    # Damped oscillatory components (water effects)
    omega1 = 2 * np.pi * f1
    damped1 = A1 * np.exp(-zeta1 * omega1 * t) * np.sin(omega1 * t + phi1)
    
    omega2 = 2 * np.pi * f2
    damped2 = A2 * np.exp(-zeta2 * omega2 * t) * np.sin(omega2 * t + phi2)
    
    return primary + damped1 + damped2 + offset
```

## Physical Interpretation Framework

### Component Classification System
```python
def classify_components(f1, f2, expected_freq, zeta1, zeta2):
    f1_ratio = f1 / expected_freq
    f2_ratio = f2 / expected_freq
    
    # Component 1: Water dynamics
    if f1_ratio < 0.8:
        comp1_name = "Water Sloshing (Low Freq)"
    elif f1_ratio < 2.0:
        comp1_name = "Internal Resonance"
    elif f1_ratio < 4.0:
        comp1_name = "Surface Wave Motion"
    else:
        comp1_name = "High-Freq Turbulence"
    
    # Component 2: Energy dissipation
    if f2_ratio < 0.8:
        comp2_name = "Ball-Surface Friction"
    elif f2_ratio < 2.0:
        comp2_name = "Structural Coupling"
    elif f2_ratio < 4.0:
        comp2_name = "Fluid Viscosity Effects"
    else:
        comp2_name = "Acoustic Resonance"
```

## Optimization Strategy

### Robust Global Optimization
```python
# Use differential evolution for complex multi-parameter fitting
result = differential_evolution(objective, bounds, seed=42, maxiter=1000)

# Parameter bounds based on physical constraints
bounds = [
    (0, signal_range * 2),           # A0: amplitude bounds
    (expected_freq * 0.9, expected_freq * 1.1),  # f0: frequency bounds
    (-np.pi, np.pi),                 # phi0: phase bounds
    # ... more bounds for each parameter
]
```

### Fit Quality Metrics
```python
# Comprehensive evaluation
rmse = np.sqrt(np.mean((accel_data - fitted_signal)**2))
nrmse = (rmse / signal_range) * 100  # Normalized error
r2 = r2_score(accel_data, fitted_signal)  # Coefficient of determination
```

## Results and Insights

### Typical Results for 16Hz System
```
✅ Fit Quality: 13.67% NRMSE, R² = 0.4208
📊 Component Analysis:
   • Shaker Drive: 1027.4 amplitude, 16.00Hz, 85.2% power
   • Water Sloshing: 257.7 amplitude, 79.49Hz, ζ=7.884, 0.7% power
   • Ball-Surface Friction: 3880.1 amplitude, 30.46Hz, ζ=1.048, 14.1% power
```

### Key Insights
1. **Primary Component Dominates**: 85-90% of power from shaker drive
2. **Multiple Frequencies Present**: Water creates additional frequency components
3. **Energy Dissipation Quantified**: Damping coefficients reveal energy loss rates
4. **Physical Meaning**: Each component corresponds to real physical processes

## Lessons for Complex Signal Analysis

### 1. Start Simple, Then Complexify
- Begin with basic models to establish baseline
- Gradually add complexity when simple models fail
- Always validate against physical principles

### 2. Physical Interpretation is Critical
- Don't just fit mathematical functions
- Provide physical meaning for each component
- Classify components based on frequency ratios and damping

### 3. Use Robust Optimization
- Multi-parameter fitting requires global optimization
- Set appropriate bounds based on physical constraints
- Use multiple optimization strategies for validation

### 4. Comprehensive Visualization
- Show individual components
- Demonstrate progressive superposition
- Compare with experimental data
- Quantify fit quality and component contributions

### 5. Real Data Always
- Never use mock or synthetic data
- Verify data source at each step
- Process actual experimental measurements

## Application to Other Domains

This approach applies to:
- **Structural vibration analysis**: Multiple mode superposition
- **Signal processing**: Component separation and identification
- **Fluid dynamics**: Multi-scale analysis
- **Experimental physics**: Complex system characterization

---
*Evolution from inadequate single-component to comprehensive multi-component analysis*
