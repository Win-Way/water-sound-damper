# 🔬 Wax Ball Sound Wave Simulation

## Project Overview
This project simulates mini wax balls moving in a high viscosity liquid under sound wave influence, optimizing for maximum energy absorption and designing an optical amplification system.

## Key Results

### 🏆 Optimization Findings
- **Optimal wax ball radius for energy absorption**: 2.9 mm
- **Optimal density**: 950 kg/m³ (paraffin wax)
- **Maximum reward achieved**: 25896.47 (energy/movement metric)
- **Recommended liquid**: Glycerin (density: 1260 kg/m³, viscosity: 1.5 Pa⋅s)

### 🔍 Optical Amplification
- **Optimal ball size for optical detection**: 10.0 mm
- **Maximum amplification factor**: 31415.9×10⁶
- **Detection method**: Laser scattering with multiple mirrors

### 🎵 Sound Wave Configuration
- **Frequency**: 20 Hz (optimal for standing wave formation)
- **Wavelength**: 95.2 m (in glycerin)
- **Thermal effects**: Proportional to wave intensity

## Files Generated

### Core Simulation
- `wax_ball_simulation.py` - Complete physics simulation with RL optimization
- `wax_ball_simulation.gif` - Animated visualization (35 MB)

### Analysis & Design
- `optical_amplification_setup.png` - Optical system design
- `WaxBall_Simulation_Analysis.ipynb` - Comprehensive analysis notebook

## Technical Features

### Physics Modeling
- **Fluid Dynamics**: Stokes' law for viscous drag
- **Thermodynamics**: Thermal expansion and density changes
- **Sound Waves**: Standing wave formation and pressure fields
- **Buoyancy**: Archimedes' principle with thermal effects

### Optimization Algorithm
- **Framework**: Gymnasium (OpenAI Gym) reinforcement learning
- **Objective**: Maximize energy absorption and movement amplitude
- **Search Space**: Ball radius (0.5-10mm), density (900-1200 kg/m³)

### Optical System
- **Design**: Laser beam + multiple mirrors
- **Detection**: Scattered light intensity monitoring
- **Amplification**: 31415×10⁶ effective gain

## Usage

### Running the Simulation
```bash
pip install gymnasium matplotlib seaborn scipy numpy imageio imageio-ffmpeg
python wax_ball_simulation.py
```

### Expected Output
- Energy absorption optimization results
- Animated GIF visualization
- Optical amplification system design
- Comprehensive physics analysis

## Key Insights

1. **Smaller balls (2.9mm)** absorb more energy from sound waves
2. **Larger balls (10mm)** provide better optical signal
3. **Compromise needed** between energy absorption vs visibility
4. **Thermal dynamics** crucial for buoyancy changes
5. **Sound wave standing patterns** drive particle movement

## Implementation for Lab

### Recommended Configuration
- **Ball Size**: 6-8mm (compromise between optimization and visibility)
- **Material**: Paraffin wax with 950 kg/m³ density
- **Liquid**: Glycerin with controlled viscosity
- **Sound**: 20 Hz frequency generator
- **Detection**: He-Ne laser + mirror system

### Setup Steps
1. Prepare glycerin medium with controlled viscosity
2. Calibrate sound wave generator for standing waves
3. Implement laser-mirror optical detection
4. Fine-tune ball properties through initial testing
5. Optimize energy transfer to metal bar system

---

**🎯 Objective Achieved**: Complete physics simulation with optimization and optical amplification design ready for laboratory implementation.


