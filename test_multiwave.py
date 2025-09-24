#!/usr/bin/env python3
"""
Test script for Multi-Wave Superposition Analysis
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import differential_evolution
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Set up matplotlib
plt.rcParams['figure.figsize'] = (20, 12)

def improved_multi_wave_model(t, A0, f0, phi0, offset, A1, f1, zeta1, phi1, A2, f2, zeta2, phi2):
    """Multi-component wave model with physical interpretation"""
    # Primary driving component (from mechanical shaker)
    primary = A0 * np.sin(2 * np.pi * f0 * t + phi0)

    # Damped oscillatory components
    omega1 = 2 * np.pi * f1
    damped1 = A1 * np.exp(-zeta1 * omega1 * t) * np.sin(omega1 * t + phi1)

    omega2 = 2 * np.pi * f2
    damped2 = A2 * np.exp(-zeta2 * omega2 * t) * np.sin(omega2 * t + phi2)

    return primary + damped1 + damped2 + offset

def classify_components(f1, f2, expected_freq, zeta1, zeta2):
    """Classify components based on frequency ratios and damping characteristics"""
    f1_ratio = f1 / expected_freq
    f2_ratio = f2 / expected_freq

    # Component 1 classification
    if f1_ratio < 0.8:
        comp1_name = "Water Sloshing (Low Freq)"
        comp1_desc = "Subharmonic water motion within ball"
    elif f1_ratio < 2.0:
        comp1_name = "Internal Resonance"
        comp1_desc = "Ball-water system natural frequency"
    elif f1_ratio < 4.0:
        comp1_name = "Surface Wave Motion"
        comp1_desc = "Water surface oscillations"
    else:
        comp1_name = "High-Freq Turbulence"
        comp1_desc = "Small-scale fluid motion"

    # Component 2 classification
    if f2_ratio < 0.8:
        comp2_name = "Ball-Surface Friction"
        comp2_desc = "Contact damping between ball and surface"
    elif f2_ratio < 2.0:
        comp2_name = "Structural Coupling"
        comp2_desc = "Ball deformation and material damping"
    elif f2_ratio < 4.0:
        comp2_name = "Fluid Viscosity Effects"
        comp2_desc = "Water internal friction and viscosity"
    else:
        comp2_name = "Acoustic Resonance"
        comp2_desc = "Sound waves within water-filled ball"

    return comp1_name, comp1_desc, comp2_name, comp2_desc

def create_improved_superposition_chart(csv_filename, expected_freq, freq_name):
    """Create comprehensive multi-wave superposition visualization with physical interpretation"""
    print(f"\n📊 Creating improved multi-wave superposition chart for {freq_name}")

    try:
        # Load and process data
        print(f"   Loading data from {csv_filename}...")
        df = pd.read_csv(csv_filename, header=None)
        time = df.iloc[:, 0].values
        displacement = df.iloc[:, 1].values
        print(f"   Data loaded: {len(time)} points, duration: {time[-1]:.2f}s")

        # Calculate acceleration
        dt = time[1] - time[0]
        velocity = np.gradient(displacement, dt)
        acceleration = np.gradient(velocity, dt)

        # Apply smoothing
        window_length = min(51, len(acceleration) // 10)
        if window_length % 2 == 0:
            window_length += 1
        smoothed_acceleration = savgol_filter(acceleration, window_length, 3)

        # Use first 2 seconds for detailed analysis
        end_time = 2.0
        mask = time <= end_time
        time_window = time[mask]
        accel_window = smoothed_acceleration[mask]
        print(f"   Analysis window: {len(time_window)} points over {end_time}s")

        # Fit multi-component model using differential evolution
        signal_range = np.max(accel_window) - np.min(accel_window)
        signal_mean = np.mean(accel_window)

        bounds = [
            (0, signal_range * 2),           # A0 bounds
            (expected_freq * 0.9, expected_freq * 1.1),  # f0 bounds
            (-np.pi, np.pi),                 # phi0 bounds
            (signal_mean - signal_range, signal_mean + signal_range),  # offset bounds
            (0, signal_range),               # A1 bounds
            (0.1, 100),                      # f1 bounds
            (0.01, 10),                      # zeta1 bounds
            (-np.pi, np.pi),                 # phi1 bounds
            (0, signal_range),               # A2 bounds
            (0.1, 100),                      # f2 bounds
            (0.01, 10),                      # zeta2 bounds
            (-np.pi, np.pi)                  # phi2 bounds
        ]

        def objective(params):
            try:
                predicted = improved_multi_wave_model(time_window, *params)
                return np.sum((accel_window - predicted)**2)
            except:
                return 1e10

        print(f"   ⏳ Fitting multi-component model (this may take a moment)...")
        result = differential_evolution(objective, bounds, seed=42, maxiter=500)  # Reduced iterations for testing

        if not result.success:
            print(f"❌ Fitting failed for {freq_name}")
            return False

        fitted_params = result.x
        fitted_signal = improved_multi_wave_model(time_window, *fitted_params)

        # Calculate fit quality metrics
        rmse = np.sqrt(np.mean((accel_window - fitted_signal)**2))
        nrmse = (rmse / signal_range) * 100
        r2 = r2_score(accel_window, fitted_signal)

        print(f"   ✅ Fitting successful! NRMSE: {nrmse:.2f}%, R²: {r2:.4f}")

        # Extract fitted parameters
        A0, f0, phi0, offset, A1, f1, zeta1, phi1, A2, f2, zeta2, phi2 = fitted_params

        # Classify components
        comp1_name, comp1_desc, comp2_name, comp2_desc = classify_components(f1, f2, expected_freq, zeta1, zeta2)

        # Generate individual components
        primary_wave = A0 * np.sin(2 * np.pi * f0 * time_window + phi0)
        omega1 = 2 * np.pi * f1
        component1 = A1 * np.exp(-zeta1 * omega1 * time_window) * np.sin(omega1 * time_window + phi1)
        omega2 = 2 * np.pi * f2
        component2 = A2 * np.exp(-zeta2 * omega2 * time_window) * np.sin(omega2 * time_window + phi2)

        # Calculate component contributions (RMS power)
        primary_power = np.sqrt(np.mean(primary_wave**2))
        comp1_power = np.sqrt(np.mean(component1**2))
        comp2_power = np.sqrt(np.mean(component2**2))
        total_power = primary_power + comp1_power + comp2_power

        primary_contrib = (primary_power / total_power) * 100 if total_power > 0 else 0
        comp1_contrib = (comp1_power / total_power) * 100 if total_power > 0 else 0
        comp2_contrib = (comp2_power / total_power) * 100 if total_power > 0 else 0

        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'{freq_name}: Multi-Wave Superposition Analysis\n' +
                    f'Physical Interpretation of Wave Components',
                    fontsize=16, fontweight='bold')

        # 1. Individual wave components (top left)
        ax = axes[0, 0]
        ax.plot(time_window, primary_wave, 'g-', linewidth=2,
               label=f'Shaker Drive ({f0:.1f}Hz)', alpha=0.8)
        ax.plot(time_window, component1, 'orange', linewidth=2,
               label=f'{comp1_name} ({f1:.1f}Hz)', alpha=0.8)
        ax.plot(time_window, component2, 'purple', linewidth=2,
               label=f'{comp2_name} ({f2:.1f}Hz)', alpha=0.8)
        ax.set_title('Individual Wave Components', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 2. Progressive superposition (top center)
        ax = axes[0, 1]
        ax.plot(time_window, primary_wave, 'g-', linewidth=2,
               label='Shaker only', alpha=0.7)
        ax.plot(time_window, primary_wave + component1, 'orange',
               linewidth=2, label='+ Water Effects', alpha=0.7)
        ax.plot(time_window, primary_wave + component1 + component2, 'r-',
               linewidth=2, label='+ All Effects', alpha=0.8)
        ax.set_title('Progressive Wave Addition', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 3. Final comparison with experimental data (top right)
        ax = axes[0, 2]
        ax.plot(time_window, accel_window, 'b-', linewidth=2,
               label='Experimental Data', alpha=0.8)
        ax.plot(time_window, fitted_signal, 'r--', linewidth=2,
               label=f'Multi-Wave Fit (R²={r2:.3f})', alpha=0.8)
        ax.fill_between(time_window, accel_window, fitted_signal, alpha=0.3,
                       color='yellow', label=f'Residual ({nrmse:.1f}% NRMSE)')
        ax.set_title('Experimental vs Multi-Wave Model', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 4. Component power contributions pie chart (bottom left)
        ax = axes[1, 0]
        contributions = [primary_contrib, comp1_contrib, comp2_contrib]
        labels = [f'Shaker Drive\n{primary_contrib:.1f}%',
                 f'{comp1_name}\n{comp1_contrib:.1f}%',
                 f'{comp2_name}\n{comp2_contrib:.1f}%']
        colors = ['lightgreen', 'orange', 'plum']

        # Fixed pie chart call (removed alpha parameter)
        wedges, texts, autotexts = ax.pie(contributions, labels=labels, colors=colors,
                                         autopct='%1.1f%%', startangle=90)
        ax.set_title('Component Power Contributions', fontweight='bold')

        # 5. Physical interpretation table (bottom center)
        ax = axes[1, 1]
        ax.axis('off')

        # Create table data
        table_data = [
            ['Component', 'Type', 'Freq (Hz)', 'Damping', 'Power %'],
            ['Primary', 'Shaker Drive', f'{f0:.1f}', 'N/A', f'{primary_contrib:.1f}%'],
            ['Comp 1', comp1_name[:15], f'{f1:.1f}', f'{zeta1:.3f}', f'{comp1_contrib:.1f}%'],
            ['Comp 2', comp2_name[:15], f'{f2:.1f}', f'{zeta2:.3f}', f'{comp2_contrib:.1f}%']
        ]

        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax.set_title('Component Parameters & Physical Meaning', fontweight='bold', pad=20)

        # 6. Energy dissipation analysis (bottom right)
        ax = axes[1, 2]

        # Calculate energy dissipation rates
        dissip1 = zeta1 * omega1 * (A1**2/2) if zeta1 > 0 and omega1 > 0 else 0
        dissip2 = zeta2 * omega2 * (A2**2/2) if zeta2 > 0 and omega2 > 0 else 0
        total_dissip = dissip1 + dissip2

        dissipation_rates = [dissip1, dissip2]
        comp_names = [comp1_name, comp2_name]
        colors_dissip = ['orange', 'purple']

        bars = ax.bar(range(len(dissipation_rates)), dissipation_rates,
                     color=colors_dissip, alpha=0.7)
        ax.set_xticks(range(len(comp_names)))
        ax.set_xticklabels([name.split('(')[0].strip()[:10] for name in comp_names],
                          fontsize=8, rotation=45, ha='right')
        ax.set_title('Energy Dissipation Rates', fontweight='bold')
        ax.set_ylabel('Dissipation Rate (W)')
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, rate in zip(bars, dissipation_rates):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{rate:.2e}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.show()  # This ensures the plot is displayed

        # Print detailed analysis summary
        print(f"\n🎯 IMPROVED MULTI-WAVE SUPERPOSITION RESULTS for {freq_name}:")
        print(f"✅ Fit Quality: {nrmse:.2f}% NRMSE, R² = {r2:.4f}")
        print(f"📊 Physical Component Analysis:")
        print(f"   • Shaker Drive: {A0:.1f} amplitude, {f0:.2f}Hz, {primary_contrib:.1f}% power")
        print(f"   • {comp1_name}: {A1:.1f} amplitude, {f1:.2f}Hz, ζ={zeta1:.3f}, {comp1_contrib:.1f}% power")
        print(f"     → {comp1_desc}")
        print(f"   • {comp2_name}: {A2:.1f} amplitude, {f2:.2f}Hz, ζ={zeta2:.3f}, {comp2_contrib:.1f}% power")
        print(f"     → {comp2_desc}")
        print(f"⚡ Energy Dissipation Analysis:")
        print(f"   • {comp1_name} decay time: {1/(zeta1*omega1):.3f}s" if zeta1*omega1 > 0 else "   • Component 1: No decay")
        print(f"   • {comp2_name} decay time: {1/(zeta2*omega2):.3f}s" if zeta2*omega2 > 0 else "   • Component 2: No decay")
        print(f"   • Total energy dissipation rate: {total_dissip:.3e} W")

        return True

    except Exception as e:
        print(f"❌ Error in improved analysis for {freq_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎯 IMPROVED MULTI-WAVE SUPERPOSITION ANALYSIS")
    print("=" * 60)
    print("Testing with 16Hz data...")

    # Test with 16Hz data
    csv_file = '10mm16Hz2Adry.csv'
    expected_freq = 16.0
    freq_name = '16Hz System'

    result = create_improved_superposition_chart(csv_file, expected_freq, freq_name)

    if result:
        print(f'\n✅ SUCCESS! Multi-wave superposition chart generated for {freq_name}')
    else:
        print(f'\n❌ FAILED to generate chart for {freq_name}')