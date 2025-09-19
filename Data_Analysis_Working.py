# Water-in-Ball Shaker Experiment Analysis
# Simple, working version

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

def analyze_file(csv_filename, expected_freq):
    """Analyze a single CSV file."""
    print(f"\n🔬 Analyzing {expected_freq}Hz system: {csv_filename}")
    
    try:
        # Load data
        df = pd.read_csv(csv_filename, header=None)
        time = df.iloc[:, 0].values
        displacement = df.iloc[:, 1].values
        
        print(f"📊 Loaded {len(time)} data points, duration: {time[-1]-time[0]:.2f}s")
        
        # Calculate acceleration using simple differentiation
        dt = time[1] - time[0]
        velocity = np.gradient(displacement, dt)
        acceleration = np.gradient(velocity, dt)
        
        # Smooth the acceleration
        window_length = min(51, len(acceleration) // 20)
        if window_length % 2 == 0:
            window_length += 1
        if window_length < 3:
            window_length = 3
        
        smoothed_acceleration = savgol_filter(acceleration, window_length, 2)
        
        # Find best 1-second window (simple approach)
        window_samples = min(int(1.0 / dt), len(smoothed_acceleration) // 2)
        start_idx = len(smoothed_acceleration) // 4  # Start at 1/4 of the data
        end_idx = start_idx + window_samples
        
        if end_idx > len(smoothed_acceleration):
            end_idx = len(smoothed_acceleration)
            start_idx = end_idx - window_samples
        
        best_time = time[start_idx:end_idx]
        best_signal = smoothed_acceleration[start_idx:end_idx]
        
        # Simple frequency detection using FFT
        fft_result = fft(best_signal)
        freqs = fftfreq(len(best_signal), dt)
        
        # Find dominant frequency
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft_result[:len(freqs)//2])
        positive_fft[0] = 0  # Remove DC component
        
        if len(positive_fft) > 0:
            detected_freq = positive_freqs[np.argmax(positive_fft)]
        else:
            detected_freq = 0
        
        # Simple curve fitting - Pure sine
        def sine_func(t, A, phi, C):
            return A * np.sin(2 * np.pi * detected_freq * t + phi) + C
        
        curve_results = {}
        try:
            A_guess = (np.max(best_signal) - np.min(best_signal)) / 2
            C_guess = np.mean(best_signal)
            
            popt, _ = curve_fit(sine_func, best_time, best_signal, 
                              p0=[A_guess, 0, C_guess], maxfev=1000)
            fitted = sine_func(best_time, *popt)
            rms_error = np.sqrt(np.mean((best_signal - fitted)**2))
            
            curve_results['sine'] = {
                'name': 'Pure Sine',
                'fitted': fitted,
                'rms_error': rms_error
            }
        except:
            print("⚠️  Curve fitting failed")
        
        deviation = abs(detected_freq - expected_freq)
        print(f"📈 Detected frequency: {detected_freq:.3f}Hz (deviation: {deviation:.3f}Hz)")
        
        return {
            'expected_freq': expected_freq,
            'detected_freq': detected_freq,
            'frequency_deviation': deviation,
            'time': time,
            'displacement': displacement,
            'smoothed_acceleration': smoothed_acceleration,
            'best_window': (best_time[0], best_time[-1]),
            'best_time': best_time,
            'best_signal': best_signal,
            'curve_results': curve_results,
            'data_duration': time[-1] - time[0]
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def create_visualization(all_results):
    """Create visualization of results."""
    if not all_results:
        print("❌ No data to visualize!")
        return
    
    print("🎨 Creating visualization...")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Water-in-Ball Shaker Experiment: Multi-Frequency Analysis', fontsize=14, fontweight='bold')
    
    colors = ['blue', 'red', 'green']
    freq_names = ['16Hz', '20Hz', '24Hz']
    
    for i, (expected_freq, result) in enumerate(all_results.items()):
        color = colors[i]
        name = freq_names[i]
        
        # Left: Full time series
        ax_left = axes[i, 0]
        ax_left.plot(result['time'], result['smoothed_acceleration'], 
                    color=color, linewidth=1, alpha=0.8, label='Acceleration')
        
        # Highlight best window
        best_start, best_end = result['best_window']
        ax_left.axvspan(best_start, best_end, alpha=0.3, color=color, 
                       label=f'Analysis Window')
        
        ax_left.set_title(f'{name} System - Full Data ({result["data_duration"]:.1f}s)')
        ax_left.set_xlabel('Time (s)')
        ax_left.set_ylabel('Acceleration')
        ax_left.legend()
        ax_left.grid(True, alpha=0.3)
        
        # Right: Best window with curve fit
        ax_right = axes[i, 1]
        ax_right.plot(result['best_time'], result['best_signal'], 
                     'o-', color='orange', linewidth=2, markersize=2, label='Data')
        
        # Plot curve fit if available
        if result['curve_results'] and 'sine' in result['curve_results']:
            curve = result['curve_results']['sine']
            ax_right.plot(result['best_time'], curve['fitted'], 
                         '--', color=color, linewidth=2, 
                         label=f'Sine Fit (RMS: {curve["rms_error"]:.2f})')
        
        ax_right.set_title(f'{name} - Analysis Window (Detected: {result["detected_freq"]:.2f}Hz)')
        ax_right.set_xlabel('Time (s)')
        ax_right.set_ylabel('Acceleration')
        ax_right.legend()
        ax_right.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Main execution
if __name__ == "__main__":
    print("🔬 WATER-IN-BALL SHAKER EXPERIMENT ANALYSIS")
    print("=" * 60)
    
    frequency_files = [
        ("10mm16Hz2Adry.csv", 16.0),
        ("10mm20Hz1Adry.csv", 20.0), 
        ("10mm24Hz1Adry.csv", 24.0)
    ]
    
    all_results = {}
    
    for csv_filename, expected_freq in frequency_files:
        result = analyze_file(csv_filename, expected_freq)
        
        if result is not None:
            all_results[expected_freq] = result
            print(f"✅ {expected_freq}Hz analysis complete!")
        else:
            print(f"❌ {expected_freq}Hz analysis failed!")
    
    print(f"\n🎯 Analysis complete! Successfully analyzed {len(all_results)}/3 frequencies")
    
    # Create visualization
    if all_results:
        create_visualization(all_results)
        
        # Print summary
        print("\n📊 RESULTS SUMMARY:")
        for freq, result in all_results.items():
            print(f"{freq}Hz: Detected {result['detected_freq']:.3f}Hz (deviation: {result['frequency_deviation']:.3f}Hz)")
        
        print("\n✅ Analysis complete!")
    else:
        print("❌ No results to visualize")
