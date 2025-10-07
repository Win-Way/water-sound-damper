# DIAGNOSTIC Energy Loss Analysis: Verification and Validation
# Double-checking energy loss calculations to ensure accuracy

print("🔍 DIAGNOSTIC ENERGY LOSS ANALYSIS: Verification and Validation")
print("=" * 70)
print("🎯 Goal: Verify energy loss calculations")
print("🔧 Method: Cross-check calculations and identify potential issues")
print("📊 Focus: Validate 60.8% average energy loss result")
print("=" * 70)

def load_csv_data_corrected(filepath):
    """Load CSV data with proper error handling"""
    try:
        df = pd.read_csv(filepath, skiprows=6)
        sample_numbers = df.iloc[:, 0].values
        voltage = df.iloc[:, 2].values
        time_seconds = sample_numbers / 1000.0
        
        return {
            'time': time_seconds,
            'voltage': voltage,
            'dt': 0.001,
            'sampling_rate': 1000.0,
            'duration': time_seconds[-1],
            'n_samples': len(time_seconds)
        }
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def calculate_energy_diagnostic(signal, dt, base_freq, debug=False):
    """Calculate energy with detailed diagnostics"""
    try:
        # Remove DC offset
        signal_centered = signal - np.mean(signal)
        
        if debug:
            print(f"    Signal stats: mean={np.mean(signal):.4f}, std={np.std(signal):.4f}")
            print(f"    Centered signal: mean={np.mean(signal_centered):.4f}, std={np.std(signal_centered):.4f}")
        
        # Apply light smoothing
        window_length = min(21, len(signal_centered) // 50)
        if window_length % 2 == 0:
            window_length += 1
        if window_length < 3:
            window_length = 3
        
        smoothed_signal = savgol_filter(signal_centered, window_length, 2)
        
        if debug:
            print(f"    Smoothed signal: mean={np.mean(smoothed_signal):.4f}, std={np.std(smoothed_signal):.4f}")
        
        # Use longer analysis window for better energy calculation
        analysis_duration = 2.0
        window_samples = int(analysis_duration / dt)
        start_idx = len(smoothed_signal) // 4
        end_idx = start_idx + window_samples
        
        if end_idx > len(smoothed_signal):
            end_idx = len(smoothed_signal)
            start_idx = end_idx - window_samples
        
        analysis_signal = smoothed_signal[start_idx:end_idx]
        
        if debug:
            print(f"    Analysis window: {len(analysis_signal)} samples, duration: {len(analysis_signal)*dt:.3f}s")
            print(f"    Analysis signal: mean={np.mean(analysis_signal):.4f}, std={np.std(analysis_signal):.4f}")
        
        # Calculate energy using Parseval's theorem
        fft_result = fft(analysis_signal)
        freqs = fftfreq(len(analysis_signal), dt)
        
        # Get positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft_result[:len(freqs)//2])
        
        # Calculate total energy (sum of squared magnitudes)
        total_energy = np.sum(positive_fft**2)
        
        # Calculate energy in different frequency bands
        fundamental_band = (positive_freqs >= 0.8 * base_freq) & (positive_freqs <= 1.2 * base_freq)
        harmonic_band = (positive_freqs > 1.2 * base_freq) & (positive_freqs <= 50)  # Up to 50 Hz
        
        fundamental_energy = np.sum(positive_fft[fundamental_band]**2)
        harmonic_energy = np.sum(positive_fft[harmonic_band]**2)
        
        if debug:
            print(f"    Total energy: {total_energy:.2e}")
            print(f"    Fundamental energy ({0.8*base_freq:.1f}-{1.2*base_freq:.1f} Hz): {fundamental_energy:.2e}")
            print(f"    Harmonic energy ({1.2*base_freq:.1f}-50 Hz): {harmonic_energy:.2e}")
            print(f"    Energy check: {fundamental_energy + harmonic_energy:.2e} vs {total_energy:.2e}")
        
        return {
            'total_energy': total_energy,
            'fundamental_energy': fundamental_energy,
            'harmonic_energy': harmonic_energy,
            'freqs': positive_freqs,
            'magnitude': positive_fft,
            'signal': analysis_signal,
            'debug_info': {
                'signal_mean': np.mean(signal),
                'signal_std': np.std(signal),
                'centered_mean': np.mean(signal_centered),
                'smoothed_std': np.std(smoothed_signal),
                'analysis_std': np.std(analysis_signal),
                'window_length': window_length,
                'analysis_duration': len(analysis_signal) * dt
            }
        }
        
    except Exception as e:
        print(f"Error in energy calculation: {e}")
        return None

def analyze_energy_loss_diagnostic(csv_file, base_freq, dry_baseline_energy=None, debug=False):
    """Analyze energy loss with detailed diagnostics"""
    try:
        data = load_csv_data_corrected(csv_file)
        if data is None:
            return None
            
        voltage = data['voltage']
        dt = data['dt']
        
        if debug:
            print(f"  Analyzing {csv_file}")
            print(f"  Base frequency: {base_freq} Hz")
            print(f"  Data duration: {data['duration']:.3f}s, samples: {data['n_samples']}")
        
        # Calculate energy for this file
        energy_result = calculate_energy_diagnostic(voltage, dt, base_freq, debug)
        
        if energy_result is None:
            return None
        
        # If dry baseline is provided, calculate energy loss
        if dry_baseline_energy is not None:
            # Calculate energy loss percentage
            energy_loss_percentage = ((dry_baseline_energy['total_energy'] - energy_result['total_energy']) / 
                                    dry_baseline_energy['total_energy']) * 100
            
            # Calculate harmonic energy gain percentage
            harmonic_gain_percentage = ((energy_result['harmonic_energy'] - dry_baseline_energy['harmonic_energy']) / 
                                     dry_baseline_energy['total_energy']) * 100
            
            # Calculate fundamental energy loss percentage
            fundamental_loss_percentage = ((dry_baseline_energy['fundamental_energy'] - energy_result['fundamental_energy']) / 
                                         dry_baseline_energy['fundamental_energy']) * 100
            
            if debug:
                print(f"    Dry baseline energy: {dry_baseline_energy['total_energy']:.2e}")
                print(f"    Water ball energy: {energy_result['total_energy']:.2e}")
                print(f"    Energy loss: {energy_loss_percentage:.1f}%")
                print(f"    Harmonic gain: {harmonic_gain_percentage:.1f}%")
                print(f"    Fundamental loss: {fundamental_loss_percentage:.1f}%")
            
            return {
                'file': csv_file,
                'base_freq': base_freq,
                'total_energy': energy_result['total_energy'],
                'fundamental_energy': energy_result['fundamental_energy'],
                'harmonic_energy': energy_result['harmonic_energy'],
                'energy_loss_percentage': energy_loss_percentage,
                'harmonic_gain_percentage': harmonic_gain_percentage,
                'fundamental_loss_percentage': fundamental_loss_percentage,
                'freqs': energy_result['freqs'],
                'magnitude': energy_result['magnitude'],
                'debug_info': energy_result['debug_info'],
                'is_compared': True
            }
        else:
            return {
                'file': csv_file,
                'base_freq': base_freq,
                'total_energy': energy_result['total_energy'],
                'fundamental_energy': energy_result['fundamental_energy'],
                'harmonic_energy': energy_result['harmonic_energy'],
                'energy_loss_percentage': 0,
                'harmonic_gain_percentage': 0,
                'fundamental_loss_percentage': 0,
                'freqs': energy_result['freqs'],
                'magnitude': energy_result['magnitude'],
                'debug_info': energy_result['debug_info'],
                'is_compared': False
            }
        
    except Exception as e:
        print(f"❌ Error analyzing energy loss: {e}")
        return None

def create_dry_energy_baseline_diagnostic(ball_size, data_structure, debug=False):
    """Create dry ball energy baseline with diagnostics"""
    print(f"\n⚡ Creating dry energy baseline for {ball_size}...")
    
    if ball_size not in data_structure or 'dry' not in data_structure[ball_size]:
        print(f"❌ No dry data found for {ball_size}")
        return None
    
    dry_data = data_structure[ball_size]['dry']
    baseline_data = {}
    
    for freq_dir, freq_data in dry_data.items():
        freq_value = float(freq_dir.replace(' Hz', ''))
        
        if debug:
            print(f"\n  Processing {freq_dir}...")
        
        # Analyze all dry files for this frequency
        all_energies = []
        all_debug_info = []
        freqs = None
        
        for batch, csv_file in freq_data.items():
            result = analyze_energy_loss_diagnostic(csv_file, freq_value, debug=debug)
            if result:
                all_energies.append({
                    'total_energy': result['total_energy'],
                    'fundamental_energy': result['fundamental_energy'],
                    'harmonic_energy': result['harmonic_energy']
                })
                all_debug_info.append(result['debug_info'])
                if freqs is None:
                    freqs = result['freqs']
        
        if all_energies and freqs is not None:
            # Average all dry ball energy measurements
            avg_total_energy = np.mean([e['total_energy'] for e in all_energies])
            avg_fundamental_energy = np.mean([e['fundamental_energy'] for e in all_energies])
            avg_harmonic_energy = np.mean([e['harmonic_energy'] for e in all_energies])
            
            # Calculate statistics
            total_energies = [e['total_energy'] for e in all_energies]
            energy_std = np.std(total_energies)
            energy_cv = energy_std / avg_total_energy if avg_total_energy > 0 else 0
            
            baseline_data[freq_dir] = {
                'total_energy': avg_total_energy,
                'fundamental_energy': avg_fundamental_energy,
                'harmonic_energy': avg_harmonic_energy,
                'freqs': freqs,
                'file_count': len(all_energies),
                'energy_std': energy_std,
                'energy_cv': energy_cv,
                'debug_info': all_debug_info
            }
            
            print(f"   ✅ {freq_dir}: {len(all_energies)} files")
            print(f"   📊 Avg total energy: {avg_total_energy:.2e} ± {energy_std:.2e}")
            print(f"   📈 Coefficient of variation: {energy_cv:.3f}")
            
            if debug:
                print(f"   🔍 Debug info:")
                for i, debug_info in enumerate(all_debug_info):
                    print(f"      File {i+1}: std={debug_info['analysis_std']:.4f}, duration={debug_info['analysis_duration']:.3f}s")
    
    return baseline_data

def diagnostic_analysis(ball_size, data_structure):
    """Perform diagnostic analysis to verify calculations"""
    print(f"\n🔍 DIAGNOSTIC ANALYSIS FOR {ball_size}")
    print("=" * 50)
    
    # Create dry energy baseline with diagnostics
    dry_baseline = create_dry_energy_baseline_diagnostic(ball_size, data_structure, debug=True)
    
    if dry_baseline is None:
        print(f"❌ Could not create dry energy baseline for {ball_size}")
        return None
    
    # Analyze a few water files with detailed diagnostics
    for water_content in ['half', 'full']:
        if water_content not in data_structure[ball_size]:
            continue
            
        print(f"\n💧 DIAGNOSTIC ANALYSIS: {water_content} water")
        print("-" * 40)
        
        water_data = data_structure[ball_size][water_content]
        
        # Analyze first few files with detailed diagnostics
        file_count = 0
        for freq_dir, freq_data in water_data.items():
            freq_value = float(freq_dir.replace(' Hz', ''))
            
            if freq_dir not in dry_baseline:
                continue
            
            baseline = dry_baseline[freq_dir]
            
            print(f"\n  📊 {freq_dir} Analysis:")
            print(f"  Dry baseline energy: {baseline['total_energy']:.2e}")
            
            # Analyze first file with detailed diagnostics
            for batch, csv_file in freq_data.items():
                if file_count >= 2:  # Limit to first 2 files for diagnostics
                    break
                    
                result = analyze_energy_loss_diagnostic(csv_file, freq_value, baseline, debug=True)
                if result:
                    print(f"  📈 Energy loss: {result['energy_loss_percentage']:.1f}%")
                    print(f"  🌊 Harmonic gain: {result['harmonic_gain_percentage']:.1f}%")
                    print(f"  📉 Fundamental loss: {result['fundamental_loss_percentage']:.1f}%")
                    file_count += 1
                    break
    
    return dry_baseline

def check_calculation_methodology():
    """Check if the calculation methodology is correct"""
    print("\n🔬 CALCULATION METHODOLOGY CHECK")
    print("=" * 50)
    
    print("Current methodology:")
    print("1. Calculate total energy using Parseval's theorem: sum(|FFT|²)")
    print("2. Calculate energy loss: (Dry_Energy - Water_Energy) / Dry_Energy × 100")
    print("3. Use 2-second analysis window")
    print("4. Apply Savitzky-Golay smoothing")
    
    print("\nPotential issues:")
    print("1. Energy calculation method - using |FFT|² vs |FFT|")
    print("2. Analysis window length - 2 seconds might be too long")
    print("3. Smoothing might be affecting energy calculations")
    print("4. Baseline subtraction might be incorrect")
    
    print("\nAlternative approaches:")
    print("1. Use RMS energy: sqrt(mean(signal²))")
    print("2. Use shorter analysis window: 0.5-1 second")
    print("3. Use raw signal without smoothing")
    print("4. Use different energy metric: peak-to-peak, variance, etc.")

def create_alternative_energy_calculation(signal, dt, method='rms'):
    """Create alternative energy calculation methods"""
    try:
        # Remove DC offset
        signal_centered = signal - np.mean(signal)
        
        # Use shorter analysis window
        analysis_duration = 1.0  # 1 second instead of 2
        window_samples = int(analysis_duration / dt)
        start_idx = len(signal_centered) // 4
        end_idx = start_idx + window_samples
        
        if end_idx > len(signal_centered):
            end_idx = len(signal_centered)
            start_idx = end_idx - window_samples
        
        analysis_signal = signal_centered[start_idx:end_idx]
        
        if method == 'rms':
            # RMS energy
            energy = np.sqrt(np.mean(analysis_signal**2))
        elif method == 'variance':
            # Variance energy
            energy = np.var(analysis_signal)
        elif method == 'peak_to_peak':
            # Peak-to-peak energy
            energy = np.max(analysis_signal) - np.min(analysis_signal)
        elif method == 'fft_magnitude':
            # FFT magnitude (not squared)
            fft_result = fft(analysis_signal)
            energy = np.sum(np.abs(fft_result))
        else:
            # Original method: FFT magnitude squared
            fft_result = fft(analysis_signal)
            energy = np.sum(np.abs(fft_result)**2)
        
        return energy
        
    except Exception as e:
        print(f"Error in alternative energy calculation: {e}")
        return None

# Execute diagnostic analysis
print("🚀 Starting diagnostic energy loss analysis...")

# Use existing data structure
# data_structure should be loaded from previous cell

# Check calculation methodology
check_calculation_methodology()

# Perform diagnostic analysis on one ball size
test_ball_size = '65 mm'  # Test with 65mm which showed high energy loss
if test_ball_size in data_structure:
    diagnostic_result = diagnostic_analysis(test_ball_size, data_structure)
    
    print(f"\n🔍 DIAGNOSTIC SUMMARY FOR {test_ball_size}")
    print("=" * 50)
    
    if diagnostic_result:
        print("✅ Diagnostic analysis completed")
        print("🔍 Check the detailed output above for potential issues")
        print("📊 Look for:")
        print("  - Unusual energy values")
        print("  - High coefficient of variation")
        print("  - Inconsistent signal statistics")
        print("  - Analysis window issues")
    else:
        print("❌ Diagnostic analysis failed")
else:
    print(f"❌ Test ball size {test_ball_size} not found in data structure")

print("\n🔧 RECOMMENDATIONS:")
print("1. Check if energy loss values are physically reasonable")
print("2. Verify that dry baseline energy is consistent")
print("3. Consider using alternative energy calculation methods")
print("4. Check if analysis window length is appropriate")
print("5. Verify that smoothing is not affecting energy calculations")
