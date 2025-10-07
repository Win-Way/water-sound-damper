# CLEANED Water Sloshing Analysis: Subtract Dry Ball Baseline
# This isolates pure water sloshing effects by removing ball resonance and system noise

print("🧹 CLEANED WATER SLOSHING ANALYSIS")
print("=" * 60)
print("🎯 Goal: Isolate pure water sloshing effects")
print("🔧 Method: Water Ball Data - Dry Ball Baseline")
print("📊 Result: Clean water sloshing without ball resonance")
print("=" * 60)

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

def analyze_harmonic_content_detailed(signal, dt, base_freq):
    """Detailed harmonic analysis returning full spectrum"""
    try:
        # Remove DC offset
        signal_centered = signal - np.mean(signal)
        
        # Apply light smoothing
        window_length = min(21, len(signal_centered) // 50)
        if window_length % 2 == 0:
            window_length += 1
        if window_length < 3:
            window_length = 3
        
        smoothed_signal = savgol_filter(signal_centered, window_length, 2)
        
        # Use longer window for better frequency resolution
        analysis_duration = 2.0
        window_samples = int(analysis_duration / dt)
        start_idx = len(smoothed_signal) // 4
        end_idx = start_idx + window_samples
        
        if end_idx > len(smoothed_signal):
            end_idx = len(smoothed_signal)
            start_idx = end_idx - window_samples
        
        analysis_signal = smoothed_signal[start_idx:end_idx]
        
        # FFT analysis
        fft_result = fft(analysis_signal)
        freqs = fftfreq(len(analysis_signal), dt)
        
        # Get positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft_result[:len(freqs)//2])
        positive_fft[0] = 0  # Remove DC
        
        return {
            'freqs': positive_freqs,
            'magnitude': positive_fft,
            'signal': analysis_signal
        }
        
    except Exception as e:
        print(f"Error in harmonic analysis: {e}")
        return None

def analyze_ball_with_baseline_subtraction(csv_file, base_freq, dry_baseline=None):
    """Analyze ball data with optional dry baseline subtraction"""
    try:
        data = load_csv_data_corrected(csv_file)
        if data is None:
            return None
            
        voltage = data['voltage']
        dt = data['dt']
        
        # Analyze harmonic content
        harmonic_result = analyze_harmonic_content_detailed(voltage, dt, base_freq)
        
        if harmonic_result is None:
            return None
        
        # If dry baseline is provided, subtract it
        if dry_baseline is not None:
            # Interpolate dry baseline to match current frequency resolution
            dry_freqs = dry_baseline['freqs']
            dry_magnitude = dry_baseline['magnitude']
            
            # Interpolate dry baseline to current frequency grid
            dry_interpolated = np.interp(harmonic_result['freqs'], dry_freqs, dry_magnitude)
            
            # Subtract dry baseline (with some scaling to avoid negative values)
            cleaned_magnitude = np.maximum(harmonic_result['magnitude'] - dry_interpolated, 0)
            
            # Calculate cleaned harmonic strength
            total_energy = np.sum(cleaned_magnitude)
            fundamental_energy = np.sum(cleaned_magnitude[(harmonic_result['freqs'] >= base_freq * 0.8) & 
                                                         (harmonic_result['freqs'] <= base_freq * 1.2)])
            harmonic_energy = total_energy - fundamental_energy
            
            cleaned_harmonic_strength = harmonic_energy / total_energy if total_energy > 0 else 0
            
            return {
                'file': csv_file,
                'base_freq': base_freq,
                'original_magnitude': harmonic_result['magnitude'],
                'cleaned_magnitude': cleaned_magnitude,
                'dry_baseline': dry_interpolated,
                'harmonic_strength': cleaned_harmonic_strength,
                'freqs': harmonic_result['freqs'],
                'is_cleaned': True
            }
        else:
            # No baseline subtraction - calculate original harmonic strength
            total_energy = np.sum(harmonic_result['magnitude'])
            fundamental_energy = np.sum(harmonic_result['magnitude'][(harmonic_result['freqs'] >= base_freq * 0.8) & 
                                                                     (harmonic_result['freqs'] <= base_freq * 1.2)])
            harmonic_energy = total_energy - fundamental_energy
            harmonic_strength = harmonic_energy / total_energy if total_energy > 0 else 0
            
            return {
                'file': csv_file,
                'base_freq': base_freq,
                'original_magnitude': harmonic_result['magnitude'],
                'cleaned_magnitude': harmonic_result['magnitude'],
                'dry_baseline': None,
                'harmonic_strength': harmonic_strength,
                'freqs': harmonic_result['freqs'],
                'is_cleaned': False
            }
        
    except Exception as e:
        print(f"❌ Error analyzing {csv_file}: {e}")
        return None

def create_dry_baseline(ball_size, data_structure):
    """Create dry ball baseline for a specific ball size"""
    print(f"\n📊 Creating dry baseline for {ball_size}...")
    
    if ball_size not in data_structure or 'dry' not in data_structure[ball_size]:
        print(f"❌ No dry data found for {ball_size}")
        return None
    
    dry_data = data_structure[ball_size]['dry']
    baseline_data = {}
    
    for freq_dir, freq_data in dry_data.items():
        freq_value = float(freq_dir.replace(' Hz', ''))
        
        # Analyze all dry files for this frequency
        all_magnitudes = []
        freqs = None
        
        for batch, csv_file in freq_data.items():
            result = analyze_ball_with_baseline_subtraction(csv_file, freq_value)
            if result:
                all_magnitudes.append(result['original_magnitude'])
                if freqs is None:
                    freqs = result['freqs']
        
        if all_magnitudes and freqs is not None:
            # Average all dry ball measurements
            avg_magnitude = np.mean(all_magnitudes, axis=0)
            baseline_data[freq_dir] = {
                'freqs': freqs,
                'magnitude': avg_magnitude,
                'file_count': len(all_magnitudes)
            }
            print(f"   ✅ {freq_dir}: {len(all_magnitudes)} files averaged")
    
    return baseline_data

def analyze_cleaned_water_effects(ball_size, data_structure):
    """Analyze water effects with dry baseline subtraction"""
    print(f"\n🧹 ANALYZING CLEANED WATER EFFECTS FOR {ball_size}")
    print("=" * 50)
    
    # Create dry baseline
    dry_baseline = create_dry_baseline(ball_size, data_structure)
    
    if dry_baseline is None:
        print(f"❌ Could not create dry baseline for {ball_size}")
        return None
    
    results = {}
    
    # Analyze half and full water with baseline subtraction
    for water_content in ['half', 'full']:
        if water_content not in data_structure[ball_size]:
            continue
            
        print(f"\n💧 Analyzing {water_content} water (cleaned)...")
        water_data = data_structure[ball_size][water_content]
        water_results = {}
        
        for freq_dir, freq_data in water_data.items():
            freq_value = float(freq_dir.replace(' Hz', ''))
            
            # Get corresponding dry baseline
            if freq_dir not in dry_baseline:
                print(f"   ⚠️  No dry baseline for {freq_dir}, skipping...")
                continue
            
            baseline = dry_baseline[freq_dir]
            
            # Analyze each file with baseline subtraction
            file_results = []
            for batch, csv_file in freq_data.items():
                result = analyze_ball_with_baseline_subtraction(csv_file, freq_value, baseline)
                if result:
                    file_results.append(result)
            
            if file_results:
                harmonic_strengths = [r['harmonic_strength'] for r in file_results]
                
                water_results[freq_dir] = {
                    'files': file_results,
                    'mean_harmonic_strength': np.mean(harmonic_strengths),
                    'std_harmonic_strength': np.std(harmonic_strengths),
                    'file_count': len(file_results)
                }
                
                print(f"   📊 {freq_dir}: {len(file_results)} files")
                print(f"   🌊 Cleaned harmonic strength: {np.mean(harmonic_strengths):.3f}")
        
        results[water_content] = water_results
    
    return results

def create_cleaned_visualization(all_results):
    """Create visualization of cleaned water sloshing effects"""
    print("\n🎨 Creating cleaned water sloshing visualization...")
    
    ball_sizes = list(all_results.keys())
    
    # Create subplot layout: 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('CLEANED Water Sloshing Analysis: Pure Water Effects (Baseline Subtracted)', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    water_colors = {'half': 'lightblue', 'full': 'darkblue'}
    
    for i, (ball_size, results) in enumerate(all_results.items()):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        if results is None:
            ax.text(0.5, 0.5, f'{ball_size}\nNo Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='red')
            ax.set_title(f'{ball_size} - No Data Available')
            continue
        
        # Plot cleaned harmonic strength for each water content
        for water_content, water_data in results.items():
            if not water_data:
                continue
                
            frequencies = []
            harmonic_strengths = []
            
            for freq_dir, freq_data in water_data.items():
                freq_value = float(freq_dir.replace(' Hz', ''))
                frequencies.append(freq_value)
                harmonic_strengths.append(freq_data['mean_harmonic_strength'])
            
            if frequencies:
                # Sort by frequency
                sorted_data = sorted(zip(frequencies, harmonic_strengths))
                frequencies, harmonic_strengths = zip(*sorted_data)
                
                ax.plot(frequencies, harmonic_strengths, 'o-', 
                       color=water_colors.get(water_content, 'black'),
                       linewidth=2, markersize=6,
                       label=f'{water_content} water (cleaned)')
        
        ax.set_xlabel('Base Frequency (Hz)')
        ax.set_ylabel('Cleaned Harmonic Strength (0-1)')
        ax.set_title(f'{ball_size} - Pure Water Sloshing Effects')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def create_comparison_visualization(all_results):
    """Create comparison between original and cleaned results"""
    print("\n📊 Creating original vs cleaned comparison...")
    
    # This would require storing both original and cleaned results
    # For now, just show the cleaned results
    return create_cleaned_visualization(all_results)

# Execute cleaned analysis
print("🚀 Starting cleaned water sloshing analysis...")

# Use existing data structure
# data_structure should be loaded from previous cell

# Analyze all ball sizes with baseline subtraction
all_results = {}
ball_sizes = ['10 mm', '30 mm', '47.5 mm', '65 mm', '82.5 mm', '100 mm']

for ball_size in ball_sizes:
    results = analyze_cleaned_water_effects(ball_size, data_structure)
    all_results[ball_size] = results

# Create visualizations
if any(results is not None for results in all_results.values()):
    create_cleaned_visualization(all_results)
    
    # Summary statistics
    print("\n🧹 CLEANED ANALYSIS SUMMARY")
    print("=" * 60)
    
    for ball_size, results in all_results.items():
        if results:
            print(f"\n🔬 {ball_size} Results (Cleaned):")
            for water_content, water_data in results.items():
                if water_data:
                    total_files = sum(freq_data['file_count'] for freq_data in water_data.values())
                    avg_harmonic_strength = np.mean([freq_data['mean_harmonic_strength'] for freq_data in water_data.values()])
                    print(f"   💧 {water_content}: {total_files} files, avg cleaned harmonic strength: {avg_harmonic_strength:.3f}")
    
    print("\n✅ Cleaned water sloshing analysis complete!")
    print("🧹 Key improvements:")
    print("  - Subtracted dry ball baseline")
    print("  - Isolated pure water sloshing effects")
    print("  - Removed ball resonance and system noise")
    print("  - Shows true water energy absorption")
else:
    print("❌ No data available for analysis")
