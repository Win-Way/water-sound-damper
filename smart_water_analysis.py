# SMART Water Sloshing Analysis: Automated Pattern Detection and Interpretation
# Automatically detects peaks, patterns, and optimal conditions without hardcoding

print("🧠 SMART WATER SLOSHING ANALYSIS")
print("=" * 60)
print("🎯 Features:")
print("  - Automatic peak detection")
print("  - Pattern recognition")
print("  - Optimal condition identification")
print("  - Comprehensive statistical analysis")
print("  - Automated interpretation")
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
        signal_centered = signal - np.mean(signal)
        
        window_length = min(21, len(signal_centered) // 50)
        if window_length % 2 == 0:
            window_length += 1
        if window_length < 3:
            window_length = 3
        
        smoothed_signal = savgol_filter(signal_centered, window_length, 2)
        
        analysis_duration = 2.0
        window_samples = int(analysis_duration / dt)
        start_idx = len(smoothed_signal) // 4
        end_idx = start_idx + window_samples
        
        if end_idx > len(smoothed_signal):
            end_idx = len(smoothed_signal)
            start_idx = end_idx - window_samples
        
        analysis_signal = smoothed_signal[start_idx:end_idx]
        
        fft_result = fft(analysis_signal)
        freqs = fftfreq(len(analysis_signal), dt)
        
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft_result[:len(freqs)//2])
        positive_fft[0] = 0
        
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
        
        harmonic_result = analyze_harmonic_content_detailed(voltage, dt, base_freq)
        
        if harmonic_result is None:
            return None
        
        if dry_baseline is not None:
            dry_freqs = dry_baseline['freqs']
            dry_magnitude = dry_baseline['magnitude']
            dry_interpolated = np.interp(harmonic_result['freqs'], dry_freqs, dry_magnitude)
            cleaned_magnitude = np.maximum(harmonic_result['magnitude'] - dry_interpolated, 0)
            
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
        
        all_magnitudes = []
        freqs = None
        
        for batch, csv_file in freq_data.items():
            result = analyze_ball_with_baseline_subtraction(csv_file, freq_value)
            if result:
                all_magnitudes.append(result['original_magnitude'])
                if freqs is None:
                    freqs = result['freqs']
        
        if all_magnitudes and freqs is not None:
            avg_magnitude = np.mean(all_magnitudes, axis=0)
            baseline_data[freq_dir] = {
                'freqs': freqs,
                'magnitude': avg_magnitude,
                'file_count': len(all_magnitudes)
            }
            print(f"   ✅ {freq_dir}: {len(all_magnitudes)} files averaged")
    
    return baseline_data

def detect_peaks_and_patterns(frequencies, harmonic_strengths, ball_size, water_content):
    """Automatically detect peaks and analyze patterns"""
    try:
        from scipy.signal import find_peaks
        
        # Convert to numpy arrays
        freqs = np.array(frequencies)
        strengths = np.array(harmonic_strengths)
        
        # Find peaks using scipy
        peaks, properties = find_peaks(strengths, height=0.1, distance=2, prominence=0.05)
        
        peak_info = []
        for peak_idx in peaks:
            peak_freq = freqs[peak_idx]
            peak_strength = strengths[peak_idx]
            
            # Calculate peak characteristics
            peak_width = 0
            if len(peaks) > 1:
                # Find width at half maximum
                half_max = peak_strength / 2
                left_idx = peak_idx
                right_idx = peak_idx
                
                # Find left edge
                while left_idx > 0 and strengths[left_idx] > half_max:
                    left_idx -= 1
                
                # Find right edge
                while right_idx < len(strengths) - 1 and strengths[right_idx] > half_max:
                    right_idx += 1
                
                peak_width = freqs[right_idx] - freqs[left_idx]
            
            peak_info.append({
                'frequency': peak_freq,
                'strength': peak_strength,
                'width': peak_width,
                'index': peak_idx
            })
        
        # Analyze overall pattern
        pattern_analysis = analyze_pattern(freqs, strengths, peak_info)
        
        return {
            'peaks': peak_info,
            'pattern': pattern_analysis,
            'ball_size': ball_size,
            'water_content': water_content
        }
        
    except Exception as e:
        print(f"Error in peak detection: {e}")
        return None

def analyze_pattern(frequencies, strengths, peaks):
    """Analyze the overall pattern of harmonic strength"""
    try:
        # Calculate trend
        if len(frequencies) > 2:
            z = np.polyfit(frequencies, strengths, 1)
            trend_slope = z[0]
            trend_type = "increasing" if trend_slope > 0.01 else "decreasing" if trend_slope < -0.01 else "stable"
        else:
            trend_slope = 0
            trend_type = "insufficient_data"
        
        # Calculate statistics
        mean_strength = np.mean(strengths)
        std_strength = np.std(strengths)
        max_strength = np.max(strengths)
        min_strength = np.min(strengths)
        
        # Determine pattern type
        if len(peaks) == 0:
            pattern_type = "flat"
        elif len(peaks) == 1:
            pattern_type = "single_peak"
        elif len(peaks) == 2:
            pattern_type = "double_peak"
        else:
            pattern_type = "multi_peak"
        
        # Calculate frequency range coverage
        freq_range = frequencies[-1] - frequencies[0]
        strength_range = max_strength - min_strength
        
        return {
            'trend_slope': trend_slope,
            'trend_type': trend_type,
            'pattern_type': pattern_type,
            'mean_strength': mean_strength,
            'std_strength': std_strength,
            'max_strength': max_strength,
            'min_strength': min_strength,
            'strength_range': strength_range,
            'freq_range': freq_range,
            'peak_count': len(peaks)
        }
        
    except Exception as e:
        print(f"Error in pattern analysis: {e}")
        return None

def analyze_cleaned_water_effects_smart(ball_size, data_structure):
    """Smart analysis of water effects with automatic pattern detection"""
    print(f"\n🧠 SMART ANALYSIS FOR {ball_size}")
    print("=" * 50)
    
    # Create dry baseline
    dry_baseline = create_dry_baseline(ball_size, data_structure)
    
    if dry_baseline is None:
        print(f"❌ Could not create dry baseline for {ball_size}")
        return None
    
    results = {}
    
    # Analyze half and full water with smart pattern detection
    for water_content in ['half', 'full']:
        if water_content not in data_structure[ball_size]:
            continue
            
        print(f"\n💧 Analyzing {water_content} water (smart analysis)...")
        water_data = data_structure[ball_size][water_content]
        water_results = {}
        
        # Collect all frequency data
        all_frequencies = []
        all_strengths = []
        
        for freq_dir, freq_data in water_data.items():
            freq_value = float(freq_dir.replace(' Hz', ''))
            
            if freq_dir not in dry_baseline:
                print(f"   ⚠️  No dry baseline for {freq_dir}, skipping...")
                continue
            
            baseline = dry_baseline[freq_dir]
            
            file_results = []
            for batch, csv_file in freq_data.items():
                result = analyze_ball_with_baseline_subtraction(csv_file, freq_value, baseline)
                if result:
                    file_results.append(result)
            
            if file_results:
                harmonic_strengths = [r['harmonic_strength'] for r in file_results]
                mean_strength = np.mean(harmonic_strengths)
                std_strength = np.std(harmonic_strengths)
                
                water_results[freq_dir] = {
                    'files': file_results,
                    'mean_harmonic_strength': mean_strength,
                    'std_harmonic_strength': std_strength,
                    'file_count': len(file_results)
                }
                
                all_frequencies.append(freq_value)
                all_strengths.append(mean_strength)
                
                print(f"   📊 {freq_dir}: {len(file_results)} files, strength: {mean_strength:.3f}")
        
        # Perform smart pattern analysis
        if all_frequencies and all_strengths:
            # Sort by frequency
            sorted_data = sorted(zip(all_frequencies, all_strengths))
            sorted_freqs, sorted_strengths = zip(*sorted_data)
            
            # Detect peaks and patterns
            pattern_analysis = detect_peaks_and_patterns(sorted_freqs, sorted_strengths, ball_size, water_content)
            
            water_results['pattern_analysis'] = pattern_analysis
            water_results['frequencies'] = sorted_freqs
            water_results['strengths'] = sorted_strengths
        
        results[water_content] = water_results
    
    return results

def create_smart_visualization(all_results):
    """Create visualization with automatic peak detection and pattern analysis"""
    print("\n🎨 Creating smart visualization with pattern detection...")
    
    ball_sizes = list(all_results.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('SMART Water Sloshing Analysis: Automatic Pattern Detection', fontsize=16, fontweight='bold')
    
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
        
        # Plot with automatic peak detection
        for water_content, water_data in results.items():
            if not water_data or 'pattern_analysis' not in water_data:
                continue
                
            pattern_analysis = water_data['pattern_analysis']
            if pattern_analysis is None:
                continue
                
            frequencies = water_data['frequencies']
            strengths = water_data['strengths']
            
            # Plot main line
            ax.plot(frequencies, strengths, 'o-', 
                   color=water_colors.get(water_content, 'black'),
                   linewidth=2, markersize=6,
                   label=f'{water_content} water')
            
            # Mark detected peaks
            for peak in pattern_analysis['peaks']:
                ax.plot(peak['frequency'], peak['strength'], 'x', 
                       color=water_colors.get(water_content, 'black'),
                       markersize=12, markeredgewidth=3,
                       label=f'Peak at {peak["frequency"]:.1f} Hz' if peak == pattern_analysis['peaks'][0] else "")
            
            # Add trend line
            if len(frequencies) > 2:
                z = np.polyfit(frequencies, strengths, 1)
                p = np.poly1d(z)
                ax.plot(frequencies, p(frequencies), "--", 
                       color=water_colors.get(water_content, 'black'),
                       alpha=0.7, linewidth=1)
        
        ax.set_xlabel('Base Frequency (Hz)')
        ax.set_ylabel('Cleaned Harmonic Strength (0-1)')
        ax.set_title(f'{ball_size} - Smart Pattern Analysis')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def generate_comprehensive_analysis_report(all_results):
    """Generate comprehensive analysis report with automatic interpretation"""
    print("\n📊 COMPREHENSIVE SMART ANALYSIS REPORT")
    print("=" * 80)
    
    # Collect all optimal conditions
    optimal_conditions = []
    pattern_summary = {}
    
    for ball_size, results in all_results.items():
        if results is None:
            continue
            
        print(f"\n🔬 {ball_size} ANALYSIS:")
        print("-" * 40)
        
        for water_content, water_data in results.items():
            if not water_data or 'pattern_analysis' not in water_data:
                continue
                
            pattern_analysis = water_data['pattern_analysis']
            if pattern_analysis is None:
                continue
            
            print(f"\n💧 {water_content.upper()} WATER:")
            print(f"   Pattern Type: {pattern_analysis['pattern']['pattern_type']}")
            print(f"   Trend: {pattern_analysis['pattern']['trend_type']}")
            print(f"   Peak Count: {pattern_analysis['pattern']['peak_count']}")
            print(f"   Max Strength: {pattern_analysis['pattern']['max_strength']:.3f}")
            print(f"   Mean Strength: {pattern_analysis['pattern']['mean_strength']:.3f}")
            
            # Find optimal conditions
            if pattern_analysis['peaks']:
                best_peak = max(pattern_analysis['peaks'], key=lambda x: x['strength'])
                optimal_conditions.append({
                    'ball_size': ball_size,
                    'water_content': water_content,
                    'optimal_freq': best_peak['frequency'],
                    'optimal_strength': best_peak['strength'],
                    'pattern_type': pattern_analysis['pattern']['pattern_type']
                })
                
                print(f"   🎯 Optimal Frequency: {best_peak['frequency']:.1f} Hz")
                print(f"   🎯 Optimal Strength: {best_peak['strength']:.3f}")
            
            # Store pattern summary
            if ball_size not in pattern_summary:
                pattern_summary[ball_size] = {}
            pattern_summary[ball_size][water_content] = pattern_analysis['pattern']
    
    # Find overall optimal conditions
    if optimal_conditions:
        print(f"\n🏆 OVERALL OPTIMAL CONDITIONS:")
        print("=" * 50)
        
        # Sort by strength
        optimal_conditions.sort(key=lambda x: x['optimal_strength'], reverse=True)
        
        for i, condition in enumerate(optimal_conditions[:5]):  # Top 5
            print(f"{i+1}. {condition['ball_size']} {condition['water_content']} water")
            print(f"   Frequency: {condition['optimal_freq']:.1f} Hz")
            print(f"   Strength: {condition['optimal_strength']:.3f}")
            print(f"   Pattern: {condition['pattern_type']}")
            print()
    
    # Pattern analysis summary
    print(f"\n📈 PATTERN ANALYSIS SUMMARY:")
    print("=" * 50)
    
    pattern_types = {}
    for ball_size, water_data in pattern_summary.items():
        for water_content, pattern in water_data.items():
            pattern_type = pattern['pattern_type']
            if pattern_type not in pattern_types:
                pattern_types[pattern_type] = []
            pattern_types[pattern_type].append(f"{ball_size} {water_content}")
    
    for pattern_type, conditions in pattern_types.items():
        print(f"{pattern_type.upper()}: {', '.join(conditions)}")
    
    return optimal_conditions, pattern_summary

# Execute smart analysis
print("🚀 Starting smart water sloshing analysis...")

# Use existing data structure
# data_structure should be loaded from previous cell

# Analyze all ball sizes with smart pattern detection
all_results = {}
ball_sizes = ['10 mm', '30 mm', '47.5 mm', '65 mm', '82.5 mm', '100 mm']

for ball_size in ball_sizes:
    results = analyze_cleaned_water_effects_smart(ball_size, data_structure)
    all_results[ball_size] = results

# Create smart visualizations
if any(results is not None for results in all_results.values()):
    create_smart_visualization(all_results)
    
    # Generate comprehensive report
    optimal_conditions, pattern_summary = generate_comprehensive_analysis_report(all_results)
    
    print("\n✅ Smart water sloshing analysis complete!")
    print("🧠 Key features:")
    print("  - Automatic peak detection")
    print("  - Pattern recognition")
    print("  - Optimal condition identification")
    print("  - Comprehensive statistical analysis")
    print("  - Automated interpretation")
else:
    print("❌ No data available for analysis")
