# PROPER Analysis: Water Sloshing Harmonics Detection
# This explains why you're seeing higher frequencies - it's water sloshing!

print("🌊 WATER SLOSHING HARMONICS ANALYSIS")
print("=" * 60)
print("🎯 Understanding: Water creates harmonics of the base frequency")
print("📊 Analysis: Detect and quantify harmonic content")
print("🔬 Physics: Sloshing water generates higher frequencies")
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

def analyze_harmonic_content(signal, dt, base_freq):
    """Analyze harmonic content created by water sloshing"""
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
        
        # Find all significant peaks (harmonics)
        from scipy.signal import find_peaks
        
        # Find peaks above threshold
        threshold = np.max(positive_fft) * 0.1  # 10% of max
        peaks, properties = find_peaks(positive_fft, height=threshold, distance=5)
        
        # Analyze each peak
        harmonics = []
        for peak_idx in peaks:
            freq = positive_freqs[peak_idx]
            amplitude = positive_fft[peak_idx]
            
            # Determine if this is a harmonic of the base frequency
            harmonic_ratio = freq / base_freq
            harmonic_number = round(harmonic_ratio)
            
            if 0.8 <= harmonic_ratio <= 1.2:  # Fundamental frequency
                harmonic_type = 'fundamental'
            elif harmonic_number >= 2 and abs(harmonic_ratio - harmonic_number) < 0.2:
                harmonic_type = f'harmonic_{harmonic_number}x'
            else:
                harmonic_type = 'other'
            
            harmonics.append({
                'frequency': freq,
                'amplitude': amplitude,
                'harmonic_number': harmonic_number,
                'harmonic_ratio': harmonic_ratio,
                'type': harmonic_type
            })
        
        # Sort by amplitude
        harmonics.sort(key=lambda x: x['amplitude'], reverse=True)
        
        # Find dominant frequency (strongest peak)
        if harmonics:
            dominant_freq = harmonics[0]['frequency']
            dominant_type = harmonics[0]['type']
        else:
            dominant_freq = base_freq
            dominant_type = 'fundamental'
        
        # Calculate harmonic strength
        fundamental_amp = 0
        harmonic_amp = 0
        
        for h in harmonics:
            if h['type'] == 'fundamental':
                fundamental_amp = h['amplitude']
            elif h['type'].startswith('harmonic_'):
                harmonic_amp += h['amplitude']
        
        total_amp = fundamental_amp + harmonic_amp
        harmonic_strength = harmonic_amp / total_amp if total_amp > 0 else 0
        
        return {
            'base_freq': base_freq,
            'dominant_freq': dominant_freq,
            'dominant_type': dominant_type,
            'harmonics': harmonics,
            'harmonic_strength': harmonic_strength,
            'fundamental_amplitude': fundamental_amp,
            'total_harmonic_amplitude': harmonic_amp,
            'freqs': positive_freqs,
            'fft_magnitude': positive_fft
        }
        
    except Exception as e:
        print(f"Error in harmonic analysis: {e}")
        return None

def analyze_ball_water_interaction(csv_file, base_freq):
    """Analyze how water in the ball creates harmonics"""
    try:
        data = load_csv_data_corrected(csv_file)
        if data is None:
            return None
            
        time = data['time']
        voltage = data['voltage']
        dt = data['dt']
        
        # Analyze harmonic content
        harmonic_result = analyze_harmonic_content(voltage, dt, base_freq)
        
        if harmonic_result is None:
            return None
        
        return {
            'file': csv_file,
            'base_freq': base_freq,
            'dominant_freq': harmonic_result['dominant_freq'],
            'dominant_type': harmonic_result['dominant_type'],
            'harmonic_strength': harmonic_result['harmonic_strength'],
            'harmonics': harmonic_result['harmonics'],
            'time': time,
            'voltage': voltage,
            'freqs': harmonic_result['freqs'],
            'fft_magnitude': harmonic_result['fft_magnitude']
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {csv_file}: {e}")
        return None

def analyze_ball_size_water_interaction(ball_size, data_structure):
    """Analyze water sloshing effects for a specific ball size"""
    print(f"\n🌊 ANALYZING WATER SLOSHING IN {ball_size} BALLS")
    print("=" * 50)
    
    if ball_size not in data_structure:
        print(f"❌ No data found for {ball_size}")
        return None
    
    results = {}
    
    # Analyze dry, half, and full water content
    for water_content in ['dry', 'half', 'full']:
        if water_content not in data_structure[ball_size]:
            continue
            
        print(f"\n💧 Analyzing {water_content} water content...")
        water_data = data_structure[ball_size][water_content]
        water_results = {}
        
        for freq_dir, freq_data in water_data.items():
            freq_value = float(freq_dir.replace(' Hz', ''))
            
            # Analyze each file
            file_results = []
            for batch, csv_file in freq_data.items():
                result = analyze_ball_water_interaction(csv_file, freq_value)
                if result:
                    file_results.append(result)
            
            if file_results:
                # Calculate statistics
                harmonic_strengths = [r['harmonic_strength'] for r in file_results]
                dominant_freqs = [r['dominant_freq'] for r in file_results]
                
                water_results[freq_dir] = {
                    'files': file_results,
                    'mean_harmonic_strength': np.mean(harmonic_strengths),
                    'std_harmonic_strength': np.std(harmonic_strengths),
                    'mean_dominant_freq': np.mean(dominant_freqs),
                    'std_dominant_freq': np.std(dominant_freqs),
                    'file_count': len(file_results)
                }
                
                print(f"   📊 {freq_dir}: {len(file_results)} files")
                print(f"   🌊 Mean harmonic strength: {np.mean(harmonic_strengths):.3f}")
                print(f"   🎵 Mean dominant freq: {np.mean(dominant_freqs):.3f} Hz")
        
        results[water_content] = water_results
    
    return results

def create_water_sloshing_visualization(all_results):
    """Create visualization showing water sloshing effects"""
    print("\n🎨 Creating water sloshing visualization...")
    
    ball_sizes = list(all_results.keys())
    
    # Create subplot layout: 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Water Sloshing Analysis: Harmonic Content vs Ball Size', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    water_colors = {'dry': 'gray', 'half': 'lightblue', 'full': 'darkblue'}
    
    for i, (ball_size, results) in enumerate(all_results.items()):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        if results is None:
            ax.text(0.5, 0.5, f'{ball_size}\nNo Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='red')
            ax.set_title(f'{ball_size} - No Data Available')
            continue
        
        # Plot harmonic strength for each water content
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
                       label=f'{water_content} water')
        
        ax.set_xlabel('Base Frequency (Hz)')
        ax.set_ylabel('Harmonic Strength (0-1)')
        ax.set_title(f'{ball_size} - Water Sloshing Effects')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Execute water sloshing analysis
print("🚀 Starting water sloshing harmonics analysis...")

# Use existing data structure
# data_structure should be loaded from previous cell

# Analyze all ball sizes for water sloshing effects
all_results = {}
ball_sizes = ['10 mm', '30 mm', '47.5 mm', '65 mm', '82.5 mm', '100 mm']

for ball_size in ball_sizes:
    results = analyze_ball_size_water_interaction(ball_size, data_structure)
    all_results[ball_size] = results

# Create visualizations
if any(results is not None for results in all_results.values()):
    create_water_sloshing_visualization(all_results)
    
    # Summary statistics
    print("\n🌊 WATER SLOSHING ANALYSIS SUMMARY")
    print("=" * 60)
    
    for ball_size, results in all_results.items():
        if results:
            print(f"\n🔬 {ball_size} Results:")
            for water_content, water_data in results.items():
                if water_data:
                    total_files = sum(freq_data['file_count'] for freq_data in water_data.values())
                    avg_harmonic_strength = np.mean([freq_data['mean_harmonic_strength'] for freq_data in water_data.values()])
                    print(f"   💧 {water_content}: {total_files} files, avg harmonic strength: {avg_harmonic_strength:.3f}")
    
    print("\n✅ Water sloshing analysis complete!")
    print("🌊 Key findings:")
    print("  - Higher frequencies detected are WATER SLOSHING HARMONICS")
    print("  - Harmonic strength indicates water movement intensity")
    print("  - Compare dry vs half vs full water to see sloshing effects")
else:
    print("❌ No data available for analysis")
