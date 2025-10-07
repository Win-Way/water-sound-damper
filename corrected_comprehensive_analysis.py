# CORRECTED Comprehensive Multi-Ball Size Analysis
# Fixed to handle actual directory structure with spaces

print("🔬 COMPREHENSIVE MULTI-BALL SIZE ANALYSIS (CORRECTED)")
print("=" * 60)
print("📊 Ball Sizes: 10mm, 30mm, 47.5mm, 65mm, 82.5mm, 100mm")
print("🎯 Focus: Dry Ball Benchmark Analysis")
print("📈 Advanced Fourier Analysis for All Ball Sizes")
print("=" * 60)

def load_csv_data_fixed(filepath):
    """Load CSV data from the experiment files with proper format handling"""
    try:
        # Read CSV with proper header handling - skip first 6 metadata lines
        df = pd.read_csv(filepath, skiprows=6)
        
        # Extract time and voltage data
        sample_numbers = df.iloc[:, 0].values  # Sample column
        voltage = df.iloc[:, 2].values  # AI0 (V) column
        
        # Convert sample numbers to time (assuming 1000 Hz sampling rate)
        time_seconds = sample_numbers / 1000.0
        
        return {
            'time': time_seconds,
            'voltage': voltage,
            'dt': 0.001,  # 1ms sampling interval
            'sampling_rate': 1000.0,
            'duration': time_seconds[-1],
            'n_samples': len(time_seconds)
        }
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def organize_data_by_ball_size_corrected(data_path):
    """Organize data by ball size, water content, and frequency - CORRECTED VERSION"""
    data_structure = {}
    
    # Get all ball size directories
    ball_sizes = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    print(f"Found ball size directories: {ball_sizes}")
    
    for ball_size in ball_sizes:
        ball_path = os.path.join(data_path, ball_size)
        data_structure[ball_size] = {}
        
        # Get water content directories (dry, full, half)
        water_contents = [d for d in os.listdir(ball_path) if os.path.isdir(os.path.join(ball_path, d))]
        print(f"  {ball_size} water contents: {water_contents}")
        
        for water_content in water_contents:
            water_path = os.path.join(ball_path, water_content)
            
            # Determine water type based on directory name
            if 'dry' in water_content.lower():
                water_type = 'dry'
            elif 'full' in water_content.lower():
                water_type = 'full'
            elif 'half' in water_content.lower():
                water_type = 'half'
            else:
                continue  # Skip unknown water content types
            
            data_structure[ball_size][water_type] = {}
            
            # Get frequency directories
            frequencies = [d for d in os.listdir(water_path) if os.path.isdir(os.path.join(water_path, d))]
            print(f"    {water_type} frequencies: {frequencies}")
            
            for frequency in frequencies:
                freq_path = os.path.join(water_path, frequency)
                data_structure[ball_size][water_type][frequency] = {}
                
                # Get batch directories
                batches = [d for d in os.listdir(freq_path) if os.path.isdir(os.path.join(freq_path, d))]
                
                for batch in batches:
                    batch_path = os.path.join(freq_path, batch)
                    
                    # Find CSV file in batch directory
                    csv_files = glob.glob(os.path.join(batch_path, "*.csv"))
                    if csv_files:
                        csv_file = csv_files[0]  # Take first (should be only) CSV file
                        data_structure[ball_size][water_type][frequency][batch] = csv_file
    
    return data_structure

def analyze_single_file_advanced(csv_file, expected_freq):
    """Analyze a single CSV file with advanced Fourier analysis"""
    try:
        # Load data using the fixed function
        data = load_csv_data_fixed(csv_file)
        if data is None:
            return None
            
        time = data['time']
        voltage = data['voltage']
        dt = data['dt']
        
        # Calculate acceleration using double differentiation
        velocity = np.gradient(voltage, dt)
        acceleration = np.gradient(velocity, dt)
        
        # Apply smoothing
        window_length = min(51, len(acceleration) // 20)
        if window_length % 2 == 0:
            window_length += 1
        if window_length < 3:
            window_length = 3
        
        smoothed_acceleration = savgol_filter(acceleration, window_length, 2)
        
        # Find optimal analysis window (1 second)
        window_samples = int(1.0 / dt)
        start_idx = len(smoothed_acceleration) // 4
        end_idx = start_idx + window_samples
        
        if end_idx > len(smoothed_acceleration):
            end_idx = len(smoothed_acceleration)
            start_idx = end_idx - window_samples
        
        best_time = time[start_idx:end_idx]
        best_signal = smoothed_acceleration[start_idx:end_idx]
        
        # Advanced Fourier Analysis
        fft_result = fft(best_signal)
        freqs = fftfreq(len(best_signal), dt)
        
        # Get positive frequencies only
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft_result[:len(freqs)//2])
        positive_fft[0] = 0  # Remove DC component
        
        # Find dominant frequency
        if len(positive_fft) > 0:
            detected_freq = positive_freqs[np.argmax(positive_fft)]
        else:
            detected_freq = expected_freq
        
        # Calculate frequency deviation
        freq_deviation = abs(detected_freq - expected_freq)
        
        # Power spectral density analysis
        f_psd, psd = welch(best_signal, fs=1/dt, nperseg=min(256, len(best_signal)//4))
        
        # Find peak in PSD
        psd_peak_freq = f_psd[np.argmax(psd)]
        
        return {
            'file': csv_file,
            'expected_freq': expected_freq,
            'detected_freq': detected_freq,
            'psd_peak_freq': psd_peak_freq,
            'frequency_deviation': freq_deviation,
            'time': time,
            'voltage': voltage,
            'acceleration': acceleration,
            'smoothed_acceleration': smoothed_acceleration,
            'best_time': best_time,
            'best_signal': best_signal,
            'fft_freqs': positive_freqs,
            'fft_magnitude': positive_fft,
            'psd_freqs': f_psd,
            'psd_values': psd,
            'data_duration': time[-1] - time[0],
            'sampling_rate': 1/dt
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {csv_file}: {e}")
        return None

def analyze_ball_size_dry_data(ball_size, data_structure):
    """Analyze all dry data for a specific ball size"""
    print(f"\n🔬 ANALYZING {ball_size} DRY BALL DATA")
    print("=" * 50)
    
    if ball_size not in data_structure or 'dry' not in data_structure[ball_size]:
        print(f"❌ No dry data found for {ball_size}")
        return None
    
    dry_data = data_structure[ball_size]['dry']
    results = {}
    
    for freq_dir, freq_data in dry_data.items():
        print(f"\n📊 Processing {freq_dir} data...")
        freq_value = float(freq_dir.replace(' Hz', ''))
        
        # Analyze each CSV file
        file_results = []
        for batch, csv_file in freq_data.items():
            result = analyze_single_file_advanced(csv_file, freq_value)
            if result:
                file_results.append(result)
        
        if file_results:
            # Calculate statistics across all files for this frequency
            detected_freqs = [r['detected_freq'] for r in file_results]
            deviations = [r['frequency_deviation'] for r in file_results]
            
            results[freq_dir] = {
                'files': file_results,
                'mean_detected_freq': np.mean(detected_freqs),
                'std_detected_freq': np.std(detected_freqs),
                'mean_deviation': np.mean(deviations),
                'std_deviation': np.std(deviations),
                'file_count': len(file_results)
            }
            
            print(f"   ✅ {len(file_results)} files analyzed")
            print(f"   📈 Mean detected freq: {np.mean(detected_freqs):.3f} Hz")
            print(f"   📊 Mean deviation: {np.mean(deviations):.3f} Hz")
    
    return results

def create_comprehensive_visualization(all_results):
    """Create comprehensive visualization for all ball sizes"""
    print("\n🎨 Creating comprehensive visualization...")
    
    ball_sizes = list(all_results.keys())
    n_balls = len(ball_sizes)
    
    # Create subplot layout: 2 rows, 3 columns for 6 ball sizes
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Comprehensive Multi-Ball Size Analysis: Advanced Fourier Analysis', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, (ball_size, results) in enumerate(all_results.items()):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        if results is None:
            ax.text(0.5, 0.5, f'{ball_size}\nNo Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='red')
            ax.set_title(f'{ball_size} - No Data Available')
            continue
        
        # Plot frequency analysis for this ball size
        frequencies = []
        deviations = []
        colors_freq = []
        
        for freq_dir, freq_data in results.items():
            freq_value = float(freq_dir.replace(' Hz', ''))
            frequencies.append(freq_value)
            deviations.append(freq_data['mean_deviation'])
            colors_freq.append(colors[i])
        
        if frequencies:
            ax.scatter(frequencies, deviations, c=colors_freq, s=100, alpha=0.7, 
                      label=f'{ball_size} Dry Balls')
            ax.plot(frequencies, deviations, color=colors[i], alpha=0.5, linewidth=2)
            
            # Add trend line
            if len(frequencies) > 1:
                z = np.polyfit(frequencies, deviations, 1)
                p = np.poly1d(z)
                ax.plot(frequencies, p(frequencies), "--", color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Expected Frequency (Hz)')
        ax.set_ylabel('Frequency Deviation (Hz)')
        ax.set_title(f'{ball_size} - Frequency Accuracy Analysis')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Execute comprehensive analysis with corrected data loading
print("🚀 Starting comprehensive multi-ball size analysis with corrected data loading...")

# Load and organize data with corrected function
data_path = './data'
data_structure = organize_data_by_ball_size_corrected(data_path)

print("\n📁 Data Structure Overview:")
for ball_size in sorted(data_structure.keys()):
    print(f"\n🎯 {ball_size}:")
    for water_content in data_structure[ball_size]:
        frequencies = list(data_structure[ball_size][water_content].keys())
        total_batches = sum(len(data_structure[ball_size][water_content][freq]) for freq in frequencies)
        print(f"   • {water_content}: {len(frequencies)} frequencies, {total_batches} total batches")
        print(f"     Frequencies: {sorted(frequencies)}")

# Analyze all ball sizes
all_results = {}
ball_sizes = ['10 mm', '30 mm', '47.5 mm', '65 mm', '82.5 mm', '100 mm']

for ball_size in ball_sizes:
    results = analyze_ball_size_dry_data(ball_size, data_structure)
    all_results[ball_size] = results

# Create visualizations
if any(results is not None for results in all_results.values()):
    create_comprehensive_visualization(all_results)
    
    # Summary statistics
    print("\n📊 COMPREHENSIVE ANALYSIS SUMMARY")
    print("=" * 60)
    
    for ball_size, results in all_results.items():
        if results:
            print(f"\n🔬 {ball_size} Results:")
            total_files = sum(freq_data['file_count'] for freq_data in results.values())
            avg_deviation = np.mean([freq_data['mean_deviation'] for freq_data in results.values()])
            print(f"   📁 Total files analyzed: {total_files}")
            print(f"   📈 Average frequency deviation: {avg_deviation:.3f} Hz")
            print(f"   🎯 Frequencies tested: {list(results.keys())}")
    
    print("\n✅ Comprehensive multi-ball size analysis complete!")
else:
    print("❌ No data available for analysis")
