# FIXED Energy Loss Analysis: Corrected Errors
# Fixed base_freq and ax2 variable issues

print("⚡ FIXED ENERGY LOSS ANALYSIS: Water Sloshing Energy Absorption")
print("=" * 70)
print("🎯 Goal: Quantify energy loss due to water sloshing")
print("📊 Method: Compare dry ball energy vs water ball energy")
print("💡 Result: Percentage of energy absorbed by water")
print("🔧 Fixed: base_freq and ax2 variable errors")
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

def calculate_total_energy(signal, dt, base_freq):
    """Calculate total energy in the signal - FIXED with base_freq parameter"""
    try:
        # Remove DC offset
        signal_centered = signal - np.mean(signal)
        
        # Apply light smoothing to reduce noise
        window_length = min(21, len(signal_centered) // 50)
        if window_length % 2 == 0:
            window_length += 1
        if window_length < 3:
            window_length = 3
        
        smoothed_signal = savgol_filter(signal_centered, window_length, 2)
        
        # Use longer analysis window for better energy calculation
        analysis_duration = 2.0
        window_samples = int(analysis_duration / dt)
        start_idx = len(smoothed_signal) // 4
        end_idx = start_idx + window_samples
        
        if end_idx > len(smoothed_signal):
            end_idx = len(smoothed_signal)
            start_idx = end_idx - window_samples
        
        analysis_signal = smoothed_signal[start_idx:end_idx]
        
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
        
        return {
            'total_energy': total_energy,
            'fundamental_energy': fundamental_energy,
            'harmonic_energy': harmonic_energy,
            'freqs': positive_freqs,
            'magnitude': positive_fft,
            'signal': analysis_signal
        }
        
    except Exception as e:
        print(f"Error in energy calculation: {e}")
        return None

def analyze_energy_loss(csv_file, base_freq, dry_baseline_energy=None):
    """Analyze energy loss by comparing to dry ball baseline"""
    try:
        data = load_csv_data_corrected(csv_file)
        if data is None:
            return None
            
        voltage = data['voltage']
        dt = data['dt']
        
        # Calculate energy for this file - FIXED: pass base_freq
        energy_result = calculate_total_energy(voltage, dt, base_freq)
        
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
                'is_compared': False
            }
        
    except Exception as e:
        print(f"❌ Error analyzing energy loss: {e}")
        return None

def create_dry_energy_baseline(ball_size, data_structure):
    """Create dry ball energy baseline for comparison"""
    print(f"\n⚡ Creating dry energy baseline for {ball_size}...")
    
    if ball_size not in data_structure or 'dry' not in data_structure[ball_size]:
        print(f"❌ No dry data found for {ball_size}")
        return None
    
    dry_data = data_structure[ball_size]['dry']
    baseline_data = {}
    
    for freq_dir, freq_data in dry_data.items():
        freq_value = float(freq_dir.replace(' Hz', ''))
        
        # Analyze all dry files for this frequency
        all_energies = []
        freqs = None
        
        for batch, csv_file in freq_data.items():
            result = analyze_energy_loss(csv_file, freq_value)
            if result:
                all_energies.append({
                    'total_energy': result['total_energy'],
                    'fundamental_energy': result['fundamental_energy'],
                    'harmonic_energy': result['harmonic_energy']
                })
                if freqs is None:
                    freqs = result['freqs']
        
        if all_energies and freqs is not None:
            # Average all dry ball energy measurements
            avg_total_energy = np.mean([e['total_energy'] for e in all_energies])
            avg_fundamental_energy = np.mean([e['fundamental_energy'] for e in all_energies])
            avg_harmonic_energy = np.mean([e['harmonic_energy'] for e in all_energies])
            
            baseline_data[freq_dir] = {
                'total_energy': avg_total_energy,
                'fundamental_energy': avg_fundamental_energy,
                'harmonic_energy': avg_harmonic_energy,
                'freqs': freqs,
                'file_count': len(all_energies)
            }
            print(f"   ✅ {freq_dir}: {len(all_energies)} files, avg total energy: {avg_total_energy:.2e}")
    
    return baseline_data

def analyze_energy_loss_by_water_content(ball_size, data_structure):
    """Analyze energy loss for different water content conditions"""
    print(f"\n⚡ ANALYZING ENERGY LOSS FOR {ball_size}")
    print("=" * 50)
    
    # Create dry energy baseline
    dry_baseline = create_dry_energy_baseline(ball_size, data_structure)
    
    if dry_baseline is None:
        print(f"❌ Could not create dry energy baseline for {ball_size}")
        return None
    
    results = {}
    
    # Analyze half and full water energy loss
    for water_content in ['half', 'full']:
        if water_content not in data_structure[ball_size]:
            continue
            
        print(f"\n💧 Analyzing {water_content} water energy loss...")
        water_data = data_structure[ball_size][water_content]
        water_results = {}
        
        for freq_dir, freq_data in water_data.items():
            freq_value = float(freq_dir.replace(' Hz', ''))
            
            # Get corresponding dry baseline
            if freq_dir not in dry_baseline:
                print(f"   ⚠️  No dry baseline for {freq_dir}, skipping...")
                continue
            
            baseline = dry_baseline[freq_dir]
            
            # Analyze each file with energy loss calculation
            file_results = []
            for batch, csv_file in freq_data.items():
                result = analyze_energy_loss(csv_file, freq_value, baseline)
                if result:
                    file_results.append(result)
            
            if file_results:
                # Calculate statistics
                energy_losses = [r['energy_loss_percentage'] for r in file_results]
                harmonic_gains = [r['harmonic_gain_percentage'] for r in file_results]
                fundamental_losses = [r['fundamental_loss_percentage'] for r in file_results]
                
                water_results[freq_dir] = {
                    'files': file_results,
                    'mean_energy_loss': np.mean(energy_losses),
                    'std_energy_loss': np.std(energy_losses),
                    'mean_harmonic_gain': np.mean(harmonic_gains),
                    'std_harmonic_gain': np.std(harmonic_gains),
                    'mean_fundamental_loss': np.mean(fundamental_losses),
                    'std_fundamental_loss': np.std(fundamental_losses),
                    'file_count': len(file_results)
                }
                
                print(f"   📊 {freq_dir}: {len(file_results)} files")
                print(f"   ⚡ Energy loss: {np.mean(energy_losses):.1f}% ± {np.std(energy_losses):.1f}%")
                print(f"   🌊 Harmonic gain: {np.mean(harmonic_gains):.1f}% ± {np.std(harmonic_gains):.1f}%")
                print(f"   📉 Fundamental loss: {np.mean(fundamental_losses):.1f}% ± {np.std(fundamental_losses):.1f}%")
        
        results[water_content] = water_results
    
    return results

def create_energy_loss_visualization(all_results):
    """Create visualization of energy loss analysis - FIXED ax2 error"""
    print("\n🎨 Creating energy loss visualization...")
    
    ball_sizes = list(all_results.keys())
    
    # Create subplot layout: 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('FIXED Energy Loss Analysis: Water Sloshing Energy Absorption', fontsize=16, fontweight='bold')
    
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
        
        # Initialize ax2 for all cases
        ax2 = ax.twinx()
        
        # Plot energy loss for each water content
        for water_content, water_data in results.items():
            if not water_data:
                continue
                
            frequencies = []
            energy_losses = []
            harmonic_gains = []
            
            for freq_dir, freq_data in water_data.items():
                freq_value = float(freq_dir.replace(' Hz', ''))
                frequencies.append(freq_value)
                energy_losses.append(freq_data['mean_energy_loss'])
                harmonic_gains.append(freq_data['mean_harmonic_gain'])
            
            if frequencies:
                # Sort by frequency
                sorted_data = sorted(zip(frequencies, energy_losses, harmonic_gains))
                frequencies, energy_losses, harmonic_gains = zip(*sorted_data)
                
                # Plot energy loss
                ax.plot(frequencies, energy_losses, 'o-', 
                       color=water_colors.get(water_content, 'black'),
                       linewidth=2, markersize=6,
                       label=f'{water_content} water - Energy Loss')
                
                # Plot harmonic gain on secondary y-axis
                ax2.plot(frequencies, harmonic_gains, 's--', 
                        color=water_colors.get(water_content, 'black'),
                        linewidth=2, markersize=4, alpha=0.7,
                        label=f'{water_content} water - Harmonic Gain')
        
        ax.set_xlabel('Base Frequency (Hz)')
        ax.set_ylabel('Energy Loss (%)', color='blue')
        ax2.set_ylabel('Harmonic Gain (%)', color='red')
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        ax.set_title(f'{ball_size} - Energy Loss Analysis')
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def generate_energy_loss_report(all_results):
    """Generate comprehensive energy loss analysis report"""
    print("\n⚡ COMPREHENSIVE ENERGY LOSS ANALYSIS REPORT")
    print("=" * 80)
    
    # Collect all energy loss data
    energy_loss_summary = []
    
    for ball_size, results in all_results.items():
        if results is None:
            continue
            
        print(f"\n🔬 {ball_size} ENERGY LOSS ANALYSIS:")
        print("-" * 40)
        
        for water_content, water_data in results.items():
            if not water_data:
                continue
                
            print(f"\n💧 {water_content.upper()} WATER:")
            
            # Calculate overall statistics
            all_energy_losses = []
            all_harmonic_gains = []
            all_fundamental_losses = []
            
            for freq_dir, freq_data in water_data.items():
                all_energy_losses.append(freq_data['mean_energy_loss'])
                all_harmonic_gains.append(freq_data['mean_harmonic_gain'])
                all_fundamental_losses.append(freq_data['mean_fundamental_loss'])
            
            if all_energy_losses:
                avg_energy_loss = np.mean(all_energy_losses)
                std_energy_loss = np.std(all_energy_losses)
                max_energy_loss = np.max(all_energy_losses)
                
                avg_harmonic_gain = np.mean(all_harmonic_gains)
                avg_fundamental_loss = np.mean(all_fundamental_losses)
                
                print(f"   ⚡ Average Energy Loss: {avg_energy_loss:.1f}% ± {std_energy_loss:.1f}%")
                print(f"   🌊 Average Harmonic Gain: {avg_harmonic_gain:.1f}%")
                print(f"   📉 Average Fundamental Loss: {avg_fundamental_loss:.1f}%")
                print(f"   🎯 Maximum Energy Loss: {max_energy_loss:.1f}%")
                
                # Find frequency with maximum energy loss
                max_loss_freq = None
                for freq_dir, freq_data in water_data.items():
                    if freq_data['mean_energy_loss'] == max_energy_loss:
                        max_loss_freq = freq_dir
                        break
                
                if max_loss_freq:
                    print(f"   🎯 Optimal Frequency: {max_loss_freq}")
                
                # Store for overall ranking
                energy_loss_summary.append({
                    'ball_size': ball_size,
                    'water_content': water_content,
                    'avg_energy_loss': avg_energy_loss,
                    'max_energy_loss': max_energy_loss,
                    'avg_harmonic_gain': avg_harmonic_gain,
                    'avg_fundamental_loss': avg_fundamental_loss
                })
    
    # Find overall optimal conditions
    if energy_loss_summary:
        print(f"\n🏆 OVERALL ENERGY LOSS RANKING:")
        print("=" * 50)
        
        # Sort by average energy loss
        energy_loss_summary.sort(key=lambda x: x['avg_energy_loss'], reverse=True)
        
        for i, condition in enumerate(energy_loss_summary):
            print(f"{i+1}. {condition['ball_size']} {condition['water_content']} water")
            print(f"   Average Energy Loss: {condition['avg_energy_loss']:.1f}%")
            print(f"   Maximum Energy Loss: {condition['max_energy_loss']:.1f}%")
            print(f"   Harmonic Gain: {condition['avg_harmonic_gain']:.1f}%")
            print(f"   Fundamental Loss: {condition['avg_fundamental_loss']:.1f}%")
            print()
    
    return energy_loss_summary

# Execute energy loss analysis
print("🚀 Starting FIXED energy loss analysis...")

# Use existing data structure
# data_structure should be loaded from previous cell

# Analyze all ball sizes for energy loss
all_results = {}
ball_sizes = ['10 mm', '30 mm', '47.5 mm', '65 mm', '82.5 mm', '100 mm']

for ball_size in ball_sizes:
    results = analyze_energy_loss_by_water_content(ball_size, data_structure)
    all_results[ball_size] = results

# Create visualizations
if any(results is not None for results in all_results.values()):
    create_energy_loss_visualization(all_results)
    
    # Generate comprehensive report
    energy_loss_summary = generate_energy_loss_report(all_results)
    
    print("\n✅ FIXED energy loss analysis complete!")
    print("⚡ Key findings:")
    print("  - Quantified energy absorption by water sloshing")
    print("  - Identified optimal conditions for energy loss")
    print("  - Analyzed harmonic gain vs fundamental loss")
    print("  - Ranked ball size/water content combinations")
    print("🔧 Fixed issues:")
    print("  - base_freq parameter added to calculate_total_energy")
    print("  - ax2 variable properly initialized")
else:
    print("❌ No data available for analysis")
