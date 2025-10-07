# ENHANCED Energy Loss Analysis: Automatic Result Interpretation
# Automatically interprets energy loss patterns and provides scientific insights

print("🧠 ENHANCED ENERGY LOSS ANALYSIS: Automatic Result Interpretation")
print("=" * 70)
print("🎯 Features:")
print("  - Automatic pattern detection")
print("  - Scientific result interpretation")
print("  - Optimal condition identification")
print("  - Physics-based explanations")
print("  - Practical application guidance")
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
    """Calculate total energy in the signal"""
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
        
        total_energy = np.sum(positive_fft**2)
        
        fundamental_band = (positive_freqs >= 0.8 * base_freq) & (positive_freqs <= 1.2 * base_freq)
        harmonic_band = (positive_freqs > 1.2 * base_freq) & (positive_freqs <= 50)
        
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
        
        energy_result = calculate_total_energy(voltage, dt, base_freq)
        
        if energy_result is None:
            return None
        
        if dry_baseline_energy is not None:
            energy_loss_percentage = ((dry_baseline_energy['total_energy'] - energy_result['total_energy']) / 
                                    dry_baseline_energy['total_energy']) * 100
            
            harmonic_gain_percentage = ((energy_result['harmonic_energy'] - dry_baseline_energy['harmonic_energy']) / 
                                     dry_baseline_energy['total_energy']) * 100
            
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

def interpret_energy_loss_pattern(frequencies, energy_losses, ball_size, water_content):
    """Automatically interpret energy loss patterns and provide scientific insights"""
    try:
        freqs = np.array(frequencies)
        losses = np.array(energy_losses)
        
        # Find peaks and valleys
        from scipy.signal import find_peaks
        
        # Find positive peaks (energy absorption)
        pos_peaks, _ = find_peaks(losses, height=10, distance=2)
        
        # Find negative peaks (energy amplification)
        neg_peaks, _ = find_peaks(-losses, height=10, distance=2)
        
        # Calculate statistics
        max_loss = np.max(losses)
        min_loss = np.min(losses)
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        
        # Determine pattern type
        if max_loss > 50 and min_loss < -20:
            pattern_type = "extreme_resonance"
        elif max_loss > 30:
            pattern_type = "strong_absorption"
        elif min_loss < -20:
            pattern_type = "strong_amplification"
        elif std_loss > 20:
            pattern_type = "highly_variable"
        else:
            pattern_type = "moderate_effect"
        
        # Find optimal frequencies
        optimal_freqs = []
        if len(pos_peaks) > 0:
            for peak_idx in pos_peaks:
                optimal_freqs.append({
                    'frequency': freqs[peak_idx],
                    'energy_loss': losses[peak_idx],
                    'type': 'absorption_peak'
                })
        
        # Find problematic frequencies
        problematic_freqs = []
        if len(neg_peaks) > 0:
            for peak_idx in neg_peaks:
                problematic_freqs.append({
                    'frequency': freqs[peak_idx],
                    'energy_loss': losses[peak_idx],
                    'type': 'amplification_peak'
                })
        
        # Generate scientific interpretation
        interpretation = generate_scientific_interpretation(
            pattern_type, max_loss, min_loss, mean_loss, 
            optimal_freqs, problematic_freqs, ball_size, water_content
        )
        
        return {
            'pattern_type': pattern_type,
            'max_loss': max_loss,
            'min_loss': min_loss,
            'mean_loss': mean_loss,
            'std_loss': std_loss,
            'optimal_frequencies': optimal_freqs,
            'problematic_frequencies': problematic_freqs,
            'interpretation': interpretation,
            'ball_size': ball_size,
            'water_content': water_content
        }
        
    except Exception as e:
        print(f"Error in pattern interpretation: {e}")
        return None

def generate_scientific_interpretation(pattern_type, max_loss, min_loss, mean_loss, 
                                     optimal_freqs, problematic_freqs, ball_size, water_content):
    """Generate scientific interpretation of energy loss patterns"""
    
    interpretation = {
        'physics_explanation': '',
        'damping_effectiveness': '',
        'optimal_conditions': '',
        'practical_applications': '',
        'warnings': []
    }
    
    # Physics explanation based on pattern type
    if pattern_type == "extreme_resonance":
        interpretation['physics_explanation'] = f"""
        The {ball_size} ball with {water_content} water shows extreme resonance behavior with energy loss ranging from {min_loss:.1f}% to {max_loss:.1f}%. 
        This indicates strong coupling between the mechanical vibration and water sloshing modes. The negative values suggest 
        energy amplification at certain frequencies, while positive values show effective energy absorption.
        """
        interpretation['damping_effectiveness'] = "Highly frequency-dependent - excellent damping at resonance peaks, but energy amplification at anti-resonance frequencies."
        
    elif pattern_type == "strong_absorption":
        interpretation['physics_explanation'] = f"""
        The {ball_size} ball with {water_content} water shows strong energy absorption with maximum loss of {max_loss:.1f}%. 
        This indicates effective conversion of mechanical energy to water sloshing motion. The water acts as an efficient 
        energy sink, dissipating vibration energy through fluid motion and viscous effects.
        """
        interpretation['damping_effectiveness'] = "Excellent damping effectiveness with consistent energy absorption."
        
    elif pattern_type == "strong_amplification":
        interpretation['physics_explanation'] = f"""
        The {ball_size} ball with {water_content} water shows energy amplification with minimum loss of {min_loss:.1f}%. 
        This indicates that water sloshing is actually adding energy to the system rather than absorbing it. This could be 
        due to resonance effects where water motion amplifies the mechanical vibration.
        """
        interpretation['damping_effectiveness'] = "Poor damping effectiveness - water amplifies vibrations rather than damping them."
        interpretation['warnings'].append("Avoid using this configuration for vibration damping applications.")
        
    elif pattern_type == "highly_variable":
        interpretation['physics_explanation'] = f"""
        The {ball_size} ball with {water_content} water shows highly variable energy loss behavior with significant 
        frequency dependence. This indicates complex interaction between mechanical vibration and water sloshing modes, 
        with different frequencies showing different energy conversion mechanisms.
        """
        interpretation['damping_effectiveness'] = "Variable effectiveness - requires careful frequency selection for optimal damping."
        
    else:  # moderate_effect
        interpretation['physics_explanation'] = f"""
        The {ball_size} ball with {water_content} water shows moderate energy loss effects with average loss of {mean_loss:.1f}%. 
        This indicates moderate coupling between mechanical vibration and water sloshing, providing consistent but not 
        dramatic energy absorption.
        """
        interpretation['damping_effectiveness'] = "Moderate damping effectiveness with consistent behavior across frequencies."
    
    # Optimal conditions
    if optimal_freqs:
        freq_list = [f"{opt['frequency']:.1f} Hz ({opt['energy_loss']:.1f}%)" for opt in optimal_freqs]
        interpretation['optimal_conditions'] = f"Optimal damping frequencies: {', '.join(freq_list)}"
    else:
        interpretation['optimal_conditions'] = "No clear optimal frequencies identified."
    
    # Practical applications
    if max_loss > 50:
        interpretation['practical_applications'] = "Excellent for vibration damping applications - high energy absorption."
    elif max_loss > 20:
        interpretation['practical_applications'] = "Good for vibration damping applications - moderate energy absorption."
    elif min_loss < -20:
        interpretation['practical_applications'] = "Suitable for energy harvesting applications - energy amplification."
    else:
        interpretation['practical_applications'] = "Limited practical applications - low energy conversion efficiency."
    
    return interpretation

def analyze_energy_loss_by_water_content_enhanced(ball_size, data_structure):
    """Enhanced analysis with automatic interpretation"""
    print(f"\n🧠 ENHANCED ANALYSIS FOR {ball_size}")
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
        
        # Collect all frequency data
        all_frequencies = []
        all_energy_losses = []
        all_harmonic_gains = []
        
        for freq_dir, freq_data in water_data.items():
            freq_value = float(freq_dir.replace(' Hz', ''))
            
            if freq_dir not in dry_baseline:
                print(f"   ⚠️  No dry baseline for {freq_dir}, skipping...")
                continue
            
            baseline = dry_baseline[freq_dir]
            
            file_results = []
            for batch, csv_file in freq_data.items():
                result = analyze_energy_loss(csv_file, freq_value, baseline)
                if result:
                    file_results.append(result)
            
            if file_results:
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
                
                all_frequencies.append(freq_value)
                all_energy_losses.append(np.mean(energy_losses))
                all_harmonic_gains.append(np.mean(harmonic_gains))
                
                print(f"   📊 {freq_dir}: {len(file_results)} files")
                print(f"   ⚡ Energy loss: {np.mean(energy_losses):.1f}% ± {np.std(energy_losses):.1f}%")
                print(f"   🌊 Harmonic gain: {np.mean(harmonic_gains):.1f}% ± {np.std(harmonic_gains):.1f}%")
        
        # Perform automatic interpretation
        if all_frequencies and all_energy_losses:
            # Sort by frequency
            sorted_data = sorted(zip(all_frequencies, all_energy_losses, all_harmonic_gains))
            sorted_freqs, sorted_losses, sorted_gains = zip(*sorted_data)
            
            # Interpret patterns
            pattern_analysis = interpret_energy_loss_pattern(sorted_freqs, sorted_losses, ball_size, water_content)
            
            water_results['pattern_analysis'] = pattern_analysis
            water_results['frequencies'] = sorted_freqs
            water_results['energy_losses'] = sorted_losses
            water_results['harmonic_gains'] = sorted_gains
        
        results[water_content] = water_results
    
    return results

def create_enhanced_visualization(all_results):
    """Create enhanced visualization with automatic interpretation"""
    print("\n🎨 Creating enhanced visualization with automatic interpretation...")
    
    ball_sizes = list(all_results.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('ENHANCED Energy Loss Analysis: Automatic Result Interpretation', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    water_colors = {'half': 'lightblue', 'full': 'darkblue'}
    
    for i, (ball_size, results) in enumerate(all_results.items()):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        ax2 = ax.twinx()
        
        if results is None:
            ax.text(0.5, 0.5, f'{ball_size}\nNo Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='red')
            ax.set_title(f'{ball_size} - No Data Available')
            continue
        
        # Plot with automatic interpretation
        for water_content, water_data in results.items():
            if not water_data or 'pattern_analysis' not in water_data:
                continue
                
            pattern_analysis = water_data['pattern_analysis']
            if pattern_analysis is None:
                continue
                
            frequencies = water_data['frequencies']
            energy_losses = water_data['energy_losses']
            harmonic_gains = water_data['harmonic_gains']
            
            # Plot main lines
            ax.plot(frequencies, energy_losses, 'o-', 
                   color=water_colors.get(water_content, 'black'),
                   linewidth=2, markersize=6,
                   label=f'{water_content} water - Energy Loss')
            
            ax2.plot(frequencies, harmonic_gains, 's--', 
                    color=water_colors.get(water_content, 'black'),
                    linewidth=2, markersize=4, alpha=0.7,
                    label=f'{water_content} water - Harmonic Gain')
            
            # Mark optimal frequencies
            for opt_freq in pattern_analysis['optimal_frequencies']:
                ax.plot(opt_freq['frequency'], opt_freq['energy_loss'], 'x', 
                       color=water_colors.get(water_content, 'black'),
                       markersize=12, markeredgewidth=3)
            
            # Mark problematic frequencies
            for prob_freq in pattern_analysis['problematic_frequencies']:
                ax.plot(prob_freq['frequency'], prob_freq['energy_loss'], '^', 
                       color=water_colors.get(water_content, 'black'),
                       markersize=12, markeredgewidth=3)
        
        ax.set_xlabel('Base Frequency (Hz)')
        ax.set_ylabel('Energy Loss (%)', color='blue')
        ax2.set_ylabel('Harmonic Gain (%)', color='red')
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        ax.set_title(f'{ball_size} - Enhanced Analysis')
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def generate_enhanced_analysis_report(all_results):
    """Generate enhanced analysis report with automatic interpretation"""
    print("\n🧠 ENHANCED ENERGY LOSS ANALYSIS REPORT")
    print("=" * 80)
    
    # Collect all analysis data
    analysis_summary = []
    
    for ball_size, results in all_results.items():
        if results is None:
            continue
            
        print(f"\n🔬 {ball_size} ENHANCED ANALYSIS:")
        print("-" * 40)
        
        for water_content, water_data in results.items():
            if not water_data or 'pattern_analysis' not in water_data:
                continue
                
            pattern_analysis = water_data['pattern_analysis']
            if pattern_analysis is None:
                continue
            
            print(f"\n💧 {water_content.upper()} WATER:")
            print(f"   Pattern Type: {pattern_analysis['pattern_type']}")
            print(f"   Energy Loss Range: {pattern_analysis['min_loss']:.1f}% to {pattern_analysis['max_loss']:.1f}%")
            print(f"   Average Energy Loss: {pattern_analysis['mean_loss']:.1f}% ± {pattern_analysis['std_loss']:.1f}%")
            
            # Print scientific interpretation
            interpretation = pattern_analysis['interpretation']
            print(f"\n   🔬 SCIENTIFIC INTERPRETATION:")
            print(f"   {interpretation['physics_explanation'].strip()}")
            print(f"   Damping Effectiveness: {interpretation['damping_effectiveness']}")
            print(f"   {interpretation['optimal_conditions']}")
            print(f"   Practical Applications: {interpretation['practical_applications']}")
            
            if interpretation['warnings']:
                print(f"   ⚠️  WARNINGS:")
                for warning in interpretation['warnings']:
                    print(f"      - {warning}")
            
            # Store for overall ranking
            analysis_summary.append({
                'ball_size': ball_size,
                'water_content': water_content,
                'pattern_type': pattern_analysis['pattern_type'],
                'max_loss': pattern_analysis['max_loss'],
                'min_loss': pattern_analysis['min_loss'],
                'mean_loss': pattern_analysis['mean_loss'],
                'damping_effectiveness': interpretation['damping_effectiveness'],
                'optimal_frequencies': pattern_analysis['optimal_frequencies']
            })
    
    # Find overall optimal conditions
    if analysis_summary:
        print(f"\n🏆 OVERALL OPTIMAL CONDITIONS:")
        print("=" * 50)
        
        # Sort by maximum energy loss
        analysis_summary.sort(key=lambda x: x['max_loss'], reverse=True)
        
        for i, condition in enumerate(analysis_summary[:5]):  # Top 5
            print(f"{i+1}. {condition['ball_size']} {condition['water_content']} water")
            print(f"   Pattern: {condition['pattern_type']}")
            print(f"   Max Energy Loss: {condition['max_loss']:.1f}%")
            print(f"   Min Energy Loss: {condition['min_loss']:.1f}%")
            print(f"   Average Energy Loss: {condition['mean_loss']:.1f}%")
            print(f"   Damping Effectiveness: {condition['damping_effectiveness']}")
            if condition['optimal_frequencies']:
                opt_freqs = [f"{opt['frequency']:.1f} Hz" for opt in condition['optimal_frequencies']]
                print(f"   Optimal Frequencies: {', '.join(opt_freqs)}")
            print()
    
    return analysis_summary

# Execute enhanced analysis
print("🚀 Starting enhanced energy loss analysis...")

# Use existing data structure
# data_structure should be loaded from previous cell

# Analyze all ball sizes with enhanced interpretation
all_results = {}
ball_sizes = ['10 mm', '30 mm', '47.5 mm', '65 mm', '82.5 mm', '100 mm']

for ball_size in ball_sizes:
    results = analyze_energy_loss_by_water_content_enhanced(ball_size, data_structure)
    all_results[ball_size] = results

# Create enhanced visualizations
if any(results is not None for results in all_results.values()):
    create_enhanced_visualization(all_results)
    
    # Generate enhanced report
    analysis_summary = generate_enhanced_analysis_report(all_results)
    
    print("\n✅ Enhanced energy loss analysis complete!")
    print("🧠 Key features:")
    print("  - Automatic pattern detection and interpretation")
    print("  - Scientific physics explanations")
    print("  - Optimal condition identification")
    print("  - Practical application guidance")
    print("  - Warning system for problematic configurations")
else:
    print("❌ No data available for analysis")
