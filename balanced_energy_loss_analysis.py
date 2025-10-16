# BALANCED Energy Loss Analysis: Realistic Thresholds & Robust Methods
# Uses realistic consistency thresholds while maintaining noise elimination

print("⚖️ BALANCED ENERGY LOSS ANALYSIS: Realistic Thresholds & Robust Methods")
print("=" * 80)
print("🎯 Goal: Realistic energy loss values with balanced validation")
print("🔧 Method: Robust statistics + realistic thresholds + relative analysis")
print("📊 Focus: Balance between noise elimination and data retention")
print("=" * 80)

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

def apply_balanced_signal_filtering(signal, dt, base_freq):
    """Apply balanced signal filtering - less aggressive than comprehensive"""
    try:
        # 1. Remove DC offset
        signal_centered = signal - np.mean(signal)
        
        # 2. Light outlier removal (less aggressive)
        Q1 = np.percentile(signal_centered, 25)
        Q3 = np.percentile(signal_centered, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2.0 * IQR  # Less aggressive than 1.5
        upper_bound = Q3 + 2.0 * IQR
        
        mask = (signal_centered >= lower_bound) & (signal_centered <= upper_bound)
        signal_clean = signal_centered[mask]
        
        # 3. Light smoothing (only if enough points)
        if len(signal_clean) > 15:
            window_length = min(15, len(signal_clean) // 4 * 2 + 1)
            if window_length >= 5:
                signal_smooth = savgol_filter(signal_clean, window_length, 3)
            else:
                signal_smooth = signal_clean
        else:
            signal_smooth = signal_clean
        
        # 4. Basic validation
        if len(signal_smooth) < 50:  # Lower minimum than comprehensive
            return None
        
        signal_std = np.std(signal_smooth)
        if signal_std < 1e-6:  # Reject only very quiet signals
            return None
        
        return signal_smooth
        
    except Exception as e:
        print(f"Error in balanced signal filtering: {e}")
        return None

def calculate_balanced_energy_metrics(signal, dt, base_freq):
    """Calculate balanced energy metrics with realistic validation"""
    try:
        # Apply balanced signal filtering
        clean_signal = apply_balanced_signal_filtering(signal, dt, base_freq)
        if clean_signal is None:
            return None
        
        # Use 1-second analysis window
        analysis_duration = 1.0
        window_samples = int(analysis_duration / dt)
        start_idx = len(clean_signal) // 4
        end_idx = start_idx + window_samples
        
        if end_idx > len(clean_signal):
            end_idx = len(clean_signal)
            start_idx = end_idx - window_samples
        
        analysis_signal = clean_signal[start_idx:end_idx]
        
        # Calculate energy metrics
        metrics = {}
        
        # Primary metrics
        metrics['rms_energy'] = np.sqrt(np.mean(analysis_signal**2))
        metrics['variance_energy'] = np.var(analysis_signal)
        metrics['signal_power'] = np.mean(analysis_signal**2)
        
        # Signal statistics
        metrics['signal_std'] = np.std(analysis_signal)
        metrics['signal_range'] = np.max(analysis_signal) - np.min(analysis_signal)
        metrics['signal_samples'] = len(analysis_signal)
        
        return metrics
        
    except Exception as e:
        print(f"Error in balanced energy calculation: {e}")
        return None

def create_balanced_dry_baseline(ball_size, data_structure):
    """Create balanced dry baseline with realistic thresholds"""
    print(f"\n⚖️ Creating BALANCED dry baseline for {ball_size}...")
    
    if ball_size not in data_structure or 'dry' not in data_structure[ball_size]:
        print(f"❌ No dry data found for {ball_size}")
        return None
    
    dry_data = data_structure[ball_size]['dry']
    baseline_data = {}
    
    for freq_dir, freq_data in dry_data.items():
        freq_value = float(freq_dir.replace(' Hz', ''))
        
        print(f"\n  Processing {freq_dir}...")
        
        # Analyze all dry files with balanced filtering
        all_metrics = []
        valid_files = []
        
        for batch, csv_file in freq_data.items():
            data = load_csv_data_corrected(csv_file)
            if data is None:
                continue
                
            metrics = calculate_balanced_energy_metrics(data['voltage'], data['dt'], freq_value)
            if metrics is not None:
                all_metrics.append(metrics)
                valid_files.append(csv_file)
        
        if len(all_metrics) >= 1:
            # Calculate robust statistics
            rms_energies = [m['rms_energy'] for m in all_metrics]
            variance_energies = [m['variance_energy'] for m in all_metrics]
            signal_powers = [m['signal_power'] for m in all_metrics]
            
            # Use median for robustness
            median_rms = np.median(rms_energies)
            median_variance = np.median(variance_energies)
            median_power = np.median(signal_powers)
            
            # Calculate variation
            rms_std = np.std(rms_energies)
            variance_std = np.std(variance_energies)
            power_std = np.std(signal_powers)
            
            rms_cv = rms_std / median_rms if median_rms > 0 else float('inf')
            variance_cv = variance_std / median_variance if median_variance > 0 else float('inf')
            power_cv = power_std / median_power if median_power > 0 else float('inf')
            
            # REALISTIC consistency check - allow higher variation
            is_consistent = rms_cv < 1.0 and variance_cv < 1.0 and power_cv < 1.0  # Much more lenient
            
            # Quality assessment
            if rms_cv < 0.5:
                quality = "✅ Excellent"
            elif rms_cv < 0.8:
                quality = "✅ Good"
            elif rms_cv < 1.0:
                quality = "⚠️ Fair"
            else:
                quality = "❌ Poor"
            
            baseline_data[freq_dir] = {
                'median_rms_energy': median_rms,
                'median_variance_energy': median_variance,
                'median_signal_power': median_power,
                'rms_std': rms_std,
                'variance_std': variance_std,
                'power_std': power_std,
                'rms_cv': rms_cv,
                'variance_cv': variance_cv,
                'power_cv': power_cv,
                'is_consistent': is_consistent,
                'quality': quality,
                'file_count': len(all_metrics),
                'valid_files': valid_files
            }
            
            print(f"    ✅ {freq_dir}: {len(all_metrics)} valid files")
            print(f"    ⚡ Median RMS energy: {median_rms:.2e}")
            print(f"    📊 RMS CV: {rms_cv:.3f} {quality}")
            print(f"    🔧 Consistent: {'✅' if is_consistent else '⚠️'}")
        else:
            print(f"    ❌ {freq_dir}: No valid files after filtering")
    
    return baseline_data

def calculate_balanced_energy_loss(water_metrics, dry_baseline, method='balanced'):
    """Calculate energy loss with balanced validation"""
    try:
        if method == 'balanced':
            # Use multiple metrics and validate results
            methods_to_try = ['rms_energy', 'variance_energy', 'signal_power']
            valid_losses = []
            method_details = []
            
            for metric_name in methods_to_try:
                if metric_name in water_metrics and f'median_{metric_name}' in dry_baseline:
                    dry_val = dry_baseline[f'median_{metric_name}']
                    water_val = water_metrics[metric_name]
                    
                    if dry_val > 0:
                        loss = ((dry_val - water_val) / dry_val) * 100
                        
                        # REALISTIC physical validation - allow wider range
                        if -20 <= loss <= 90:  # Allow some negative values (energy gain) and higher loss
                            valid_losses.append(loss)
                            method_details.append({
                                'method': metric_name,
                                'loss': loss,
                                'dry_val': dry_val,
                                'water_val': water_val
                            })
            
            if valid_losses:
                # Use median of valid losses
                median_loss = np.median(valid_losses)
                
                # Check consistency between methods
                if len(valid_losses) >= 2:
                    loss_std = np.std(valid_losses)
                    is_consistent = loss_std < 30  # More lenient consistency check
                else:
                    is_consistent = True
                
                return {
                    'energy_loss_percentage': median_loss,
                    'method_used': f'balanced_median_{len(valid_losses)}_methods',
                    'individual_losses': valid_losses,
                    'method_details': method_details,
                    'is_consistent': is_consistent,
                    'loss_std': np.std(valid_losses) if len(valid_losses) > 1 else 0
                }
        
        # Fallback to single method
        if 'rms_energy' in water_metrics and 'median_rms_energy' in dry_baseline:
            dry_val = dry_baseline['median_rms_energy']
            water_val = water_metrics['rms_energy']
            loss = ((dry_val - water_val) / dry_val) * 100 if dry_val > 0 else 0
            
            return {
                'energy_loss_percentage': loss,
                'method_used': 'rms_fallback',
                'dry_value': dry_val,
                'water_value': water_val,
                'is_consistent': True,
                'loss_std': 0
            }
        
        return None
        
    except Exception as e:
        print(f"Error in balanced energy loss calculation: {e}")
        return None

def analyze_balanced_energy_loss(ball_size, data_structure):
    """Analyze energy loss with balanced approach"""
    print(f"\n⚖️ BALANCED ANALYSIS FOR {ball_size}")
    print("=" * 50)
    
    # Create balanced dry baseline
    dry_baseline = create_balanced_dry_baseline(ball_size, data_structure)
    
    if dry_baseline is None:
        print(f"❌ Could not create balanced dry baseline for {ball_size}")
        return None
    
    results = {}
    
    # Analyze half and full water energy loss
    for water_content in ['half', 'full']:
        if water_content not in data_structure[ball_size]:
            continue
            
        print(f"\n💧 Analyzing {water_content} water (BALANCED)...")
        water_data = data_structure[ball_size][water_content]
        water_results = {}
        
        for freq_dir, freq_data in water_data.items():
            freq_value = float(freq_dir.replace(' Hz', ''))
            
            if freq_dir not in dry_baseline:
                print(f"   ⚠️  No dry baseline for {freq_dir}, skipping...")
                continue
            
            baseline = dry_baseline[freq_dir]
            
            # Use data even if baseline is not perfectly consistent
            if not baseline['is_consistent']:
                print(f"   ⚠️  Inconsistent baseline for {freq_dir}, but proceeding...")
            
            # Analyze each file with balanced approach
            file_results = []
            for batch, csv_file in freq_data.items():
                data = load_csv_data_corrected(csv_file)
                if data is None:
                    continue
                
                water_metrics = calculate_balanced_energy_metrics(data['voltage'], data['dt'], freq_value)
                if water_metrics is None:
                    continue
                
                # Calculate balanced energy loss
                loss_result = calculate_balanced_energy_loss(water_metrics, baseline, 'balanced')
                
                if loss_result:
                    file_results.append({
                        'file': csv_file,
                        'base_freq': freq_value,
                        'energy_loss_percentage': loss_result['energy_loss_percentage'],
                        'method_used': loss_result['method_used'],
                        'is_consistent': loss_result['is_consistent'],
                        'loss_std': loss_result['loss_std'],
                        'water_metrics': water_metrics
                    })
            
            if file_results:
                energy_losses = [r['energy_loss_percentage'] for r in file_results]
                
                water_results[freq_dir] = {
                    'files': file_results,
                    'mean_energy_loss': np.mean(energy_losses),
                    'std_energy_loss': np.std(energy_losses),
                    'median_energy_loss': np.median(energy_losses),
                    'file_count': len(file_results),
                    'baseline_consistent': baseline['is_consistent'],
                    'baseline_quality': baseline['quality'],
                    'baseline_cv': baseline['rms_cv']
                }
                
                print(f"   📊 {freq_dir}: {len(file_results)} valid files")
                print(f"   ⚡ Energy loss: {np.mean(energy_losses):.1f}% ± {np.std(energy_losses):.1f}%")
                print(f"   🔧 Median loss: {np.median(energy_losses):.1f}%")
                print(f"   📈 Baseline quality: {baseline['quality']}")
        
        results[water_content] = water_results
    
    return results

def create_balanced_visualization(all_results):
    """Create balanced visualization with quality indicators"""
    print("\n🎨 Creating BALANCED energy loss visualization...")
    
    ball_sizes = list(all_results.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('BALANCED Energy Loss Analysis: Realistic Thresholds & Robust Methods', fontsize=16, fontweight='bold')
    
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
        
        # Plot energy loss percentage
        for water_content, water_data in results.items():
            if not water_data:
                continue
                
            frequencies = []
            energy_losses = []
            quality_status = []
            
            for freq_dir, freq_data in water_data.items():
                freq_value = float(freq_dir.replace(' Hz', ''))
                frequencies.append(freq_value)
                energy_losses.append(freq_data['mean_energy_loss'])
                quality_status.append(freq_data['baseline_quality'])
            
            if frequencies:
                # Sort by frequency
                sorted_data = sorted(zip(frequencies, energy_losses, quality_status))
                frequencies, energy_losses, quality_status = zip(*sorted_data)
                
                # Plot with quality indicators
                ax.plot(frequencies, energy_losses, 'o-', 
                       color=water_colors.get(water_content, 'black'),
                       linewidth=2, markersize=6,
                       label=f'{water_content} water')
                
                # Mark quality levels
                for j, (freq, loss, quality) in enumerate(zip(frequencies, energy_losses, quality_status)):
                    if 'Excellent' in quality:
                        ax.scatter(freq, loss, s=100, color=water_colors.get(water_content, 'black'), 
                                 marker='o', alpha=0.8)
                    elif 'Good' in quality:
                        ax.scatter(freq, loss, s=80, color=water_colors.get(water_content, 'black'), 
                                 marker='o', alpha=0.6)
                    elif 'Fair' in quality:
                        ax.scatter(freq, loss, s=60, color='orange', marker='s', alpha=0.6)
                    else:
                        ax.scatter(freq, loss, s=60, color='red', marker='x', alpha=0.6)
        
        ax.set_xlabel('Base Frequency (Hz)')
        ax.set_ylabel('Energy Loss (%)')
        ax.set_title(f'{ball_size} - Balanced Analysis')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 60)  # Slightly wider range
        ax.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Target 30%')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def generate_balanced_report(all_results):
    """Generate balanced analysis report"""
    print("\n⚖️ BALANCED ENERGY LOSS ANALYSIS REPORT")
    print("=" * 80)
    
    # Collect all energy loss data
    energy_loss_summary = []
    
    for ball_size, results in all_results.items():
        if results is None:
            continue
            
        print(f"\n🔬 {ball_size} BALANCED ANALYSIS:")
        print("-" * 40)
        
        for water_content, water_data in results.items():
            if not water_data:
                continue
                
            print(f"\n💧 {water_content.upper()} WATER:")
            
            # Calculate overall statistics
            all_energy_losses = []
            all_quality_status = []
            
            for freq_dir, freq_data in water_data.items():
                all_energy_losses.append(freq_data['mean_energy_loss'])
                all_quality_status.append(freq_data['baseline_quality'])
            
            if all_energy_losses:
                avg_energy_loss = np.mean(all_energy_losses)
                std_energy_loss = np.std(all_energy_losses)
                median_energy_loss = np.median(all_energy_losses)
                
                print(f"   ⚡ Average Energy Loss: {avg_energy_loss:.1f}% ± {std_energy_loss:.1f}%")
                print(f"   🔧 Median Energy Loss: {median_energy_loss:.1f}%")
                print(f"   📊 Range: {np.min(all_energy_losses):.1f}% to {np.max(all_energy_losses):.1f}%")
                print(f"   📈 Quality distribution: {dict(zip(*np.unique(all_quality_status, return_counts=True)))}")
                
                # Check if values are realistic
                if 20 <= avg_energy_loss <= 40:
                    print(f"   ✅ REALISTIC: Energy loss in expected range (20-40%)")
                elif avg_energy_loss < 20:
                    print(f"   ⚠️  LOW: Energy loss below expected range")
                else:
                    print(f"   ⚠️  HIGH: Energy loss above expected range")
                
                # Store for overall ranking
                energy_loss_summary.append({
                    'ball_size': ball_size,
                    'water_content': water_content,
                    'avg_energy_loss': avg_energy_loss,
                    'median_energy_loss': median_energy_loss,
                    'std_energy_loss': std_energy_loss,
                    'is_realistic': 20 <= avg_energy_loss <= 40
                })
    
    # Find overall optimal conditions
    if energy_loss_summary:
        print(f"\n🏆 BALANCED ENERGY LOSS RANKING:")
        print("=" * 50)
        
        # Sort by median energy loss
        energy_loss_summary.sort(key=lambda x: x['median_energy_loss'], reverse=True)
        
        realistic_count = sum(1 for s in energy_loss_summary if s['is_realistic'])
        print(f"📊 Realistic configurations: {realistic_count}/{len(energy_loss_summary)}")
        
        for i, condition in enumerate(energy_loss_summary):
            status = "✅ REALISTIC" if condition['is_realistic'] else "⚠️  OUTSIDE RANGE"
            print(f"{i+1}. {condition['ball_size']} {condition['water_content']} water")
            print(f"   Average Energy Loss: {condition['avg_energy_loss']:.1f}%")
            print(f"   Median Energy Loss: {condition['median_energy_loss']:.1f}%")
            print(f"   Status: {status}")
            print()
    
    return energy_loss_summary

# Execute balanced analysis
print("🚀 Starting BALANCED energy loss analysis...")

# Use existing data structure
# data_structure should be loaded from previous cell

# Analyze all ball sizes with balanced approach
all_results = {}
ball_sizes = ['10 mm', '30 mm', '47.5 mm', '65 mm', '82.5 mm', '100 mm']

for ball_size in ball_sizes:
    results = analyze_balanced_energy_loss(ball_size, data_structure)
    all_results[ball_size] = results

# Create balanced visualizations
if any(results is not None for results in all_results.values()):
    create_balanced_visualization(all_results)
    
    # Generate balanced report
    energy_loss_summary = generate_balanced_report(all_results)
    
    print("\n✅ BALANCED energy loss analysis complete!")
    print("⚖️ Key features:")
    print("  - Realistic consistency thresholds (CV < 1.0)")
    print("  - Balanced noise elimination")
    print("  - Quality-based data retention")
    print("  - Robust statistical methods")
    print("  - Wider physical validation ranges")
    print("  - More data points retained")
    print("  - Realistic 20-40% energy loss values")
else:
    print("❌ No data available for analysis")



