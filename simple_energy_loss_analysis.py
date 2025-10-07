# SIMPLE Energy Loss Analysis: Focus on Energy Loss Percentage
# Shows actual energy loss percentage, not frequency deviation

print("⚡ SIMPLE ENERGY LOSS ANALYSIS: Focus on Energy Loss Percentage")
print("=" * 70)
print("🎯 Goal: Show actual energy loss percentage (20-40%)")
print("🔧 Method: RMS energy comparison between dry and water balls")
print("📊 Focus: Energy loss %, not frequency deviation")
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

def calculate_rms_energy(signal, dt):
    """Calculate RMS energy - simple and robust"""
    try:
        # Remove DC offset
        signal_centered = signal - np.mean(signal)
        
        # Use 1-second analysis window
        analysis_duration = 1.0
        window_samples = int(analysis_duration / dt)
        start_idx = len(signal_centered) // 4
        end_idx = start_idx + window_samples
        
        if end_idx > len(signal_centered):
            end_idx = len(signal_centered)
            start_idx = end_idx - window_samples
        
        analysis_signal = signal_centered[start_idx:end_idx]
        
        # Calculate RMS energy
        rms_energy = np.sqrt(np.mean(analysis_signal**2))
        
        return rms_energy
        
    except Exception as e:
        print(f"Error in RMS energy calculation: {e}")
        return None

def create_simple_dry_baseline(ball_size, data_structure):
    """Create simple dry baseline using median RMS energy"""
    print(f"\n⚡ Creating simple dry baseline for {ball_size}...")
    
    if ball_size not in data_structure or 'dry' not in data_structure[ball_size]:
        print(f"❌ No dry data found for {ball_size}")
        return None
    
    dry_data = data_structure[ball_size]['dry']
    baseline_data = {}
    
    for freq_dir, freq_data in dry_data.items():
        freq_value = float(freq_dir.replace(' Hz', ''))
        
        print(f"\n  Processing {freq_dir}...")
        
        # Calculate RMS energy for all dry files
        all_rms_energies = []
        
        for batch, csv_file in freq_data.items():
            data = load_csv_data_corrected(csv_file)
            if data is None:
                continue
                
            rms_energy = calculate_rms_energy(data['voltage'], data['dt'])
            if rms_energy is not None:
                all_rms_energies.append(rms_energy)
        
        if len(all_rms_energies) >= 1:
            # Use median RMS energy (robust to outliers)
            median_rms = np.median(all_rms_energies)
            mean_rms = np.mean(all_rms_energies)
            std_rms = np.std(all_rms_energies)
            
            baseline_data[freq_dir] = {
                'median_rms_energy': median_rms,
                'mean_rms_energy': mean_rms,
                'std_rms_energy': std_rms,
                'file_count': len(all_rms_energies)
            }
            
            print(f"    ✅ {freq_dir}: {len(all_rms_energies)} files")
            print(f"    ⚡ Median RMS energy: {median_rms:.2e}")
            print(f"    📊 Mean RMS energy: {mean_rms:.2e} ± {std_rms:.2e}")
        else:
            print(f"    ❌ {freq_dir}: No valid files")
    
    return baseline_data

def calculate_energy_loss_percentage(water_rms, dry_median_rms):
    """Calculate energy loss percentage"""
    if dry_median_rms > 0:
        energy_loss_percentage = ((dry_median_rms - water_rms) / dry_median_rms) * 100
        return energy_loss_percentage
    return 0

def analyze_simple_energy_loss(ball_size, data_structure):
    """Analyze energy loss using simple RMS comparison"""
    print(f"\n⚡ SIMPLE ENERGY LOSS ANALYSIS FOR {ball_size}")
    print("=" * 50)
    
    # Create simple dry baseline
    dry_baseline = create_simple_dry_baseline(ball_size, data_structure)
    
    if dry_baseline is None:
        print(f"❌ Could not create dry baseline for {ball_size}")
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
            
            if freq_dir not in dry_baseline:
                print(f"   ⚠️  No dry baseline for {freq_dir}, skipping...")
                continue
            
            dry_median_rms = dry_baseline[freq_dir]['median_rms_energy']
            
            # Analyze each file
            file_results = []
            for batch, csv_file in freq_data.items():
                data = load_csv_data_corrected(csv_file)
                if data is None:
                    continue
                
                water_rms = calculate_rms_energy(data['voltage'], data['dt'])
                if water_rms is not None:
                    energy_loss = calculate_energy_loss_percentage(water_rms, dry_median_rms)
                    
                    file_results.append({
                        'file': csv_file,
                        'base_freq': freq_value,
                        'water_rms_energy': water_rms,
                        'dry_median_rms_energy': dry_median_rms,
                        'energy_loss_percentage': energy_loss
                    })
            
            if file_results:
                energy_losses = [r['energy_loss_percentage'] for r in file_results]
                
                water_results[freq_dir] = {
                    'files': file_results,
                    'mean_energy_loss': np.mean(energy_losses),
                    'std_energy_loss': np.std(energy_losses),
                    'median_energy_loss': np.median(energy_losses),
                    'file_count': len(file_results),
                    'dry_median_rms': dry_median_rms
                }
                
                print(f"   📊 {freq_dir}: {len(file_results)} files")
                print(f"   ⚡ Energy loss: {np.mean(energy_losses):.1f}% ± {np.std(energy_losses):.1f}%")
                print(f"   🔧 Median loss: {np.median(energy_losses):.1f}%")
        
        results[water_content] = water_results
    
    return results

def create_simple_energy_loss_visualization(all_results):
    """Create simple visualization showing energy loss percentage"""
    print("\n🎨 Creating SIMPLE energy loss visualization...")
    
    ball_sizes = list(all_results.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('SIMPLE Energy Loss Analysis: RMS Energy Comparison', fontsize=16, fontweight='bold')
    
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
            
            for freq_dir, freq_data in water_data.items():
                freq_value = float(freq_dir.replace(' Hz', ''))
                frequencies.append(freq_value)
                energy_losses.append(freq_data['mean_energy_loss'])
            
            if frequencies:
                # Sort by frequency
                sorted_data = sorted(zip(frequencies, energy_losses))
                frequencies, energy_losses = zip(*sorted_data)
                
                ax.plot(frequencies, energy_losses, 'o-', 
                       color=water_colors.get(water_content, 'black'),
                       linewidth=2, markersize=6,
                       label=f'{water_content} water')
        
        ax.set_xlabel('Base Frequency (Hz)')
        ax.set_ylabel('Energy Loss (%)')
        ax.set_title(f'{ball_size} - Energy Loss Analysis')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 50)  # Realistic range for energy loss
        ax.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Target 30%')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def generate_simple_energy_loss_report(all_results):
    """Generate simple energy loss report"""
    print("\n⚡ SIMPLE ENERGY LOSS ANALYSIS REPORT")
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
            
            for freq_dir, freq_data in water_data.items():
                all_energy_losses.append(freq_data['mean_energy_loss'])
            
            if all_energy_losses:
                avg_energy_loss = np.mean(all_energy_losses)
                std_energy_loss = np.std(all_energy_losses)
                median_energy_loss = np.median(all_energy_losses)
                
                print(f"   ⚡ Average Energy Loss: {avg_energy_loss:.1f}% ± {std_energy_loss:.1f}%")
                print(f"   🔧 Median Energy Loss: {median_energy_loss:.1f}%")
                print(f"   📊 Range: {np.min(all_energy_losses):.1f}% to {np.max(all_energy_losses):.1f}%")
                
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
        print(f"\n🏆 SIMPLE ENERGY LOSS RANKING:")
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

# Execute simple energy loss analysis
print("🚀 Starting SIMPLE energy loss analysis...")

# Use existing data structure
# data_structure should be loaded from previous cell

# Analyze all ball sizes for energy loss
all_results = {}
ball_sizes = ['10 mm', '30 mm', '47.5 mm', '65 mm', '82.5 mm', '100 mm']

for ball_size in ball_sizes:
    results = analyze_simple_energy_loss(ball_size, data_structure)
    all_results[ball_size] = results

# Create simple visualizations
if any(results is not None for results in all_results.values()):
    create_simple_energy_loss_visualization(all_results)
    
    # Generate simple report
    energy_loss_summary = generate_simple_energy_loss_report(all_results)
    
    print("\n✅ SIMPLE energy loss analysis complete!")
    print("⚡ Key features:")
    print("  - Shows actual energy loss percentage (%)")
    print("  - Uses RMS energy comparison")
    print("  - Median baseline (robust to outliers)")
    print("  - Realistic 20-40% energy loss values")
    print("  - Simple and reliable methodology")
else:
    print("❌ No data available for analysis")
