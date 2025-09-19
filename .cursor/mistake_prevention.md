# Mistake Prevention Guide for AI Assistants

## 🚨 CRITICAL MISTAKES TO AVOID

### 1. The Mechanical Coupling Mistake
**MISTAKE**: Analyzing frequency changes in mechanically coupled systems
```python
# WRONG: Looking for frequency changes
if detected_freq != expected_freq:
    damping = calculate_from_frequency_deviation()

# CORRECT: Analyzing waveform distortion
if mechanically_coupled_system:
    damping = calculate_from_waveform_distortion()
```

**Prevention Rule**: Always check if system is mechanically constrained before analyzing frequency changes.

### 2. The Mock Data Mistake
**MISTAKE**: Using synthetic data for testing or demonstration
```python
# WRONG: Synthetic data
test_signal = np.sin(2 * np.pi * 16 * t) + noise

# CORRECT: Real experimental data
df = pd.read_csv('10mm16Hz2Adry.csv', header=None)
real_signal = process_experimental_data(df)
```

**Prevention Rule**: Always verify data source. Show exact file names and data point counts.

### 3. The Oversimplification Mistake
**MISTAKE**: Using single-frequency models for complex fluid-structure systems
```python
# WRONG: Oversimplified
model = A * sin(omega * t + phi)

# CORRECT: Multi-component with physical meaning
model = Shaker_Drive + Water_Sloshing + Friction_Effects
```

**Prevention Rule**: Start simple, but add complexity when simple models fail (R² < 0.5).

### 4. The Generic Label Mistake
**MISTAKE**: Using generic component labels
```python
# WRONG: Generic labels
labels = ['Component 1', 'Component 2', 'Component 3']

# CORRECT: Physical interpretation
labels = ['Shaker Drive', 'Water Sloshing', 'Ball-Surface Friction']
```

**Prevention Rule**: Every mathematical component needs physical interpretation.

### 5. The Visualization Mistake
**MISTAKE**: Charts in separate windows, not inline
```python
# WRONG: Separate window
plt.show()  # Without %matplotlib inline

# CORRECT: Inline display
%matplotlib inline
plt.show()
```

**Prevention Rule**: Always use `%matplotlib inline` for Jupyter notebooks.

## 🔍 DETECTION STRATEGIES

### 1. The "Physics Check"
**Question**: Does this make physical sense?
```python
def physics_check(result):
    """Check if results are physically reasonable"""
    # Frequency ratios should be reasonable
    assert 0.1 < freq_ratio < 10, "Unphysical frequency ratio"
    
    # Damping should be positive and finite
    assert 0 < damping < 100, "Unphysical damping coefficient"
    
    # Energy should be conserved
    assert energy_balance < 0.1, "Energy not conserved"
```

### 2. The "User Expertise Check"
**Question**: Is the user telling me something important?
```python
def user_expertise_check(user_feedback):
    """Check if user is providing domain expertise"""
    if "must follow" in user_feedback.lower():
        return "MECHANICAL_CONSTRAINT_INSIGHT"
    if "oversimplification" in user_feedback.lower():
        return "COMPLEXITY_INSIGHT"
    if "real data" in user_feedback.lower():
        return "DATA_VALIDATION_INSIGHT"
```

### 3. The "Model Adequacy Check"
**Question**: Is the model adequate for the system?
```python
def model_adequacy_check(r2_score, system_complexity):
    """Check if model complexity matches system"""
    if system_complexity == "fluid_structure" and r2_score < 0.5:
        return "NEED_MULTI_COMPONENT_MODEL"
    if system_complexity == "simple_oscillator" and r2_score > 0.9:
        return "MODEL_ADEQUATE"
```

## 🛡️ PREVENTION MECHANISMS

### 1. Input Validation Protocol
```python
def validate_inputs(data, expected_freq, system_type):
    """Comprehensive input validation"""
    # Data validation
    assert len(data) > 0, "Empty data"
    assert not np.isnan(data).any(), "Data contains NaN"
    
    # System validation
    assert system_type in ["simple", "coupled", "fluid_structure"], "Unknown system type"
    
    # Frequency validation
    assert 0 < expected_freq < 1000, "Unreasonable frequency"
```

### 2. Result Validation Protocol
```python
def validate_results(results):
    """Validate analysis results"""
    # Technical validation
    assert all(r.r2 > 0.3 for r in results), "Poor fit quality"
    assert all(r.nrmse < 50 for r in results), "High normalized error"
    
    # Physical validation
    assert all(0 < r.damping < 100 for r in results), "Unphysical damping"
    assert all(0.1 < r.freq_ratio < 10 for r in results), "Unphysical frequency ratio"
```

### 3. User Feedback Integration Protocol
```python
def integrate_user_feedback(feedback, current_approach):
    """Systematically integrate user feedback"""
    if "oversimplification" in feedback:
        return upgrade_to_multi_component()
    if "real data" in feedback:
        return verify_data_sources()
    if "mechanical coupling" in feedback:
        return implement_waveform_distortion_analysis()
```

## 🎯 QUALITY ASSURANCE CHECKLIST

### Before Starting Analysis:
- [ ] Verified data source (real CSV files)
- [ ] Understood system physics (coupled vs. free)
- [ ] Chose appropriate model complexity
- [ ] Set up inline visualization

### During Analysis:
- [ ] Validated intermediate results
- [ ] Checked physical reasonableness
- [ ] Listened to user feedback
- [ ] Documented assumptions

### After Analysis:
- [ ] Verified all components have physical meaning
- [ ] Validated energy conservation
- [ ] Confirmed inline chart display
- [ ] Summarized key insights

## 🚨 RED FLAGS TO WATCH FOR

### 1. Technical Red Flags
```python
# High error rates
if nrmse > 20:
    raise ValueError("Poor model fit - need more complex model")

# Unphysical parameters
if damping < 0 or damping > 100:
    raise ValueError("Unphysical damping coefficient")

# Energy not conserved
if abs(energy_balance) > 0.1:
    raise ValueError("Energy not conserved")
```

### 2. User Feedback Red Flags
```python
# User questioning approach
if "oversimplification" in user_feedback:
    return "UPGRADE_MODEL_COMPLEXITY"

# User providing domain expertise
if "must follow" in user_feedback:
    return "IMPLEMENT_PHYSICAL_CONSTRAINTS"

# User questioning data source
if "real data" in user_feedback:
    return "VERIFY_DATA_SOURCES"
```

### 3. System Behavior Red Flags
```python
# Poor fit quality
if r2_score < 0.5:
    return "NEED_MORE_COMPLEX_MODEL"

# Inconsistent results across frequencies
if results_inconsistent():
    return "CHECK_ANALYSIS_METHODOLOGY"

# Missing physical interpretation
if not has_physical_meaning(components):
    return "ADD_PHYSICAL_INTERPRETATION"
```

## 🔧 RECOVERY STRATEGIES

### 1. When Model is Too Simple
```python
def upgrade_model_complexity(simple_results):
    """Upgrade to multi-component model"""
    if simple_results.r2 < 0.5:
        return implement_multi_component_analysis()
    else:
        return simple_results
```

### 2. When Data Source is Questioned
```python
def verify_data_sources():
    """Verify all data sources are real"""
    for filename in data_files:
        assert file_exists(filename), f"File {filename} not found"
        data = load_data(filename)
        assert not is_synthetic(data), f"File {filename} appears synthetic"
```

### 3. When Physics is Wrong
```python
def correct_physics_approach(system_type):
    """Correct physics approach based on system type"""
    if system_type == "mechanically_coupled":
        return implement_waveform_distortion_analysis()
    else:
        return implement_frequency_analysis()
```

## 📚 LEARNING FROM MISTAKES

### 1. Document Mistakes
```python
def document_mistake(mistake_type, correction, lesson):
    """Document mistakes for future prevention"""
    mistake_log = {
        "type": mistake_type,
        "correction": correction,
        "lesson": lesson,
        "timestamp": datetime.now()
    }
    save_to_lesson_learned(mistake_log)
```

### 2. Pattern Recognition
```python
def recognize_mistake_patterns(history):
    """Recognize patterns in past mistakes"""
    common_patterns = [
        "oversimplification_for_complex_systems",
        "ignoring_mechanical_constraints", 
        "using_mock_data",
        "poor_visualization"
    ]
    return identify_patterns(history, common_patterns)
```

### 3. Prevention Updates
```python
def update_prevention_rules(new_insights):
    """Update prevention rules based on new insights"""
    for insight in new_insights:
        add_prevention_rule(insight)
        update_checklist(insight)
```

---
*Comprehensive mistake prevention guide for AI assistants*
