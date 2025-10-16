# 🔧 FIXED FLUID DYNAMICS ANALYSIS
# Fixes the empty motion intensity chart and improves calculations

print("🔧 FIXED FLUID DYNAMICS ANALYSIS")
print("=" * 50)
print("✅ Fixed motion intensity calculation")
print("📊 Improved frame difference analysis")
print("🔬 Better scaling and visualization")
print("=" * 50)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from skimage import measure, filters
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_simulation_video_fixed(video_path):
    """Load simulation video and extract frames with better error handling"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return None
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_count += 1
            
            # Limit to reasonable number of frames for analysis
            if frame_count >= 50:
                break
        
        cap.release()
        
        print(f"✅ Loaded {len(frames)} frames from {Path(video_path).name}")
        return frames
        
    except Exception as e:
        print(f"❌ Error loading video {video_path}: {e}")
        return None

def analyze_fluid_motion_fixed(frames):
    """Fixed fluid motion analysis with proper scaling"""
    try:
        if not frames or len(frames) < 2:
            return None
        
        # Convert frames to grayscale for motion analysis
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
        
        # Calculate frame differences to detect motion
        motion_intensity = []
        flow_vectors = []
        
        for i in range(1, len(gray_frames)):
            # Calculate frame difference with proper scaling
            diff = cv2.absdiff(gray_frames[i-1], gray_frames[i])
            
            # Calculate motion intensity with better scaling
            motion_value = np.mean(diff) * 100  # Scale up for visibility
            motion_intensity.append(motion_value)
            
            # Calculate optical flow (simplified)
            try:
                # Use corner detection for optical flow
                corners = cv2.goodFeaturesToTrack(gray_frames[i-1], maxCorners=100, qualityLevel=0.01, minDistance=10)
                if corners is not None and len(corners) > 0:
                    flow, status, error = cv2.calcOpticalFlowPyrLK(
                        gray_frames[i-1], gray_frames[i], corners, None
                    )
                    if flow is not None and len(flow) > 0:
                        # Calculate average flow magnitude
                        flow_magnitude = np.mean(np.sqrt(np.sum((flow - corners)**2, axis=2)))
                        flow_vectors.append(flow_magnitude)
                    else:
                        flow_vectors.append(0)
                else:
                    flow_vectors.append(0)
            except:
                flow_vectors.append(0)
        
        # Calculate fluid dynamics metrics
        avg_motion = np.mean(motion_intensity)
        max_motion = np.max(motion_intensity)
        motion_variance = np.var(motion_intensity)
        
        avg_flow = np.mean(flow_vectors)
        max_flow = np.max(flow_vectors)
        
        print(f"🔬 Motion Analysis Results:")
        print(f"   Average motion intensity: {avg_motion:.2f}")
        print(f"   Max motion intensity: {max_motion:.2f}")
        print(f"   Motion variance: {motion_variance:.2f}")
        print(f"   Average flow velocity: {avg_flow:.2f}")
        print(f"   Max flow velocity: {max_flow:.2f}")
        
        return {
            'avg_motion_intensity': avg_motion,
            'max_motion_intensity': max_motion,
            'motion_variance': motion_variance,
            'avg_flow_velocity': avg_flow,
            'max_flow_velocity': max_flow,
            'motion_intensity': motion_intensity,
            'flow_vectors': flow_vectors,
            'frame_count': len(frames)
        }
        
    except Exception as e:
        print(f"❌ Error analyzing fluid motion: {e}")
        return None

def analyze_water_air_interface_fixed(frames):
    """Fixed water-air interface analysis"""
    try:
        if not frames:
            return None
        
        interface_metrics = []
        
        for frame in frames:
            # Convert to HSV for better color separation
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            
            # Define water and air color ranges (simplified)
            # Water: blue-ish colors
            water_lower = np.array([100, 50, 50])
            water_upper = np.array([130, 255, 255])
            
            # Air: lighter colors
            air_lower = np.array([0, 0, 200])
            air_upper = np.array([180, 30, 255])
            
            # Create masks
            water_mask = cv2.inRange(hsv, water_lower, water_upper)
            air_mask = cv2.inRange(hsv, air_lower, air_upper)
            
            # Calculate interface metrics
            water_area = np.sum(water_mask > 0)
            air_area = np.sum(air_mask > 0)
            total_area = water_area + air_area
            
            if total_area > 0:
                water_fraction = water_area / total_area
                interface_complexity = np.sum(cv2.Canny(water_mask, 50, 150)) / total_area
                
                interface_metrics.append({
                    'water_fraction': water_fraction,
                    'interface_complexity': interface_complexity,
                    'water_area': water_area,
                    'air_area': air_area
                })
        
        if interface_metrics:
            avg_water_fraction = np.mean([m['water_fraction'] for m in interface_metrics])
            avg_interface_complexity = np.mean([m['interface_complexity'] for m in interface_metrics])
            interface_variance = np.var([m['water_fraction'] for m in interface_metrics])
            
            return {
                'avg_water_fraction': avg_water_fraction,
                'avg_interface_complexity': avg_interface_complexity,
                'interface_variance': interface_variance,
                'interface_metrics': interface_metrics
            }
        
        return None
        
    except Exception as e:
        print(f"❌ Error analyzing water-air interface: {e}")
        return None

def create_fixed_fluid_dynamics_visualization(simulation_results, ball_size):
    """Create fixed fluid dynamics visualization with proper scaling"""
    try:
        if not simulation_results:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'FIXED Fluid Dynamics Analysis: {ball_size}', fontsize=16, fontweight='bold')
        
        # 1. Motion intensity over time (FIXED)
        ax1 = axes[0, 0]
        motion_data = simulation_results.get('motion_intensity', [])
        if motion_data and len(motion_data) > 0:
            frame_numbers = list(range(len(motion_data)))
            ax1.plot(frame_numbers, motion_data, 'b-', linewidth=2, label='Motion Intensity')
            ax1.set_xlabel('Frame Number')
            ax1.set_ylabel('Motion Intensity (scaled)')
            ax1.set_title('Fluid Motion Over Time (FIXED)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Add statistics text
            avg_motion = np.mean(motion_data)
            max_motion = np.max(motion_data)
            ax1.text(0.02, 0.98, f'Avg: {avg_motion:.2f}\nMax: {max_motion:.2f}', 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax1.text(0.5, 0.5, 'No motion data', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Fluid Motion Over Time (No Data)')
        
        # 2. Flow velocity over time
        ax2 = axes[0, 1]
        flow_data = simulation_results.get('flow_vectors', [])
        if flow_data and len(flow_data) > 0:
            frame_numbers = list(range(len(flow_data)))
            ax2.plot(frame_numbers, flow_data, 'r-', linewidth=2, label='Flow Velocity')
            ax2.set_xlabel('Frame Number')
            ax2.set_ylabel('Flow Velocity (pixels/frame)')
            ax2.set_title('Flow Velocity Over Time')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No flow data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Flow Velocity Over Time (No Data)')
        
        # 3. Water-air interface metrics
        ax3 = axes[1, 0]
        interface_data = simulation_results.get('interface_metrics', [])
        if interface_data and len(interface_data) > 0:
            frame_numbers = list(range(len(interface_data)))
            water_fractions = [m['water_fraction'] for m in interface_data]
            ax3.plot(frame_numbers, water_fractions, 'g-', linewidth=2, label='Water Fraction')
            ax3.set_xlabel('Frame Number')
            ax3.set_ylabel('Water Fraction')
            ax3.set_title('Water-Air Interface Evolution')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No interface data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Water-Air Interface Evolution (No Data)')
        
        # 4. Summary metrics (FIXED)
        ax4 = axes[1, 1]
        metrics = [
            simulation_results.get('avg_motion_intensity', 0),
            simulation_results.get('avg_flow_velocity', 0),
            simulation_results.get('avg_interface_complexity', 0),
            simulation_results.get('avg_water_fraction', 0)
        ]
        metric_names = ['Motion\nIntensity', 'Flow\nVelocity', 'Interface\nComplexity', 'Water\nFraction']
        
        bars = ax4.bar(metric_names, metrics, color=['blue', 'red', 'green', 'orange'])
        ax4.set_ylabel('Normalized Value')
        ax4.set_title('Fluid Dynamics Summary (FIXED)')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(metrics)*0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return fig
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        return None

def analyze_single_simulation_fixed(video_path):
    """Analyze a single simulation video with fixes"""
    try:
        ball_size = Path(video_path).stem
        print(f"\n🔬 FIXED ANALYSIS FOR {ball_size}")
        print("=" * 50)
        
        # Load video
        frames = load_simulation_video_fixed(video_path)
        if not frames:
            return None
        
        # Analyze fluid motion (FIXED)
        motion_results = analyze_fluid_motion_fixed(frames)
        
        # Analyze water-air interface
        interface_results = analyze_water_air_interface_fixed(frames)
        
        # Combine results
        combined_results = {}
        if motion_results:
            combined_results.update(motion_results)
        if interface_results:
            combined_results.update(interface_results)
        
        # Create visualization (FIXED)
        create_fixed_fluid_dynamics_visualization(combined_results, ball_size)
        
        return combined_results
        
    except Exception as e:
        print(f"❌ Error analyzing {video_path}: {e}")
        return None

# Execute fixed analysis
print("🚀 Starting FIXED fluid dynamics analysis...")

# Test with one video first
simulation_dir = Path("simulation/ansys-fluent")
video_files = list(simulation_dir.glob("*.mpeg"))

if video_files:
    test_video = str(video_files[0])
    print(f"🔬 Testing with: {Path(test_video).name}")
    
    result = analyze_single_simulation_fixed(test_video)
    
    if result:
        print("\n✅ FIXED ANALYSIS COMPLETE!")
        print("🔧 Key fixes:")
        print("  - Fixed motion intensity scaling (×100)")
        print("  - Corrected frame number x-axis")
        print("  - Added statistics display")
        print("  - Better error handling")
        print("  - Improved visualization")
    else:
        print("❌ Analysis failed")
else:
    print("❌ No simulation videos found!")




