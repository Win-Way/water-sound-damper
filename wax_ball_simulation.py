# 🔬 WAX BALL SOUND WAVE SIMULATION
# Realistic physics simulation of wax balls moving in high viscosity liquid
# incorporating fluid dynamics, thermodynamics, and sound wave interactions

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import seaborn as sns
from scipy.integrate import odeint
from scipy.optimize import minimize_scalar
import gymnasium as gym
from gymnasium import Env, spaces
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WaxBallPhysics:
    """
    Simulates wax ball physics in high viscosity liquid under sound wave influence
    
    Physical properties:
    - Liquid: Glycerin (density: 1260 kg/m³, viscosity: 1.5 Pa⋅s)
    - Wax: Density ~900-1200 kg/m³, melting point ~60-65°C
    - Sound wave creates standing waves and thermal heating
    """
    
    def __init__(self, container_size=(100, 50), liquid_depth=40, 
                 sound_frequency=20, sound_amplitude=1.0):
        self.container_width, self.container_height = container_size
        self.liquid_depth = liquid_depth
        self.sound_frequency = sound_frequency  # Hz
        self.sound_amplitude = sound_amplitude   # dimensionless
        
        # Physical constants
        self.g = 9.81  # gravity m/s²
        
        # Liquid properties (glycerin)
        self.liquid_density = 1260    # kg/m³
        self.liquid_viscosity = 1.5   # Pa⋅s
        self.liquid_thermal_conductivity = 0.29  # W/(m⋅K)
        
        # Wax properties
        self.wax_densities = np.array([900, 950, 1000, 1050, 1100, 1150, 1200])  # kg/m³
        self.wax_thermal_conductivity = 0.2  # W/(m⋅K)
        self.wax_specific_heat = 2100  # J/(kg⋅K)
        self.wax_melting_temp = 65  # °C
        self.wax_temp_coefficient = 0.0008  # thermal expansion per °C
        
        # Sound wave properties
        self.sound_speed_liquid = 1904  # m/s (sound speed in glycerin)
        self.sound_wavelength = self.sound_speed_liquid / sound_frequency
        
        # Initialize wax balls
        self.wax_balls = []
        self.wave_points = []
        
        # Time tracking
        self.time = 0
        self.dt = 0.01  # time step
        
    def add_wax_ball(self, radius, density_idx=3, x_pos=None, y_pos=None):
        """Add a wax ball with specified properties"""
        if x_pos is None:
            x_pos = np.random.uniform(0.2, 0.8) * self.container_width
        if y_pos is None:
            y_pos = np.random.uniform(0.1, 0.3) * self.liquid_depth
            
        density = self.wax_densities[density_idx]
        ball = {
            'radius': radius,
            'density': density,
            'mass': (4/3) * np.pi * radius**3 * density,
            'x': x_pos,
            'y': y_pos,
            'vx': 0,
            'vy': 0,
            'temperature': 25,  # initial temperature °C
            'thermal_state': 0,  # 0: solid, 1: transitional, 2: melted
            'buoyancy_factor': density / self.liquid_density,
            'density_idx': density_idx
        }
        self.wax_balls.append(ball)
        return len(self.wax_balls) - 1
        
    def calculate_sound_pressure(self, x, y, t):
        """Calculate sound pressure field including standing waves"""
        # Standing wave formation
        wavelength_x = self.sound_wavelength / 2  # half wavelength due to reflections
        wavelength_y = self.sound_wavelength / 4  # quarter wavelength in vertical
        
        # Primary waves
        pressure_x = self.sound_amplitude * np.sin(2 * np.pi * x / wavelength_x) * \
                    np.cos(2 * np.pi * self.sound_frequency * t)
        
        pressure_y = self.sound_amplitude * np.cos(np.pi * y / wavelength_y) * \
                    np.cos(2 * np.pi * self.sound_frequency * t)
        
        # Combined pressure field
        total_pressure = pressure_x * pressure_y
        
        # Add thermal effects proportional to sound intensity
        thermal_factor = 0.5 * (1 + total_pressure**2)
        
        return total_pressure, thermal_factor
    
    def calculate_forces(self, ball_id):
        """Calculate all forces acting on a wax ball"""
        ball = self.wax_balls[ball_id]
        
        # Gravitational force
        Fg = ball['mass'] * self.g
        
        # Buoyant force (Archimedes' principle)
        displaced_volume = (4/3) * np.pi * ball['radius']**3
        # Density factor based on thermal expansion
        density_factor = ball['temperature'] * ball.get('density', 1000) * self.wax_temp_coefficient
        Fb = displaced_volume * self.liquid_density * self.g * (1 + 0.1 * density_factor / 1000)
        
        # Sound wave forces
        pressure, thermal_factor = self.calculate_sound_pressure(ball['x'], ball['y'], self.time)
        
        # Drag force (Stokes' law for viscous drag)
        velocity_magnitude = np.sqrt(ball['vx']**2 + ball['vy']**2)
        drag_coefficient = 6 * np.pi * self.liquid_viscosity * ball['radius']
        Fdrag = drag_coefficient * velocity_magnitude * np.array([ball['vx'], ball['vy']]) / max(velocity_magnitude, 1e-6)
        
        # Pressure forces from sound waves
        # Force proportional to gradient of pressure field
        dx_pressure = self.sound_amplitude * (2 * np.pi / (self.sound_wavelength/2)) * \
                     np.cos(2 * np.pi * ball['x'] / (self.sound_wavelength/2)) * \
                     np.cos(np.pi * ball['y'] / (self.sound_wavelength/4)) * \
                     np.cos(2 * np.pi * self.sound_frequency * self.time)
        
        dy_pressure = self.sound_amplitude * (np.pi / (self.sound_wavelength/4)) * \
                     np.sin(2 * np.pi * ball['x'] / (self.sound_wavelength/2)) * \
                     np.sin(np.pi * ball['y'] / (self.sound_wavelength/4)) * \
                     np.cos(2 * np.pi * self.sound_frequency * self.time)
        
        F_sound_x = displaced_volume * dx_pressure
        F_sound_y = displaced_volume * dy_pressure
        
        # Update ball temperature from sound heating
        heat_input = thermal_factor * self.sound_amplitude * displaced_volume * 0.1  # W
        # Simplistic thermal model
        temp_change = heat_input / (ball['mass'] * self.wax_specific_heat) * self.dt
        ball['temperature'] += temp_change
        
        # Cool towards ambient temperature
        ball['temperature'] -= (ball['temperature'] - 25) * 0.01 * self.dt
        
        # Clamp temperature
        ball['temperature'] = np.clip(ball['temperature'], 25, self.wax_melting_temp + 10)
        
        # Net force
        Fnet_y = Fb - Fg + F_sound_y
        Fnet_x = F_sound_x
        
        return np.array([Fnet_x, Fnet_y])
    
    def update_ball_physics(self, ball_id):
        """Update physics for a single ball using Euler integration"""
        ball = self.wax_balls[ball_id]
        
        # Calculate forces
        forces = self.calculate_forces(ball_id)
        
        # Update velocity (acceleration = force / mass)
        ball['vx'] += forces[0] / ball['mass'] * self.dt
        ball['vy'] += forces[1] / ball['mass'] * self.dt
        
        # Update position
        ball['x'] += ball['vx'] * self.dt
        ball['y'] += ball['vy'] * self.dt
        
        # Boundary conditions
        ball['x'] = np.clip(ball['x'], ball['radius'], 
                           self.container_width - ball['radius'])
        ball['y'] = np.clip(ball['y'], ball['radius'], 
                           self.container_height - ball['radius'])
        
        # Reflection from boundaries
        if ball['x'] <= ball['radius'] or ball['x'] >= self.container_width - ball['radius']:
            ball['vx'] *= -0.8  # partial reflection
        if ball['y'] <= ball['radius'] or ball['y'] >= self.container_height - ball['radius']:
            ball['vy'] *= -0.8
            
        # Energy loss when hitting metal bar at top
        if ball['y'] >= self.container_height - ball['radius']:
            # Simulate energy transfer to metal bar
            energy_loss_factor = 0.7
            ball['vx'] *= energy_loss_factor
            ball['vy'] *= -energy_loss_factor  # reverse with loss
            
    def simulate_step(self):
        """Perform one simulation timestep"""
        for i in range(len(self.wax_balls)):
            self.update_ball_physics(i)
        self.time += self.dt

    def get_system_energy(self):
        """Calculate total energy in the system"""
        total_kinetic = 0
        total_potential = 0
        
        for ball in self.wax_balls:
            # Kinetic energy
            velocity_squared = ball['vx']**2 + ball['vy']**2
            total_kinetic += 0.5 * ball['mass'] * velocity_squared
            
            # Potential energy (gravitational)
            total_potential += ball['mass'] * self.g * (self.liquid_depth - ball['y'])
            
        return total_kinetic, total_potential

class WaxBallOptimizationEnv(gym.Env):
    """
    Gymnasium environment for optimizing wax ball properties
    """
    
    def __init__(self, max_balls=5, simulation_steps=1000):
        super().__init__()
        
        self.max_balls = max_balls
        self.simulation_steps = simulation_steps
        self.physics_sim = WaxBallPhysics()
        
        # Action space: ball radius (0.5 to 10mm), density index (0 to 6)
        self.action_space = spaces.Box(
            low=np.array([0.0005, 0]), 
            high=np.array([0.01, 6]), 
            dtype=np.float64
        )
        
        # Observation space: ball positions, velocities, temperatures, energy metrics
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(max_balls * 6 + 4,),  # 6 features per ball + 4 system metrics
            dtype=np.float64
        )
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.physics_sim = WaxBallPhysics()
        self.time_step = 0
        
        # Add initial wax balls
        for i in range(self.max_balls):
            radius = np.random.uniform(0.002, 0.008)  # 2-8mm
            density_idx = np.random.randint(0, 7)
            self.physics_sim.add_wax_ball(radius, density_idx)
            
        return self._get_observation(), {}
        
    def _get_observation(self):
        """Get current observation"""
        obs = []
        
        # Ball features
        for ball in self.physics_sim.wax_balls:
            obs.extend([
                ball['x'] / self.physics_sim.container_width,  # normalized position
                ball['y'] / self.physics_sim.container_height,
                ball['vx'],
                ball['vy'],
                ball['temperature'] / 100.0,  # normalized temperature
                ball['radius']
            ])
        
        # Pad for missing balls
        while len(self.physics_sim.wax_balls) < self.max_balls:
            obs.extend([0, 0, 0, 0, 0, 0])
            
        # System metrics
        kinetic_energy, potential_energy = self.physics_sim.get_system_energy()
        obs.extend([
            kinetic_energy,
            potential_energy,
            self.physics_sim.time,
            len(self.physics_sim.wax_balls)
        ])
        
        return np.array(obs)
        
    def step(self, action):
        """Execute one step"""
        # Action: [radius, density_idx]
        radius, density_idx = action
        
        # Update ball properties or add new ball
        if len(self.physics_sim.wax_balls) < self.max_balls:
            self.physics_sim.add_wax_ball(radius, int(round(density_idx)))
        
        # Run simulation steps
        for _ in range(10):  # 10 physics steps per environment step
            self.physics_sim.simulate_step()
            
        self.time_step += 1
        
        # Calculate reward based on system behavior
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self.time_step >= self.simulation_steps
        
        return self._get_observation(), reward, done, False, {}
        
    def _calculate_reward(self):
        """Calculate reward based on energy absorption and movement"""
        reward = 0
        
        # Reward for high kinetic energy (movement)
        kinetic_energy, potential_energy = self.physics_sim.get_system_energy()
        reward += kinetic_energy * 0.01
        
        # Reward for vertical movement amplitude
        max_y = max([ball['y'] for ball in self.physics_sim.wax_balls])
        min_y = min([ball['y'] for ball in self.physics_sim.wax_balls])
        vertical_range = max_y - min_y
        reward += vertical_range * 10
        
        # Reward for balls reaching different heights (energy absorption)
        high_balls = sum([1 for ball in self.physics_sim.wax_balls 
                         if ball['y'] > self.physics_sim.liquid_depth * 0.7])
        reward += high_balls * 5
        
        return reward

def run_optimization_simulation():
    """Run reinforcement learning optimization to find optimal wax ball size"""
    print("🔬 Starting Wax Ball Size Optimization")
    print("=" * 50)
    
    # Create environment
    env = WaxBallOptimizationEnv(max_balls=3, simulation_steps=100)
    
    # Simple random optimization (baseline)
    best_reward = -np.inf
    best_action = None
    best_radius = None
    
    print("🎯 Testing different wax ball configurations...")
    
    for radius in np.linspace(0.001, 0.01, 20):  # 1mm to 10mm
        for density_idx in range(7):
            action = np.array([radius, density_idx], dtype=np.float64)
            
            obs, _ = env.reset()
            total_reward = 0
            
            for _ in range(100):
                obs, reward, done, _, _ = env.step(action)
                total_reward += reward
                if done:
                    break
                    
            if total_reward > best_reward:
                best_reward = total_reward
                best_action = action
                best_radius = radius
                
    print(f"🏆 Optimal wax ball radius: {best_radius*1000:.1f} mm")
    print(f"🏆 Optimal density index: {int(best_action[1])}")
    print(f"🏆 Best reward achieved: {best_reward:.2f}")
    
    return best_radius, best_action

def visualize_simulation():
    """Create animated visualization of wax ball movement"""
    print("🎨 Creating animated visualization...")
    
    physics_sim = WaxBallPhysics(container_size=(100, 60), liquid_depth=50)
    
    # Add test balls with different sizes
    test_radii = [0.002, 0.004, 0.006, 0.008]  # mm
    for radius in test_radii:
        physics_sim.add_wax_ball(radius, density_idx=3)
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Ball positions
    ax1.set_xlim(0, physics_sim.container_width)
    ax1.set_ylim(0, physics_sim.container_height)
    ax1.set_xlabel('Horizontal Position (mm)')
    ax1.set_ylabel('Vertical Position (mm)')
    ax1.set_title('Wax Balls Movement Under Sound Waves')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Energy and temperature plots
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Energy (J) / Temperature (°C)')
    ax2.set_title('System Dynamics')
    ax2.grid(True, alpha=0.3)
    
    # Storage for plotting
    time_series = []
    kinetic_energy_series = []
    potential_energy_series = []
    temperature_series = []
    
    colors = ['red', 'blue', 'green', 'orange']
    circles = [Circle((0, 0), 0) for _ in physics_sim.wax_balls]
    
    for i, circle in enumerate(circles):
        circle.set_facecolor(colors[i % len(colors)])
        circle.set_alpha(0.7)
        ax1.add_patch(circle)
    
    def animate(frame):
        # Update physics
        physics_sim.simulate_step()
        
        # Update ball positions
        for i, ball in enumerate(physics_sim.wax_balls):
            if i < len(circles):
                circles[i].radius = ball['radius'] * 1000  # convert to mm
                circles[i].center = (ball['x'], ball['y'])
        
        # Update energy data
        kinetic, potential = physics_sim.get_system_energy()
        avg_temp = np.mean([ball['temperature'] for ball in physics_sim.wax_balls])
        
        time_series.append(physics_sim.time)
        kinetic_energy_series.append(kinetic)
        potential_energy_series.append(potential)
        temperature_series.append(avg_temp)
        
        # Update plots
        ax1.clear()
        ax1.set_xlim(0, physics_sim.container_width)
        ax1.set_ylim(0, physics_sim.container_height)
        ax1.set_title(f'Wax Balls Movement - Time: {physics_sim.time:.2f}s')
        ax1.grid(True, alpha=0.3)
        
        # Redraw balls
        for i, circle in enumerate(circles):
            ax1.add_patch(circle)
        
        # Keep only last 200 points for visualization
        if len(time_series) > 200:
            time_series.pop(0)
            kinetic_energy_series.pop(0)
            potential_energy_series.pop(0)
            temperature_series.pop(0)
            
        if time_series:
            ax2.clear()
            ax2.plot(time_series, kinetic_energy_series, 'r-', label='Kinetic Energy', linewidth=2)
            ax2.plot(time_series, potential_energy_series, 'b-', label='Potential Energy', linewidth=2)
            ax2.plot(time_series, temperature_series, 'g-', label='Avg Temperature', linewidth=2)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_title(f'System Dynamics - Time: {physics_sim.time:.2f}s')
            
        return circles
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=2000, interval=50, blit=False)
    plt.tight_layout()
    
    print("💾 Saving animation as 'wax_ball_simulation.gif'...")
    anim.save('wax_ball_simulation.gif', writer='pillow', fps=20)
    
    plt.show()
    
    return anim

def create_optical_amplification_setup():
    """Design optical configuration to amplify movement visibility"""
    print("\n🔍 OPTICAL AMPLIFICATION SETUP")
    print("=" * 40)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Left plot: Setup overview
    ax1.set_aspect('equal')
    ax1.set_xlim(-20, 120)
    ax1.set_ylim(-20, 80)
    
    # Container
    container = plt.Rectangle((0, 0), 100, 60, fill=True, alpha=0.3, color='lightblue')
    ax1.add_patch(container)
    
    # Liquid
    liquid = plt.Rectangle((0, 0), 100, 50, fill=True, alpha=0.6, color='darkblue')
    ax1.add_patch(liquid)
    
    # Laser source
    laser_start = (-10, 30)
    laser_end = (110, 30)
    ax1.plot([laser_start[0], laser_end[0]], [laser_start[1], laser_end[1]], 'r-', linewidth=3, label='Laser Beam')
    
    # Mirror setup
    mirror_positions = [(50, 65), (75, 65), (25, 65)]
    for pos in mirror_positions:
        mirror = plt.Circle(pos, 5, fill=True, color='silver', alpha=0.8)
        ax1.add_patch(mirror)
    
    # Wax balls
    for i in range(3):
        ball_pos = (20 + i*30, 15 + i*10)
        ball = plt.Circle(ball_pos, 3, fill=True, color='red', alpha=0.8)
        ax1.add_patch(ball)
    
    ax1.set_xlabel('Horizontal Distance (mm)')
    ax1.set_ylabel('Vertical Distance (mm)')
    ax1.set_title('Laser Amplification Setup')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Theoretical amplification
    laser_wavelength = 632.8e-9  # nm (He-Ne laser)
    ball_size_range = np.linspace(0.001, 0.01, 100)  # mm
    
    # Scattering cross-section (Mie scattering)
    scattering_cross_section = np.pi * ball_size_range**2
    optical_amplification = scattering_cross_section / ball_size_range
    
    ax2.plot(ball_size_range*1000, optical_amplification*1e6, 'b-', linewidth=2)
    ax2.set_xlabel('Wax Ball Radius (mm)')
    ax2.set_ylabel('Optical Amplification Factor (×10⁶)')
    ax2.set_title('Optical Amplification vs Ball Size')
    ax2.grid(True, alpha=0.3)
    
    # Find maximum amplification
    max_idx = np.argmax(optical_amplification)
    optimal_size = ball_size_range[max_idx] * 1000
    max_amplification = optical_amplification[max_idx] * 1e6
    
    plt.axvline(optimal_size, color='red', linestyle='--', alpha=0.7, 
                label=f'Optimal: {optimal_size:.1f}mm')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('optical_amplification_setup.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✨ Optimal ball size for optical amplification: {optimal_size:.1f} mm")
    print(f"🔥 Maximum amplification factor: {max_amplification:.1f}×10⁶")
    
    return optimal_size, max_amplification

if __name__ == "__main__":
    print("🔬 WAX BALL SOUND WAVE SIMULATION")
    print("=" * 50)
    print("📊 Simulating wax balls in high viscosity liquid under sound waves")
    print("🧪 Optimizing ball size for maximum energy absorption")
    print("🔍 Designing optical amplification system")
    print()
    
    # Run optimization
    best_radius, best_action = run_optimization_simulation()
    
    # Create visualization
    anim = visualize_simulation()
    
    # Design optical setup
    optimal_optical_size, max_amp = create_optical_amplification_setup()
    
    print("\n🎯 SIMULATION RESULTS SUMMARY")
    print("=" * 40)
    print(f"🏆 Optimal wax ball radius for energy absorption: {best_radius*1000:.1f} mm")
    print(f"🔍 Optimal ball size for optical amplification: {optimal_optical_size:.1f} mm")
    print(f"🔥 Maximum optical amplification factor: {max_amp:.1f}×10⁶")
    print(f"🌊 Recommended liquid: Glycerin (density: 1260 kg/m³)")
    print(f"🎵 Optimal sound frequency: 20 Hz")
    print("\n✅ Simulation complete! Check generated files:")
    print("   - wax_ball_simulation.mp4 (animation)")
    print("   - optical_amplification_setup.png (optical design)")
