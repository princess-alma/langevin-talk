import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize

# Set style for publication-quality plots
try:
    import seaborn as sns
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
except ImportError:
    plt.style.use('classic')
    print("Seaborn not available, using classic style")

class ElongatedValleyVisualizer:
    """
    Creates matplotlib visualizations of elongated valley landscapes
    to demonstrate optimization challenges for SGD and SGLD.
    """
    
    def __init__(self, valley_ratio=100, noise_strength=0.5):
        """
        Args:
            valley_ratio: Controls elongation (higher = more elongated)
            noise_strength: SGLD noise parameter
        """
        self.valley_ratio = valley_ratio
        self.noise_strength = noise_strength
        self.x_range = (-4, 4)
        self.y_range = (-3, 3)
        
    def rosenbrock_valley(self, x, y):
        """
        Modified Rosenbrock function creating an elongated valley.
        Classic example of poor conditioning in optimization.
        """
        a = 1.0
        b = self.valley_ratio
        return (a - x)**2 + b * (y - x**2)**2
    
    def himmelblau_elongated(self, x, y):
        """
        Elongated version of Himmelblau's function with multiple valleys.
        """
        # Original Himmelblau terms
        term1 = (x**2 + y - 11)**2
        term2 = (x + y**2 - 7)**2
        
        # Add elongation factor
        elongation = self.valley_ratio * 0.01 * (y - 0.1*x**2)**2
        
        return term1 + term2 + elongation
    
    def create_landscape_grid(self, func):
        """Create meshgrid for landscape visualization."""
        x = np.linspace(self.x_range[0], self.x_range[1], 200)
        y = np.linspace(self.y_range[0], self.y_range[1], 200)
        X, Y = np.meshgrid(x, y)
        Z = func(X, Y)
        return X, Y, Z
    
    def gradient(self, func, x, y, h=1e-5):
        """Compute numerical gradient."""
        dx = (func(x + h, y) - func(x - h, y)) / (2 * h)
        dy = (func(x, y + h) - func(x, y - h)) / (2 * h)
        return np.array([dx, dy])
    
    def sgd_path(self, func, start_point, learning_rate=0.01, num_steps=500):
        """Generate SGD optimization path."""
        path = [start_point.copy()]
        current = start_point.copy()
        
        for _ in range(num_steps):
            grad = self.gradient(func, current[0], current[1])
            
            # Gradient clipping for stability
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 10:
                grad = grad / grad_norm * 10
                
            current = current - learning_rate * grad
            
            # Keep within bounds
            current = np.clip(current, [self.x_range[0], self.y_range[0]], 
                            [self.x_range[1], self.y_range[1]])
            path.append(current.copy())
            
        return np.array(path)
    
    def sgld_path(self, func, start_point, learning_rate=0.01, num_steps=500):
        """Generate SGLD optimization path with Langevin noise."""
        path = [start_point.copy()]
        current = start_point.copy()
        
        for _ in range(num_steps):
            grad = self.gradient(func, current[0], current[1])
            
            # Gradient clipping
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 10:
                grad = grad / grad_norm * 10
            
            # Add Langevin noise
            noise = np.random.normal(0, 1, size=2)
            current = (current - learning_rate * grad + 
                      noise * np.sqrt(2 * learning_rate * self.noise_strength))
            
            # Keep within bounds
            current = np.clip(current, [self.x_range[0], self.y_range[0]], 
                            [self.x_range[1], self.y_range[1]])
            path.append(current.copy())
            
        return np.array(path)
    
    def plot_static_comparison(self, func_name="rosenbrock", save_path=None):
        """
        Create static plot comparing SGD vs SGLD on elongated valley.
        Perfect for slides!
        """
        # Choose function
        if func_name == "rosenbrock":
            func = self.rosenbrock_valley
            title = f"SGD vs SGLD: Rosenbrock Valley (ratio={self.valley_ratio})"
            start_point = np.array([-3.0, 2.5])
        else:
            func = self.himmelblau_elongated
            title = f"SGD vs SGLD: Elongated Himmelblau (ratio={self.valley_ratio})"
            start_point = np.array([-3.0, 2.0])
        
        # Create landscape
        X, Y, Z = self.create_landscape_grid(func)
        
        # Generate paths
        sgd_trajectory = self.sgd_path(func, start_point, learning_rate=0.01, num_steps=400)
        sgld_trajectory = self.sgld_path(func, start_point, learning_rate=0.015, num_steps=400)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot contours with logarithmic spacing for better visualization
        levels = np.logspace(np.log10(Z.min() + 1), np.log10(Z.max()), 20)
        contour = ax.contour(X, Y, Z, levels=levels, colors='gray', alpha=0.6, linewidths=0.8)
        ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.3)
        
        # Plot optimization paths
        ax.plot(sgd_trajectory[:, 0], sgd_trajectory[:, 1], 
               'r-', linewidth=3, label='SGD', alpha=0.8)
        ax.plot(sgld_trajectory[:, 0], sgld_trajectory[:, 1], 
               'b-', linewidth=3, label='SGLD', alpha=0.8)
        
        # Mark starting points
        ax.plot(start_point[0], start_point[1], 'ko', markersize=10, 
               label='Start', markerfacecolor='yellow', markeredgewidth=2)
        
        # Mark end points
        ax.plot(sgd_trajectory[-1, 0], sgd_trajectory[-1, 1], 'rs', 
               markersize=8, label='SGD End')
        ax.plot(sgld_trajectory[-1, 0], sgld_trajectory[-1, 1], 'bs', 
               markersize=8, label='SGLD End')
        
        # Formatting
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add text box explaining the challenge
        textstr = f'Valley Ratio: {self.valley_ratio}\nNoise Strength: {self.noise_strength}\n\nSGD struggles with\npoor conditioning.\nSGLD noise helps\nescape narrow valleys.'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        return fig, ax
    
    def plot_multiple_valleys(self, save_path=None):
        """
        Create a comparison plot showing different valley ratios.
        Great for demonstrating the effect of conditioning!
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        valley_ratios = [10, 50, 100, 200]
        start_point = np.array([-3.0, 2.5])
        
        for i, ratio in enumerate(valley_ratios):
            # Temporarily change valley ratio
            original_ratio = self.valley_ratio
            self.valley_ratio = ratio
            
            # Create landscape and paths
            X, Y, Z = self.create_landscape_grid(self.rosenbrock_valley)
            sgd_traj = self.sgd_path(self.rosenbrock_valley, start_point, 0.01, 300)
            sgld_traj = self.sgld_path(self.rosenbrock_valley, start_point, 0.015, 300)
            
            # Plot
            levels = np.logspace(np.log10(Z.min() + 1), np.log10(Z.max()), 15)
            axes[i].contour(X, Y, Z, levels=levels, colors='gray', alpha=0.6)
            axes[i].contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.3)
            
            axes[i].plot(sgd_traj[:, 0], sgd_traj[:, 1], 'r-', linewidth=2, label='SGD')
            axes[i].plot(sgld_traj[:, 0], sgld_traj[:, 1], 'b-', linewidth=2, label='SGLD')
            axes[i].plot(start_point[0], start_point[1], 'ko', markersize=8)
            
            axes[i].set_title(f'Valley Ratio = {ratio}', fontsize=14)
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('y')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Restore original ratio
            self.valley_ratio = original_ratio
        
        plt.suptitle('Effect of Valley Elongation on SGD vs SGLD', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()
        return fig, axes
    
    def create_3d_surface_plot(self, func_name="rosenbrock", save_path=None):
        """
        Create beautiful 3D surface plot of the elongated valley.
        Perfect for showing the landscape structure!
        """
        # Choose function
        func = self.rosenbrock_valley if func_name == "rosenbrock" else self.himmelblau_elongated
        
        # Create landscape
        X, Y, Z = self.create_landscape_grid(func)
        
        # Create 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface with custom colormap
        surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                                linewidth=0, antialiased=True, shade=True)
        
        # Add contour lines on the bottom for reference
        contour_levels = np.linspace(Z.min(), Z.max(), 10)
        ax.contour(X, Y, Z, levels=contour_levels, zdir='z', 
                  offset=Z.min()-10, cmap='gray', alpha=0.5)
        
        # Generate and plot paths
        start_point = np.array([-3.0, 2.5])
        sgd_traj = self.sgd_path(func, start_point, 0.01, 300)
        sgld_traj = self.sgld_path(func, start_point, 0.015, 300)
        
        # Evaluate function along paths for 3D plotting
        sgd_z = [func(point[0], point[1]) for point in sgd_traj]
        sgld_z = [func(point[0], point[1]) for point in sgld_traj]
        
        # Plot paths on surface
        ax.plot(sgd_traj[:, 0], sgd_traj[:, 1], sgd_z, 
               'r-', linewidth=4, label='SGD Path', alpha=0.9)
        ax.plot(sgld_traj[:, 0], sgld_traj[:, 1], sgld_z, 
               'b-', linewidth=4, label='SGLD Path', alpha=0.9)
        
        # Mark start point
        start_z = func(start_point[0], start_point[1])
        ax.scatter([start_point[0]], [start_point[1]], [start_z], 
                  color='yellow', s=100, label='Start')
        
        # Formatting
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Energy', fontsize=12)
        ax.set_title(f'3D Elongated Valley (Ratio={self.valley_ratio})', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        
        # Set viewing angle for best visualization
        ax.view_init(elev=30, azim=45)
        
        # Add colorbar
        fig.colorbar(surface, shrink=0.5, aspect=10)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D plot saved to {save_path}")
        
        plt.show()
        return fig, ax
    
    def create_clean_3d_surface(self, func_name="rosenbrock", save_path=None):
        """
        Create clean 3D surface plot of the elongated valley without paths, legends, or labels.
        Perfect for slides where you just want to show the landscape structure!
        """
        # Choose function
        func = self.rosenbrock_valley if func_name == "rosenbrock" else self.himmelblau_elongated
        
        # Create landscape
        X, Y, Z = self.create_landscape_grid(func)
        
        # Create 3D plot with larger figure size for slides
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface with clean appearance
        surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9, 
                                linewidth=0, antialiased=True, shade=True, 
                                rcount=100, ccount=100)
        
        # Remove all labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        
        # Remove the axis lines and panes for ultra-clean look
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')
        ax.xaxis.pane.set_alpha(0)
        ax.yaxis.pane.set_alpha(0)
        ax.zaxis.pane.set_alpha(0)
        
        # Set viewing angle for best visualization of the valley structure
        ax.view_init(elev=25, azim=45)
        
        # Make background transparent for slides
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='none', edgecolor='none', transparent=True)
            print(f"Clean 3D plot saved to {save_path}")
        
        plt.show()
        return fig, ax
    
    def create_animation(self, func_name="rosenbrock", save_path=None):
        """
        Create animated version showing paths being traced.
        """
        func = self.rosenbrock_valley if func_name == "rosenbrock" else self.himmelblau_elongated
        X, Y, Z = self.create_landscape_grid(func)
        
        start_point = np.array([-3.0, 2.5])
        sgd_traj = self.sgd_path(func, start_point, 0.01, 200)
        sgld_traj = self.sgld_path(func, start_point, 0.015, 200)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot static elements
        levels = np.logspace(np.log10(Z.min() + 1), np.log10(Z.max()), 20)
        ax.contour(X, Y, Z, levels=levels, colors='gray', alpha=0.6)
        ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.3)
        ax.plot(start_point[0], start_point[1], 'ko', markersize=10, markerfacecolor='yellow')
        
        # Initialize empty lines for paths
        sgd_line, = ax.plot([], [], 'r-', linewidth=3, label='SGD')
        sgld_line, = ax.plot([], [], 'b-', linewidth=3, label='SGLD')
        
        ax.set_xlim(self.x_range)
        ax.set_ylim(self.y_range)
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.set_title(f'SGD vs SGLD Animation (Ratio={self.valley_ratio})', fontsize=16)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        def animate(frame):
            # Update SGD line
            sgd_line.set_data(sgd_traj[:frame, 0], sgd_traj[:frame, 1])
            # Update SGLD line
            sgld_line.set_data(sgld_traj[:frame, 0], sgld_traj[:frame, 1])
            return sgd_line, sgld_line
        
        anim = FuncAnimation(fig, animate, frames=len(sgd_traj), 
                           interval=50, blit=True, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=20)
            print(f"Animation saved to {save_path}")
        
        plt.show()
        return anim


def main():
    """
    Main function demonstrating different visualizations.
    Perfect for generating slides!
    """
    print("Creating Elongated Valley Visualizations for SGD vs SGLD...")
    
    # Create visualizer with different valley ratios
    viz_narrow = ElongatedValleyVisualizer(valley_ratio=100, noise_strength=0.5)
    viz_extreme = ElongatedValleyVisualizer(valley_ratio=200, noise_strength=0.6)
    
    # 1. Clean 3D surface (perfect for slides)
    print("\n1. Creating clean 3D surface plot...")
    viz_narrow.create_clean_3d_surface("rosenbrock", "clean_elongated_valley_3d.png")
    
    # 2. Static comparison plot (best for slides)
    print("\n2. Creating static comparison plot...")
    viz_narrow.plot_static_comparison("rosenbrock", "elongated_valley_comparison.png")
    
    # 3. Multiple valley ratios comparison
    print("\n3. Creating multiple valleys comparison...")
    viz_narrow.plot_multiple_valleys("valley_ratios_comparison.png")
    
    # 4. 3D surface plot with paths
    print("\n4. Creating 3D surface plot with paths...")
    viz_narrow.create_3d_surface_plot("rosenbrock", "elongated_valley_3d.png")
    
    # 5. Extremely elongated valley
    print("\n5. Creating extremely elongated valley...")
    viz_extreme.plot_static_comparison("rosenbrock", "extreme_elongated_valley.png")
    
    # 6. Create animation (optional)
    print("\n6. Creating animation...")
    try:
        viz_narrow.create_animation("rosenbrock", "valley_animation.gif")
    except Exception as e:
        print(f"Animation creation failed: {e}")
        print("This is normal if you don't have pillow installed.")
    
    print("\nAll visualizations complete!")
    print("\nFiles generated:")
    print("- clean_elongated_valley_3d.png (CLEAN 3D for slides)")
    print("- elongated_valley_comparison.png (main slide)")
    print("- valley_ratios_comparison.png (comparison slide)")
    print("- elongated_valley_3d.png (3D visualization with paths)")
    print("- extreme_elongated_valley.png (extreme case)")
    print("- valley_animation.gif (if pillow available)")


if __name__ == "__main__":
    main()