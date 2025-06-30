import matplotlib.pyplot as plt
import numpy as np

def create_path_comparison():
    """Create a comparison plot showing Newtonian vs Langevin paths"""
    
    # Set up the figure with two subplots, one above the other
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.subplots_adjust(hspace=0)  # Remove space between subplots
    
    # Define points A and B
    point_A = np.array([1, 2])
    point_B = np.array([9, 7])
    
    # Calculate direction vector from A to B (for force field)
    direction = point_B - point_A
    direction_normalized = direction / np.linalg.norm(direction)
    
    # Colors
    point_color = 'red'
    newtonian_color = 'blue'
    force_field_color = 'gray'
    
    # Different colors for Langevin paths
    langevin_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', 
                      '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43']
    
    def add_force_field_background(ax, direction_normalized):
        """Add background force field arrows parallel to A-B line"""
        # Create a grid of points for the force field
        x_grid = np.linspace(0.5, 9.5, 10)
        y_grid = np.linspace(1.5, 7.5, 6)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # All arrows point in the direction from A to B
        U = np.full(X.shape, direction_normalized[0])
        V = np.full(Y.shape, direction_normalized[1])
        
        # Fixed: Adjusted scale for proper arrow length with scale_units='xy'
        # When using scale_units='xy', the scale needs to be much smaller
        ax.quiver(X, Y, U, V, alpha=0.2, color=force_field_color, 
                 scale=1.9, width=0.004, headwidth=3, headlength=4, zorder=1,
                 angles='uv', scale_units='xy')
    
    # Add force field background to both plots
    add_force_field_background(ax1, direction_normalized)
    add_force_field_background(ax2, direction_normalized)
    
    # Top plot: Newtonian path (straight line)
    ax1.plot([point_A[0], point_B[0]], [point_A[1], point_B[1]], 
             color=newtonian_color, linewidth=3, label='Newtonian Path', zorder=3)
    ax1.scatter(*point_A, color=point_color, s=120, zorder=5, label='Point A')
    ax1.scatter(*point_B, color=point_color, s=120, zorder=5, label='Point B')
    ax1.text(point_A[0]-0.3, point_A[1]+0.2, 'A', fontsize=14, fontweight='bold')
    ax1.text(point_B[0]+0.1, point_B[1]+0.2, 'B', fontsize=14, fontweight='bold')
    ax1.set_title('Newtonian Mechanics: Deterministic Path Along Force Field', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(1, 8)
    
    # Bottom plot: Langevin paths (jittery)
    ax2.scatter(*point_A, color=point_color, s=120, zorder=5)
    ax2.scatter(*point_B, color=point_color, s=120, zorder=5)
    ax2.text(point_A[0]-0.3, point_A[1]+0.2, 'A', fontsize=14, fontweight='bold')
    ax2.text(point_B[0]+0.1, point_B[1]+0.2, 'B', fontsize=14, fontweight='bold')
    
    # Generate 10 different jittery Langevin paths with different colors
    np.random.seed(42)  # For reproducible results
    n_paths = 10
    n_points = 50  # Number of points along each path
    
    for i in range(n_paths):
        # Create base path (straight line)
        t = np.linspace(0, 1, n_points)
        base_x = point_A[0] + t * (point_B[0] - point_A[0])
        base_y = point_A[1] + t * (point_B[1] - point_A[1])
        
        # Add Langevin noise (random walk component)
        noise_strength = 0.3
        noise_x = np.cumsum(np.random.normal(0, noise_strength, n_points))
        noise_y = np.cumsum(np.random.normal(0, noise_strength, n_points))
        
        # Ensure paths start and end at the correct points
        noise_x = noise_x - np.linspace(noise_x[0], noise_x[-1], n_points)
        noise_y = noise_y - np.linspace(noise_y[0], noise_y[-1], n_points)
        
        # Create jittery path
        jittery_x = base_x + noise_x
        jittery_y = base_y + noise_y
        
        # Plot the path with different colors
        color = langevin_colors[i % len(langevin_colors)]
        alpha = 0.8 if i == 0 else 0.7
        linewidth = 2.0 if i == 0 else 1.5
        ax2.plot(jittery_x, jittery_y, color=color, 
                alpha=alpha, linewidth=linewidth, zorder=3,
                label='Langevin Paths' if i == 0 else '')
    
    ax2.set_title('Langevin Dynamics: Stochastic Paths in Same Force Field', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(1, 8)
    ax2.set_xlabel('X Position', fontsize=12)
    
    # Add y-labels
    ax1.set_ylabel('Y Position', fontsize=12)
    ax2.set_ylabel('Y Position', fontsize=12)
    
    # Remove x-axis labels from top plot
    ax1.set_xticklabels([])
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('newtonian_vs_langevin_paths.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Path comparison image saved as 'newtonian_vs_langevin_paths.png'")
    print(f"Direction vector A->B: {direction}")
    print(f"Normalized direction: {direction_normalized}")

if __name__ == "__main__":
    create_path_comparison()