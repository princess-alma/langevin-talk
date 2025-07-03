import numpy as np
from manim import *
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from typing import List, Tuple, Callable


class PathVisualization(Scene):
    """
    Modular Manim scene for visualizing optimization paths over distribution contours.
    
    This class takes a distribution function and paths as input parameters,
    then creates an animated visualization showing the paths traced over contour plots.
    """
    
    def __init__(self, 
                 distribution_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 paths: List[np.ndarray],
                 path_labels: List[str],
                 path_colors: List[str],
                 x_range: Tuple[float, float, float] = (-3, 3, 1), # DEBUG: Added step to x_range/y_range
                 y_range: Tuple[float, float, float] = (-3, 3, 1),
                 contour_levels: int = 15,
                 contour_color_map = plt.cm.viridis, # DEBUG: Added colormap for contours
                 animation_time: float = 8.0,
                 show_arrows: bool = True,
                 show_dots: bool = True,
                 dot_size: float = 0.05, # DEBUG: Adjusted default size
                 arrow_scale: float = 0.3,
                 **kwargs):
        """
        Initialize the path visualization.
        
        Args:
            distribution_func: Function that takes (X, Y) meshgrid and returns Z values
            paths: List of paths, each path is an array of shape (n_steps, 2)
            path_labels: Labels for each path (e.g., ["SGD", "SGLD"])
            path_colors: Colors for each path (e.g., ["#FF6B6B", "#4ECDC4"])
            x_range: Range for x-axis as (min, max, step)
            y_range: Range for y-axis as (min, max, step)
            contour_levels: Number of contour levels to display
            contour_color_map: Matplotlib colormap for the contours.
            animation_time: Total time for path animation
            show_arrows: Whether to show direction arrows along paths
            show_dots: Whether to show dots at path points
            dot_size: Size of dots along paths
            arrow_scale: Scale factor for arrows
        """
        super().__init__(**kwargs)
        self.distribution_func = distribution_func
        self.paths = paths
        self.path_labels = path_labels
        self.path_colors = path_colors
        self.x_range = x_range
        self.y_range = y_range
        self.contour_levels = contour_levels
        self.contour_color_map = contour_color_map
        self.animation_time = animation_time
        self.show_arrows = show_arrows
        self.show_dots = show_dots
        self.dot_size = dot_size
        self.arrow_scale = arrow_scale
        
    # DEBUG: This method is now completely rewritten to generate REAL contours.
    def create_contour_background(self, axes: Axes) -> VGroup:
        """Create a real contour plot background using the provided distribution."""
        # Generate meshgrid for contour computation
        x = np.linspace(axes.x_range[0], axes.x_range[1], 100)
        y = np.linspace(axes.y_range[0], axes.y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = self.distribution_func(X, Y)
        
        # Use Matplotlib to calculate contour lines
        fig, ax = plt.subplots(figsize=(1, 1))
        contour_set = ax.contour(X, Y, Z, self.contour_levels, cmap=self.contour_color_map)
        
        contour_lines = VGroup()
        # DEBUG: Use colormap to get colors directly
        colormap = self.contour_color_map
        num_levels = len(contour_set.allsegs)
        
        for i, level_paths in enumerate(contour_set.allsegs):
            # Get color from colormap
            color_intensity = i / max(1, num_levels - 1)
            color = colormap(color_intensity)[:3]  # Get RGB, ignore alpha
            
            for path_vertices in level_paths:
                if len(path_vertices) > 2:
                    # Convert vertices from data coords to manim coords
                    points = [axes.coords_to_point(v[0], v[1]) for v in path_vertices]
                    
                    # Create a Manim VMobject from the points
                    line = VMobject(stroke_color=rgb_to_color(color), stroke_width=2.5)
                    line.set_points_as_corners(points)
                    contour_lines.add(line)
        
        # Close the temporary matplotlib plot to avoid displaying it
        plt.close(fig)
        return contour_lines
    
    # DEBUG: This method now accepts an 'axes' object instead of creating its own.
    def create_path_mobjects(self, axes: Axes) -> List[VGroup]:
        """Create Manim objects for each path, using the main scene axes."""
        path_mobjects = []
        
        for i, (path, color) in enumerate(zip(self.paths, self.path_colors)):
            path_group = VGroup()
            
            # Convert path points to Manim coordinates using the provided axes
            manim_points = [
                axes.coords_to_point(point[0], point[1]) 
                for point in path
            ]
            
            # Create path line - made thinner
            if len(manim_points) >= 2:
                path_line = VMobject()
                path_line.set_points_as_corners(manim_points)
                path_line.set_stroke(color=color, width=2.5, opacity=0.9)  # Reduced from 4 to 2.5
                path_group.add(path_line)
            
            if self.show_dots:
                dots = VGroup()
                dot_interval = max(1, len(manim_points) // 25) # Adjusted for clarity
                for j in range(0, len(manim_points), dot_interval):
                    dot = Dot(manim_points[j], radius=self.dot_size, color=color)
                    dots.add(dot)
                path_group.add(dots)
            
            # Remove arrows - commenting out the entire arrow section
            # if self.show_arrows and len(manim_points) >= 2:
            #     arrows = VGroup()
            #     arrow_interval = max(1, len(manim_points) // 12) # Adjusted for clarity
            #     for j in range(0, len(manim_points) - arrow_interval, arrow_interval):
            #         start_pt = manim_points[j]
            #         end_pt = manim_points[j + arrow_interval]
            #         direction = end_pt - start_pt
            #         if np.linalg.norm(direction) > 0.01: # Avoid zero-length arrows
            #             arrow = Arrow(
            #                 start_pt, end_pt, 
            #                 color=color, 
            #                 stroke_width=5, 
            #                 tip_length=0.2, 
            #                 max_tip_length_to_length_ratio=0.5,
            #                 buff=0
            #             )
            #             arrows.add(arrow)
            #     path_group.add(arrows)
            
            # Add empty VGroup for arrows to maintain structure
            arrows = VGroup()
            path_group.add(arrows)
            
            path_mobjects.append(path_group)
        
        return path_mobjects
    
    def create_legend(self) -> VGroup:
        """Create a legend for the paths positioned to the right."""
        legend = VGroup()
        for i, (label, color) in enumerate(zip(self.path_labels, self.path_colors)):
            line = Line(ORIGIN, RIGHT * 0.5, color=color, stroke_width=6)
            # DEBUG: Use a color that contrasts with the background
            text_color = self.camera.background_color.invert()
            text = Text(label, font_size=18, color=text_color)  # Reduced from 24 to 18
            
            entry = VGroup(line, text.next_to(line, RIGHT, buff=0.2))
            legend.add(entry)
        
        # Arrange legend entries vertically
        legend.arrange(DOWN, buff=0.2)
        # Position the legend in the top right corner, but keep it within frame
        legend.to_corner(UR, buff=0.5)  # Removed the extra RIGHT shift to keep it in frame
        return legend
    
    def construct(self):
        """Main construction method for the Manim scene."""
        self.camera.background_color = WHITE
        
        # DEBUG: Create ONE Axes object for the whole scene.
        axes = Axes(
            x_range=self.x_range,
            y_range=self.y_range,
            x_length=8,
            y_length=8,
            axis_config={"color": BLACK, "include_tip": False},
            x_axis_config={"numbers_to_include": np.arange(self.x_range[0], self.x_range[1]+1, self.x_range[2])},
            y_axis_config={"numbers_to_include": np.arange(self.y_range[0], self.y_range[1]+1, self.y_range[2])}
        ).add_coordinates()
        
        # DEBUG: Pass the axes to the helper methods.
        background_contours = self.create_contour_background(axes)
        path_mobjects = self.create_path_mobjects(axes)
        legend = self.create_legend()
        
        # Animation sequence
        self.play(Create(axes), Create(background_contours), run_time=2)
        self.play(Write(legend), run_time=1)
        
        # DEBUG: Animate the components separately for a better effect.
        # First, create animations that TRACE the path lines.
        line_animations = []
        details_group = VGroup() # Group for all dots and arrows
        for path_mob in path_mobjects:
            line_animations.append(Create(path_mob[0])) # path_mob[0] is the line
            details_group.add(path_mob[1]) # path_mob[1] is the dots
            details_group.add(path_mob[2]) # path_mob[2] is the arrows

        # Play the tracing of the paths
        self.play(AnimationGroup(*line_animations, lag_ratio=0.2), run_time=self.animation_time)
        # Now, fade in the dots and arrows all at once for clarity
        self.play(FadeIn(details_group), run_time=1)
        
        self.wait(2)


def generate_sample_distribution(distribution_type: str = "teardrop_bimodal") -> Callable:
    """
    Generate sample distribution functions for testing.
    The function now returns the NEGATIVE LOG of the probability density.
    """
    if distribution_type == "teardrop_bimodal":
        def teardrop_bimodal(X, Y):
            g1 = multivariate_normal.pdf(np.dstack([X, Y]), mean=[-1.5, -1.5], cov=[[0.8, 0.3], [0.3, 0.6]])
            g2 = multivariate_normal.pdf(np.dstack([X, Y]), mean=[1.5, 1.5], cov=[[0.4, -0.1], [-0.1, 0.7]])
            # DEBUG: Use the negative log-probability for a proper energy landscape with usable gradients.
            # Add a small epsilon to prevent log(0).
            probability = 2.0 * g1 + 1.5 * g2
            return -np.log(probability + 1e-9)
        return teardrop_bimodal
    
    elif distribution_type == "gaussian_mixture":
        def gaussian_mixture(X, Y):
            g1 = multivariate_normal.pdf(np.dstack([X, Y]), mean=[-1, -1], cov=[[0.5, 0.2], [0.2, 0.5]])
            g2 = multivariate_normal.pdf(np.dstack([X, Y]), mean=[1, 1], cov=[[0.3, -0.1], [-0.1, 0.3]])
            # DEBUG: Use the negative log-probability here as well.
            probability = g1 + 0.7 * g2
            return -np.log(probability + 1e-9)
        return gaussian_mixture
    
    else: # Fallback to a simple gaussian
        def simple_gaussian(X, Y):
            return (X**2 + Y**2) / 2 # Corresponds to -log(P) for a standard normal
        return simple_gaussian


def generate_sample_paths(dist_func: Callable) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    Generate sample optimization paths that follow the gradient of the given distribution.
    """
    def get_grad(x, y):
        h = 1e-5
        dx = (dist_func(x + h, y) - dist_func(x - h, y)) / (2 * h)
        dy = (dist_func(x, y + h) - dist_func(x, y - h)) / (2 * h)
        return np.array([dx, dy])

    # DEBUG: Adjusted learning rates and noise for the new -log(P) landscape
    # Simulate SGD path (gradient descent)
    sgd_path = []
    current_pos = np.array([-2.5, -2.5])
    learning_rate = 0.1
    for _ in range(80):
        sgd_path.append(current_pos.copy())
        grad = get_grad(current_pos[0], current_pos[1])
        current_pos -= learning_rate * grad

    # Simulate SGLD path (Langevin Dynamics)
    sgld_path = []
    current_pos = np.array([-2.5, -2.5])
    learning_rate = 0.1
    noise_strength = 0.3
    for _ in range(120):
        sgld_path.append(current_pos.copy())
        grad = get_grad(current_pos[0], current_pos[1])
        noise = np.random.normal(0, 1, size=2)
        current_pos -= learning_rate * grad + noise * np.sqrt(2 * learning_rate * noise_strength)

    paths = [np.array(sgd_path), np.array(sgld_path)]
    labels = ["SGD", "SGLD"]
    colors = ["#FF6B6B", "#4ECDC4"]
    
    return paths, labels, colors


if __name__ == "__main__":
    # Example usage - but this won't work with manim command line
    # Use ExamplePathVisualization instead for rendering
    
    # Generate sample distribution and paths
    distribution = generate_sample_distribution("teardrop_bimodal")
    paths, labels, colors = generate_sample_paths(distribution)
    
    print("PathVisualization module loaded successfully!")
    print("To render an animation, use:")
    print("manim path_visualization.py ExamplePathVisualization -p")


class ExamplePathVisualization(PathVisualization):
    """
    Example scene that can be rendered directly with Manim.
    This demonstrates SGD vs SGLD on a teardrop bimodal distribution.
    """
    
    def __init__(self, **kwargs):
        distribution = generate_sample_distribution("teardrop_bimodal")
        # DEBUG: Generate paths that actually follow the distribution's gradient
        paths, labels, colors = generate_sample_paths(distribution)
        
        super().__init__(
            distribution_func=distribution,
            paths=paths,
            path_labels=labels,
            path_colors=colors,
            x_range=(-4, 4, 1), # DEBUG: Expanded range to better fit paths
            y_range=(-4, 4, 1),
            contour_levels=15,
            animation_time=8.0,
            **kwargs
        )

# Add this at the end of path_visualization.py for easy tuning

class CustomPathVisualization(PathVisualization):
    """
    Custom scene for manual parameter tuning.
    Modify the parameters below to experiment with different settings.
    """
    
    def __init__(self, **kwargs):
        distribution = generate_sample_distribution("teardrop_bimodal")
        
        # ========================
        
        # Starting points (can be different for each algorithm)
        sgd_start = np.array([3.0, -3.0])    # SGD starting position
        sgld_start = np.array([3.0, -3.0])   # SGLD starting position
        
        # Learning rates
        sgd_lr = 0.08                         # SGD step size
        sgld_lr = 0.05                        # SGLD step size
        
        # Number of optimization steps
        sgd_steps = 100                       # SGD iterations
        sgld_steps = 300                      # SGLD iterations
        
        # SGLD noise parameters
        noise_strength = 1.2                  # SGLD exploration strength
        
        # Visualization parameters
        coord_range = (-4, 4, 1)             # Coordinate system range
        contour_levels = 20                   # Number of contour lines
        animation_duration = 10.0             # Animation length in seconds
        
        # ========================
        # Generate custom paths
        paths = self.generate_custom_paths(
            distribution, sgd_start, sgld_start, 
            sgd_lr, sgld_lr, sgd_steps, sgld_steps, noise_strength
        )
        
        super().__init__(
            distribution_func=distribution,
            paths=paths,
            path_labels=["SGD", "SGLD"],
            path_colors=["#FF6B6B", "#4ECDC4"],
            x_range=coord_range,
            y_range=coord_range,
            contour_levels=contour_levels,
            animation_time=animation_duration,
            **kwargs
        )
    
    def generate_custom_paths(self, dist_func, sgd_start, sgld_start, 
                            sgd_lr, sgld_lr, sgd_steps, sgld_steps, noise_strength):
        """Generate paths with custom parameters."""
        def get_grad(x, y):
            h = 1e-5
            dx = (dist_func(x + h, y) - dist_func(x - h, y)) / (2 * h)
            dy = (dist_func(x, y + h) - dist_func(x, y - h)) / (2 * h)
            return np.array([dx, dy])

        # Generate SGD path
        sgd_path = []
        current_pos = sgd_start.copy()
        for _ in range(sgd_steps):
            sgd_path.append(current_pos.copy())
            grad = get_grad(current_pos[0], current_pos[1])
            current_pos -= sgd_lr * grad

        # Generate SGLD path
        sgld_path = []
        current_pos = sgld_start.copy()
        for _ in range(sgld_steps):
            sgld_path.append(current_pos.copy())
            grad = get_grad(current_pos[0], current_pos[1])
            noise = np.random.normal(0, 1, size=2)
            current_pos -= sgld_lr * grad + noise * np.sqrt(2 * sgld_lr * noise_strength)

        return [np.array(sgd_path), np.array(sgld_path)]


# Add this new class for multimodal SGLD schedule comparison

class MultimodalSGLDVisualization(PathVisualization):
    """
    Visualization comparing different SGLD step size schedules on a highly multimodal landscape.
    Shows three SGLD variants:
    1. Polynomial decay (high decay rate)
    2. Polynomial decay (low decay rate) 
    3. Cosine annealing schedule
    """
    
    def __init__(self, **kwargs):
        distribution = self.create_multimodal_distribution()
        
        # Parameters for all SGLD variants
        start_point = np.array([3.5, 3.5])    # Start far from any mode
        base_lr = 0.15                        # Base learning rate
        num_steps = 800                       # More steps to see schedule effects
        noise_strength = 0.8                  # Strong noise for exploration
        
        # Generate paths with different schedules
        paths = self.generate_sgld_schedule_paths(
            distribution, start_point, base_lr, num_steps, noise_strength
        )
        
        super().__init__(
            distribution_func=distribution,
            paths=paths,
            path_labels=["High Decay", "Low Decay", "Cyclical"],
            path_colors=["#FF6B6B", "#4ECDC4", "#9B59B6"],  # Red, Teal, Purple
            x_range=(-4, 4, 1), 
            y_range=(-4, 4, 1),
            contour_levels=25,  # More contours for complex landscape
            contour_color_map=plt.cm.gray,  # Use gray colormap for black contours
            animation_time=15.0,  # Longer animation for more steps
            **kwargs
        )
    
    def create_multimodal_distribution(self) -> Callable:
        """Create a highly multimodal distribution with grid of modes."""
        def multimodal_grid(X, Y):
            # Create a 3x3 grid of Gaussian modes
            modes = []
            
            # Grid positions
            positions = [
                [-2.5, -2.5], [0, -2.5], [2.5, -2.5],    # Bottom row
                [-2.5, 0],    [0, 0],    [2.5, 0],       # Middle row
                [-2.5, 2.5],  [0, 2.5],  [2.5, 2.5]      # Top row
            ]
            
            # Different strengths and shapes for variety
            strengths = [1.5, 2.0, 1.2, 1.8, 2.5, 1.3, 1.6, 1.9, 1.4]
            covariances = [
                [[0.3, 0.1], [0.1, 0.3]],     # Slightly elongated
                [[0.2, 0], [0, 0.4]],         # Vertically stretched
                [[0.4, -0.1], [-0.1, 0.2]],   # Tilted
                [[0.25, 0], [0, 0.25]],       # Circular
                [[0.2, 0], [0, 0.2]],         # Small and tight (strongest mode)
                [[0.35, 0.05], [0.05, 0.35]], # Medium
                [[0.3, -0.05], [-0.05, 0.3]], # Slightly tilted
                [[0.28, 0], [0, 0.3]],        # Oval
                [[0.32, 0.08], [0.08, 0.25]]  # Irregular
            ]
            
            total_prob = 0
            for pos, strength, cov in zip(positions, strengths, covariances):
                mode_prob = strength * multivariate_normal.pdf(
                    np.dstack([X, Y]), mean=pos, cov=cov
                )
                total_prob += mode_prob
            
            # Return negative log probability for energy landscape
            return -np.log(total_prob + 1e-9)
        
        return multimodal_grid
    
    def generate_sgld_schedule_paths(self, dist_func, start_point, base_lr, num_steps, noise_strength):
        """Generate SGLD paths with different step size schedules."""
        
        def get_grad(x, y):
            h = 1e-5
            dx = (dist_func(x + h, y) - dist_func(x - h, y)) / (2 * h)
            dy = (dist_func(x, y + h) - dist_func(x, y - h)) / (2 * h)
            return np.array([dx, dy])
        
        def polynomial_schedule(step, total_steps, base_lr, decay_rate):
            """Polynomial decay: lr = base_lr * (1 - step/total_steps)^decay_rate"""
            progress = step / total_steps
            return base_lr * (1 - progress) ** decay_rate
        
        def cosine_annealing_schedule(step, total_steps, base_lr, min_lr=0.01):
            """Cosine annealing with multiple cycles: lr oscillates continuously"""
            # Use multiple cycles (e.g., 4 cycles over the total steps)
            num_cycles = 8
            cycle_length = total_steps / num_cycles
            cycle_position = (step % cycle_length) / cycle_length
            return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(2 * np.pi * cycle_position))
        
        all_paths = []
        
        # 1. SGLD with High Polynomial Decay (aggressive cooling)
        path_high_decay = []
        current_pos = start_point.copy()
        for step in range(num_steps):
            path_high_decay.append(current_pos.copy())
            
            # High decay rate (lr drops quickly)
            lr = polynomial_schedule(step, num_steps, base_lr, decay_rate=2.5)
            
            grad = get_grad(current_pos[0], current_pos[1])
            noise = np.random.normal(0, 1, size=2)
            current_pos -= lr * grad + noise * np.sqrt(2 * lr * noise_strength)
        
        all_paths.append(np.array(path_high_decay))
        
        # 2. SGLD with Low Polynomial Decay (gentle cooling)
        path_low_decay = []
        current_pos = start_point.copy()
        for step in range(num_steps):
            path_low_decay.append(current_pos.copy())
            
            # Low decay rate (lr drops slowly)
            lr = polynomial_schedule(step, num_steps, base_lr, decay_rate=1.0)
            
            grad = get_grad(current_pos[0], current_pos[1])
            noise = np.random.normal(0, 1, size=2)
            current_pos -= lr * grad + noise * np.sqrt(2 * lr * noise_strength)
        
        all_paths.append(np.array(path_low_decay))
        
        # 3. SGLD with Cosine Annealing (periodic exploration)
        path_cosine = []
        current_pos = start_point.copy()
        for step in range(num_steps):
            path_cosine.append(current_pos.copy())
            
            # Cosine annealing (lr oscillates)
            lr = cosine_annealing_schedule(step, num_steps, base_lr + 0.05)
            
            grad = get_grad(current_pos[0], current_pos[1])
            noise = np.random.normal(0, 1, size=2)
            current_pos -= lr * grad + noise * np.sqrt(2 * lr * noise_strength)
        
        all_paths.append(np.array(path_cosine))
        
        return all_paths


# Also create a custom legend for the schedule comparison
class MultimodalSGLDVisualizationWithLegend(MultimodalSGLDVisualization):
    """Extended version with detailed legend showing the schedules."""
    
    def create_legend(self) -> VGroup:
        """Create a detailed legend explaining the different schedules."""
        legend = VGroup()
        text_color = self.camera.background_color.invert()
        
        # Title
        title = Text("SGLD Step Size Schedules", font_size=20, color=text_color, weight=BOLD)
        legend.add(title)
        
        # Schedule descriptions
        descriptions = [
            "High Polynomial Decay: lr ∝ (1-t)²",
            "Low Polynomial Decay: lr ∝ (1-t)^0.5", 
            "Cosine Annealing: lr oscillates"
        ]
        
        for i, (label, color, desc) in enumerate(zip(self.path_labels, self.path_colors, descriptions)):
            # Color line
            line = Line(ORIGIN, RIGHT * 0.4, color=color, stroke_width=4)
            
            # Algorithm name
            algo_text = Text(label, font_size=16, color=text_color, weight=BOLD)
            
            # Description
            desc_text = Text(desc, font_size=12, color=text_color)
            
            # Arrange horizontally
            entry = VGroup(
                line,
                algo_text.next_to(line, RIGHT, buff=0.15),
                desc_text.next_to(algo_text, DOWN, buff=0.05, aligned_edge=LEFT)
            )
            
            legend.add(entry)
        
        # Arrange legend vertically with spacing
        legend.arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        
        # Position in top-left corner
        legend.to_corner(UL, buff=0.5)
        legend.scale(0.8)  # Make it a bit smaller
        
        return legend