from manim import *
import numpy as np
from newtonian_mechanics import create_newtonian_ball_updater
from langevin_dynamics import (
    create_langevin_ball_updater, 
    create_small_balls, 
    create_small_ball_updater,
    check_ball_collision,
    handle_wall_collision
)

class CombinedDynamics(Scene):
    def construct(self):
        # Configuration options:
        # Option 1: Newton (left) + Langevin invisible balls (right)
        # Option 2: Langevin invisible balls (left) + Langevin red balls (right)
        show_option = 2  # Change to 2 to debug the synchronized version
        
        # Set background color to white
        self.camera.background_color = WHITE

        # Parameters
        box_size = 2.5
        box_separation = 1.5
        num_small_balls = 200  # Reduced from 250 to 20 for testing
        large_ball_radius = 0.08
        small_ball_radius = 0.05  # Slightly larger for visibility
        
        # Create separate, but identically seeded, random number generators
        rng_seed = 42
        left_rng = np.random.RandomState(rng_seed)
        right_rng = np.random.RandomState(rng_seed)
        
        # Create two black square boxes
        left_box = Square(side_length=box_size * 2, color=BLACK, fill_opacity=0, stroke_width=3)
        left_box.shift(LEFT * (box_size + box_separation))
        
        right_box = Square(side_length=box_size * 2, color=BLACK, fill_opacity=0, stroke_width=3)
        right_box.shift(RIGHT * (box_size + box_separation))
        
        self.add(left_box, right_box)

        # Create the main blue balls (same starting position relative to their boxes)
        relative_start = np.array([-1.5, -0.8, 0])
        
        left_ball = Circle(radius=large_ball_radius, color=BLUE, fill_opacity=1)
        left_ball.move_to(left_box.get_center() + relative_start)
        
        right_ball = Circle(radius=large_ball_radius, color=BLUE, fill_opacity=1)
        right_ball.move_to(right_box.get_center() + relative_start)
        
        self.add(left_ball, right_ball)

        # Initialize velocities (same for both balls)
        initial_velocity = np.array([2.5, 1.8])
        left_velocity = [initial_velocity.copy()]
        right_velocity = [initial_velocity.copy()]

        if show_option == 1:
            # Option 1: Newton (left) + Langevin invisible balls (right)
            
            # Left side: Pure Newtonian mechanics
            left_ball.add_updater(create_newtonian_ball_updater(
                left_ball, left_velocity, large_ball_radius, 
                left_box.get_center()[:2], box_size
            ))
            
            # Right side: Langevin with invisible small balls
            right_small_balls, right_small_velocities = create_small_balls(
                num_small_balls, small_ball_radius, right_box.get_center()[:2], box_size,
                right_ball.get_center(), large_ball_radius, visible=False,
                rng=right_rng  # Pass the right RNG for consistency
            )
            
            for small_ball in right_small_balls:
                self.add(small_ball)
            
            right_ball.add_updater(create_langevin_ball_updater(
                right_ball, right_velocity, large_ball_radius, 
                right_box.get_center()[:2], box_size,
                right_small_balls, right_small_velocities
            ))
            
            for i, small_ball in enumerate(right_small_balls):
                small_ball.add_updater(create_small_ball_updater(
                    i, right_small_balls, right_small_velocities, small_ball_radius,
                    right_box.get_center()[:2], box_size,
                    random_state=right_rng  # Pass the right RNG to the updater
                ))

        else:  # show_option == 2
            # Option 2: Langevin invisible balls (left) + Langevin red balls (right)
            # SYNCHRONIZED APPROACH: Use dedicated RNG objects for perfect synchronization
            
            # Left side: Langevin with invisible small balls
            left_small_balls, left_small_velocities = create_small_balls(
                num_small_balls, small_ball_radius, left_box.get_center()[:2], box_size,
                left_ball.get_center(), large_ball_radius, visible=False,  # invisible
                rng=left_rng  # Pass the left RNG
            )
            
            for small_ball in left_small_balls:
                self.add(small_ball)
            
            left_ball.add_updater(create_langevin_ball_updater(
                left_ball, left_velocity, large_ball_radius, 
                left_box.get_center()[:2], box_size,
                left_small_balls, left_small_velocities
            ))
            
            for i, small_ball in enumerate(left_small_balls):
                small_ball.add_updater(create_small_ball_updater(
                    i, left_small_balls, left_small_velocities, small_ball_radius,
                    left_box.get_center()[:2], box_size,
                    random_state=left_rng  # Pass the left RNG to the updater
                ))

            # Right side: Langevin with visible red balls  
            right_small_balls, right_small_velocities = create_small_balls(
                num_small_balls, small_ball_radius, right_box.get_center()[:2], box_size,
                right_ball.get_center(), large_ball_radius, visible=True,  # visible
                rng=right_rng  # Pass the right RNG
            )
            
            for small_ball in right_small_balls:
                self.add(small_ball)
            
            right_ball.add_updater(create_langevin_ball_updater(
                right_ball, right_velocity, large_ball_radius, 
                right_box.get_center()[:2], box_size,
                right_small_balls, right_small_velocities
            ))
            
            for i, small_ball in enumerate(right_small_balls):
                small_ball.add_updater(create_small_ball_updater(
                    i, right_small_balls, right_small_velocities, small_ball_radius,
                    right_box.get_center()[:2], box_size,
                    random_state=right_rng  # Pass the right RNG to the updater
                ))

        # Run animation
        self.wait(30)  # Reduced from 30 to 15 seconds for faster testing