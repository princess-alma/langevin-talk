from manim import *
import numpy as np
from newtonian_mechanics import create_newtonian_ball_updater
from langevin_dynamics import (
    create_langevin_ball_updater, 
    create_small_balls, 
    check_ball_collision,
    handle_wall_collision
)

class CombinedDynamics(Scene):
    def construct(self):
        # Configuration options:
        # Option 1: Newton (left) + Langevin invisible balls (right)
        # Option 2: Langevin invisible balls (left) + Langevin red balls (right)
        show_option = 2  # Change to 2 for the second option
        
        # Set background color to white
        self.camera.background_color = WHITE

        # Parameters
        box_size = 2.5
        box_separation = 1.5
        num_small_balls = 250
        large_ball_radius = 0.08
        small_ball_radius = 0.04
        
        # IMPORTANT: Set random seed for reproducible small ball positions
        np.random.seed(42)
        
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
                right_ball.get_center(), large_ball_radius, visible=False
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
                    right_box.get_center()[:2], box_size
                ))

        else:  # show_option == 2
            # Option 2: Langevin invisible balls (left) + Langevin red balls (right)
            
            # Generate small ball positions and velocities ONCE for both sides
            template_small_balls, template_small_velocities = create_small_balls(
                num_small_balls, small_ball_radius, np.array([0, 0]), box_size,
                relative_start, large_ball_radius, visible=False
            )
            
            # Create shared thermal noise arrays that will be updated synchronously
            shared_thermal_noise = [np.zeros(2) for _ in range(num_small_balls)]
            shared_random_state = np.random.RandomState(123)
            
            # Left side: Langevin with invisible small balls (using template)
            left_small_balls = []
            left_small_velocities = []
            
            for i, template_ball in enumerate(template_small_balls):
                # Create invisible ball at corresponding position in left box
                template_pos = template_ball.get_center()
                left_pos = template_pos + left_box.get_center()
                
                left_small_ball = Circle(radius=small_ball_radius, color=RED, fill_opacity=0, stroke_opacity=0)
                left_small_ball.move_to(left_pos)
                left_small_balls.append(left_small_ball)
                self.add(left_small_ball)
                
                # Copy velocity
                left_small_velocities.append(template_small_velocities[i].copy())
            
            # Right side: Langevin with visible red balls (using same template)
            right_small_balls = []
            right_small_velocities = []
            
            for i, template_ball in enumerate(template_small_balls):
                # Create visible ball at corresponding position in right box
                template_pos = template_ball.get_center()
                right_pos = template_pos + right_box.get_center()
                
                right_small_ball = Circle(radius=small_ball_radius, color=RED, fill_opacity=0.6, stroke_width=0.5)
                right_small_ball.move_to(right_pos)
                right_small_balls.append(right_small_ball)
                self.add(right_small_ball)
                
                # Copy velocity
                right_small_velocities.append(template_small_velocities[i].copy())
            
            # Create a thermal noise updater that generates noise for all balls at once
            def update_thermal_noise(mob, dt):
                for i in range(num_small_balls):
                    shared_thermal_noise[i] = shared_random_state.normal(0, 0.4, 2)
            
            # Add invisible object just for thermal noise generation
            thermal_noise_generator = Dot().set_opacity(0)
            thermal_noise_generator.add_updater(update_thermal_noise)
            self.add(thermal_noise_generator)
            
            # Modified small ball updater that uses pre-generated thermal noise
            def create_synchronized_small_ball_updater(ball_index, small_balls, small_velocities, 
                                                     small_ball_radius, box_center, box_size):
                def update_small_ball(mob, dt):
                    # Current position
                    current_pos = mob.get_center()[:2]
                    
                    # Use pre-generated thermal noise
                    small_velocities[ball_index] += shared_thermal_noise[ball_index] * dt
                    
                    # Limit maximum speed to prevent runaway velocities
                    speed = np.linalg.norm(small_velocities[ball_index])
                    if speed > 5.0:
                        small_velocities[ball_index] = small_velocities[ball_index] / speed * 5.0
                    
                    # Check collisions with other small balls (limited for performance)
                    for j in range(max(0, ball_index - 2), min(len(small_balls), ball_index + 3)):
                        if j != ball_index:
                            other_pos = small_balls[j].get_center()[:2]
                            small_velocities[ball_index], small_velocities[j] = check_ball_collision(
                                current_pos, small_velocities[ball_index], small_ball_radius,
                                other_pos, small_velocities[j], small_ball_radius
                            )
                    
                    # Update position
                    new_pos = current_pos + small_velocities[ball_index] * dt
                    new_pos, small_velocities[ball_index] = handle_wall_collision(
                        new_pos, small_velocities[ball_index], small_ball_radius, box_center, box_size)
                    
                    # Move the ball
                    mob.move_to([new_pos[0], new_pos[1], 0])
                
                return update_small_ball
            
            # Add updaters for left side (invisible)
            left_ball.add_updater(create_langevin_ball_updater(
                left_ball, left_velocity, large_ball_radius, 
                left_box.get_center()[:2], box_size,
                left_small_balls, left_small_velocities
            ))
            
            for i, small_ball in enumerate(left_small_balls):
                small_ball.add_updater(create_synchronized_small_ball_updater(
                    i, left_small_balls, left_small_velocities, small_ball_radius,
                    left_box.get_center()[:2], box_size
                ))
            
            # Add updaters for right side (visible)
            right_ball.add_updater(create_langevin_ball_updater(
                right_ball, right_velocity, large_ball_radius, 
                right_box.get_center()[:2], box_size,
                right_small_balls, right_small_velocities
            ))
            
            for i, small_ball in enumerate(right_small_balls):
                small_ball.add_updater(create_synchronized_small_ball_updater(
                    i, right_small_balls, right_small_velocities, small_ball_radius,
                    right_box.get_center()[:2], box_size
                ))

        # Run animation
        self.wait(30)