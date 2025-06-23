from manim import *
import numpy as np
from newtonian_mechanics import handle_wall_collision

def check_ball_collision(pos1, vel1, r1, pos2, vel2, r2, mass1=1.0, mass2=1.0):
    """Check and handle elastic collision between two balls"""
    distance = np.linalg.norm(pos1 - pos2)
    min_distance = r1 + r2
    
    if distance < min_distance and distance > 0.01:
        # Collision normal
        normal = (pos1 - pos2) / distance
        
        # Relative velocity
        rel_vel = vel1 - vel2
        vel_along_normal = np.dot(rel_vel, normal)
        
        # Don't resolve if velocities are separating
        if vel_along_normal > 0:
            return vel1, vel2
        
        # Elastic collision with mass consideration
        impulse = 2 * vel_along_normal / (mass1 + mass2)
        new_vel1 = vel1 - impulse * mass2 * normal
        new_vel2 = vel2 + impulse * mass1 * normal
        
        return new_vel1, new_vel2
    
    return vel1, vel2

def create_langevin_ball_updater(ball, velocity_ref, ball_radius, box_center, box_size, 
                                small_balls, small_velocities, large_ball_mass=20.0):
    """Create an updater for a Langevin ball (with thermal interactions)"""
    def update_ball(mob, dt):
        # Current position
        current_pos = mob.get_center()[:2]
        
        # Check collisions with small balls
        for i, small_ball in enumerate(small_balls):
            small_pos = small_ball.get_center()[:2]
            velocity_ref[0], small_velocities[i] = check_ball_collision(
                current_pos, velocity_ref[0], ball_radius,
                small_pos, small_velocities[i], small_ball.width / 2,
                mass1=large_ball_mass, mass2=1.0
            )
        
        # Update position
        new_pos = current_pos + velocity_ref[0] * dt
        new_pos, velocity_ref[0] = handle_wall_collision(new_pos, velocity_ref[0], ball_radius, box_center, box_size)
        
        # Move the ball
        mob.move_to([new_pos[0], new_pos[1], 0])
    
    return update_ball

def create_small_ball_updater(ball_index, small_balls, small_velocities, small_ball_radius, 
                             box_center, box_size, random_state=None):
    """Create an updater for a small thermal ball"""
    def update_small_ball(mob, dt):
        # Current position
        current_pos = mob.get_center()[:2]
        
        # Add small random thermal motion (use shared random state if provided)
        if random_state is not None:
            thermal_noise = random_state.normal(0, 0.4, 2)
        else:
            thermal_noise = np.random.normal(0, 0.4, 2)
        small_velocities[ball_index] += thermal_noise * dt
        
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

def create_small_balls(num_small_balls, small_ball_radius, box_center, box_size, 
                      large_ball_position, large_ball_radius, visible=True):
    """Create small thermal balls with random positions and velocities"""
    small_balls = []
    small_velocities = []
    
    for i in range(num_small_balls):
        # Random position inside the box
        while True:
            x = np.random.uniform(-box_size + small_ball_radius + 0.05, 
                                box_size - small_ball_radius - 0.05)
            y = np.random.uniform(-box_size + small_ball_radius + 0.05, 
                                box_size - small_ball_radius - 0.05)
            pos = np.array([x + box_center[0], y + box_center[1], 0])
            
            # Ensure small balls don't start too close to large ball
            if np.linalg.norm(pos - large_ball_position) > large_ball_radius + small_ball_radius + 0.15:
                # Check distance from other small balls
                too_close = False
                for existing_ball in small_balls:
                    if np.linalg.norm(pos - existing_ball.get_center()) < 2 * small_ball_radius + 0.05:
                        too_close = True
                        break
                if not too_close:
                    break
        
        # Set color and opacity based on visibility parameter
        if visible:
            small_ball = Circle(radius=small_ball_radius, color=RED, fill_opacity=0.6, stroke_width=0.5)
        else:
            small_ball = Circle(radius=small_ball_radius, color=RED, fill_opacity=0, stroke_opacity=0)
        
        small_ball.move_to(pos)
        small_balls.append(small_ball)
        
        # Random velocity for thermal motion
        speed = np.random.uniform(1.5, 3.5)
        angle = np.random.uniform(0, 2 * np.pi)
        velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
        small_velocities.append(velocity)
    
    return small_balls, small_velocities

class LangevinDynamics(Scene):
    def construct(self):
        # Top-level parameter to control small ball visibility
        small_balls_visible = True  # Set to False to make small balls transparent
        
        # Set background color to white
        self.camera.background_color = WHITE

        # Parameters
        box_size = 3
        num_small_balls = 80
        large_ball_radius = 0.1
        small_ball_radius = 0.06
        
        # Create a black square box
        box = Square(side_length=box_size * 2, color=BLACK, fill_opacity=0, stroke_width=3)
        self.add(box)

        # Create the main blue ball
        large_ball = Circle(radius=large_ball_radius, color=BLUE, fill_opacity=1)
        large_ball.move_to(LEFT * 2 + DOWN * 1)
        self.add(large_ball)

        # Create small balls
        small_balls, small_velocities = create_small_balls(
            num_small_balls, small_ball_radius, box.get_center()[:2], box_size,
            large_ball.get_center(), large_ball_radius, small_balls_visible
        )
        
        for small_ball in small_balls:
            self.add(small_ball)

        # Initialize velocity
        large_velocity = [np.array([3.0, 2.0])]  # Wrap in list for reference

        # Add updaters
        large_ball.add_updater(create_langevin_ball_updater(
            large_ball, large_velocity, large_ball_radius, box.get_center()[:2], box_size,
            small_balls, small_velocities
        ))
        
        random_state = np.random.RandomState(0)  # Shared random state for reproducibility
        for i, small_ball in enumerate(small_balls):
            small_ball.add_updater(create_small_ball_updater(
                i, small_balls, small_velocities, small_ball_radius, 
                box.get_center()[:2], box_size, random_state=random_state
            ))

        # Run animation
        self.wait(15)