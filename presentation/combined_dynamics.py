from manim import *
import numpy as np

class CombinedDynamics(Scene):
    def construct(self):
        # Top-level parameter to control small ball visibility
        small_balls_visible = False  # Set to False to make small balls transparent
        
        # Set background color to white
        self.camera.background_color = WHITE

        # Parameters
        box_size = 2.5  # Slightly smaller boxes to fit side by side
        box_separation = 1.5  # Space between the two boxes
        num_small_balls = 250  # Increased from 40
        large_ball_radius = 0.1  # Keep same size
        small_ball_radius = 0.06  # Increased from 0.04
        
        # Create two black square boxes
        left_box = Square(side_length=box_size * 2, color=BLACK, fill_opacity=0, stroke_width=3)
        left_box.shift(LEFT * (box_size + box_separation))
        
        right_box = Square(side_length=box_size * 2, color=BLACK, fill_opacity=0, stroke_width=3)
        right_box.shift(RIGHT * (box_size + box_separation))
        
        self.add(left_box, right_box)

        # Create the main blue balls (same starting position relative to their boxes)
        # Starting position: left side, bottom of each box
        relative_start = np.array([-1.5, -0.8, 0])  # Relative to box center
        
        left_ball = Circle(radius=large_ball_radius, color=BLUE, fill_opacity=1)
        left_ball.move_to(left_box.get_center() + relative_start)
        
        right_ball = Circle(radius=large_ball_radius, color=BLUE, fill_opacity=1)
        right_ball.move_to(right_box.get_center() + relative_start)
        
        self.add(left_ball, right_ball)

        # Create small balls only for the right (Langevin) box
        small_balls = []
        right_box_center = right_box.get_center()
        
        for i in range(num_small_balls):
            # Random position inside the right box only
            while True:
                x = np.random.uniform(-box_size + small_ball_radius + 0.05, 
                                    box_size - small_ball_radius - 0.05)
                y = np.random.uniform(-box_size + small_ball_radius + 0.05, 
                                    box_size - small_ball_radius - 0.05)
                pos = np.array([x, y, 0]) + right_box_center
                
                # Ensure small balls don't start too close to large ball
                if np.linalg.norm(pos - right_ball.get_center()) > large_ball_radius + small_ball_radius + 0.15:
                    # Check distance from other small balls
                    too_close = False
                    for existing_ball in small_balls:
                        if np.linalg.norm(pos - existing_ball.get_center()) < 2 * small_ball_radius + 0.05:
                            too_close = True
                            break
                    if not too_close:
                        break
            
            # Set color and opacity based on visibility parameter
            if small_balls_visible:
                small_ball = Circle(radius=small_ball_radius, color=RED, fill_opacity=0.6, stroke_width=0.5)
            else:
                small_ball = Circle(radius=small_ball_radius, color=RED, fill_opacity=0, stroke_opacity=0)
            
            small_ball.move_to(pos)
            small_balls.append(small_ball)
            self.add(small_ball)

        # Initialize velocities (same for both balls)
        initial_velocity = np.array([2.5, 1.8])  # Same starting velocity
        left_velocity = initial_velocity.copy()
        right_velocity = initial_velocity.copy()
        
        # Random velocities for small balls (thermal motion)
        small_velocities = []
        for _ in range(num_small_balls):
            speed = np.random.uniform(1.5, 3.5)
            angle = np.random.uniform(0, 2 * np.pi)
            velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
            small_velocities.append(velocity)

        def handle_wall_collision(pos, vel, radius, box_center):
            """Handle wall collisions and return corrected position and velocity"""
            new_pos = pos.copy()
            new_vel = vel.copy()
            
            # Calculate boundaries relative to box center
            left_bound = box_center[0] - box_size + radius
            right_bound = box_center[0] + box_size - radius
            bottom_bound = box_center[1] - box_size + radius
            top_bound = box_center[1] + box_size - radius
            
            # Left and right walls
            if pos[0] <= left_bound:
                new_vel[0] = abs(new_vel[0])
                new_pos[0] = left_bound
            elif pos[0] >= right_bound:
                new_vel[0] = -abs(new_vel[0])
                new_pos[0] = right_bound
            
            # Top and bottom walls
            if pos[1] <= bottom_bound:
                new_vel[1] = abs(new_vel[1])
                new_pos[1] = bottom_bound
            elif pos[1] >= top_bound:
                new_vel[1] = -abs(new_vel[1])
                new_pos[1] = top_bound
                
            return new_pos, new_vel

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

        def update_left_ball(mob, dt):
            """Update the Newtonian ball (left side) - no thermal interactions"""
            nonlocal left_velocity
            
            # Current position
            current_pos = mob.get_center()[:2]
            
            # Update position
            new_pos = current_pos + left_velocity * dt
            new_pos, left_velocity = handle_wall_collision(new_pos, left_velocity, large_ball_radius, left_box.get_center()[:2])
            
            # Move the ball
            mob.move_to([new_pos[0], new_pos[1], 0])

        def update_right_ball(mob, dt):
            """Update the Langevin ball (right side) - with thermal interactions"""
            nonlocal right_velocity
            
            # Current position
            current_pos = mob.get_center()[:2]
            
            # Check collisions with small balls
            for i, small_ball in enumerate(small_balls):
                small_pos = small_ball.get_center()[:2]
                right_velocity, small_velocities[i] = check_ball_collision(
                    current_pos, right_velocity, large_ball_radius,
                    small_pos, small_velocities[i], small_ball_radius,
                    mass1=20.0, mass2=1.0  # Heavy blue ball
                )
            
            # Update position
            new_pos = current_pos + right_velocity * dt
            new_pos, right_velocity = handle_wall_collision(new_pos, right_velocity, large_ball_radius, right_box.get_center()[:2])
            
            # Move the ball
            mob.move_to([new_pos[0], new_pos[1], 0])

        def create_small_ball_updater(ball_index):
            def update_small_ball(mob, dt):
                # Current position
                current_pos = mob.get_center()[:2]
                
                # Add small random thermal motion
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
                    new_pos, small_velocities[ball_index], small_ball_radius, right_box.get_center()[:2])
                
                # Move the ball
                mob.move_to([new_pos[0], new_pos[1], 0])
            
            return update_small_ball

        # Add updaters
        left_ball.add_updater(update_left_ball)
        right_ball.add_updater(update_right_ball)
        
        for i, small_ball in enumerate(small_balls):
            small_ball.add_updater(create_small_ball_updater(i))

        # Run animation
        self.wait(30)