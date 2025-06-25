from manim import *
import numpy as np

def handle_wall_collision(pos, vel, radius, box_center, box_size):
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

def create_newtonian_ball_updater(ball, velocity_ref, ball_radius, box_center, box_size):
    """Create an updater for a Newtonian ball (no thermal interactions)"""
    def update_ball(mob, dt):
        # Current position
        current_pos = mob.get_center()[:2]
        
        # Update position
        new_pos = current_pos + velocity_ref[0] * dt
        new_pos, velocity_ref[0] = handle_wall_collision(new_pos, velocity_ref[0], ball_radius, box_center, box_size)
        
        # Move the ball
        mob.move_to([new_pos[0], new_pos[1], 0])
    
    return update_ball

class BouncingBall(Scene):
    def construct(self):
        # Set background color to white
        self.camera.background_color = WHITE

        # Create a black square box
        box = Square(side_length=6, color=BLACK, fill_opacity=0, stroke_width=3)
        self.add(box)

        # Create a blue ball (circle) inside the box
        ball = Circle(radius=0.1, color=BLUE, fill_opacity=1)
        ball.move_to(LEFT * 2 + DOWN * 1)  # Start near bottom-left inside the box
        self.add(ball)
    
        # Define velocity vector (constant speed)
        velocity = [np.array([3, 2, 0])]  # Wrap in list for reference
        ball_radius = 0.1
        box_center = np.array([0, 0])
        box_size = 3

        # Add updater
        ball.add_updater(create_newtonian_ball_updater(ball, velocity, ball_radius, box_center, box_size))

        self.wait(15)
