from manim import *

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
        velocity = np.array([3, 2, 0])  # units per second

        # Duration and frame rate
        run_time = 10
        dt = 1 / self.camera.frame_rate

        def update_ball(mob, dt):
            nonlocal velocity
            # Update position
            mob.shift(velocity * dt)

            # Get current position and ball radius
            pos = mob.get_center()
            ball_radius = mob.width / 2

            # Box boundaries (half side length = 3, adjusted for ball radius)
            left_bound = -3 + ball_radius
            right_bound = 3 - ball_radius
            bottom_bound = -3 + ball_radius
            top_bound = 3 - ball_radius

            # Bounce off vertical walls with position correction
            if pos[0] <= left_bound:
                velocity[0] = abs(velocity[0])  # Ensure velocity is positive (moving right)
                mob.move_to([left_bound, pos[1], pos[2]])  # Correct position
            elif pos[0] >= right_bound:
                velocity[0] = -abs(velocity[0])  # Ensure velocity is negative (moving left)
                mob.move_to([right_bound, pos[1], pos[2]])  # Correct position

            # Bounce off horizontal walls with position correction
            if pos[1] <= bottom_bound:
                velocity[1] = abs(velocity[1])  # Ensure velocity is positive (moving up)
                mob.move_to([pos[0], bottom_bound, pos[2]])  # Correct position
            elif pos[1] >= top_bound:
                velocity[1] = -abs(velocity[1])  # Ensure velocity is negative (moving down)
                mob.move_to([pos[0], top_bound, pos[2]])  # Correct position

        ball.add_updater(update_ball)

        self.wait(run_time)
