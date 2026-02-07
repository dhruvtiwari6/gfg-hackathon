from manim import *

class Classification(Scene):
    def construct(self):
        # Create blue dots on the left
        blue_dots = VGroup(*[Dot(color=BLUE, point=[x, y, 0]) for x in [-3, -2, -1] for y in [-1, 0, 1]])
        
        # Create red dots on the right
        red_dots = VGroup(*[Dot(color=RED, point=[x, y, 0]) for x in [1, 2, 3] for y in [-1, 0, 1]])
        
        # Create a yellow decision boundary line
        decision_boundary = Line(start=[-4, 0, 0], end=[4, 0, 0], color=YELLOW)
        
        # Add elements to the scene
        self.play(FadeIn(blue_dots), FadeIn(red_dots))
        self.play(Create(decision_boundary))
        self.wait(1)