from manim import *

class Classification(Scene):
    def construct(self):
        # Add blue dots on the left
        blue_dots = VGroup(
            *[Dot(color=BLUE, point=[x, y, 0]) for x in [-3, -2, -1] for y in [-1, 0, 1]]
        )
        self.play(FadeIn(blue_dots))

        # Add red dots on the right
        red_dots = VGroup(
            *[Dot(color=RED, point=[x, y, 0]) for x in [1, 2, 3] for y in [-1, 0, 1]]
        )
        self.play(FadeIn(red_dots))

        # Add a yellow decision boundary line
        decision_boundary = Line(LEFT * 3.5, RIGHT * 3.5, color=YELLOW)
        self.play(Create(decision_boundary))

        self.wait(2)