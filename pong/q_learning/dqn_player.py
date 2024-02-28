import sys

sys.path.append("../")

import constants as c
from paddle import Paddle


class DQNPaddle(Paddle):

    def __init__(
        self,
        x,
        y,
        width,
        height,
        speed,
        field,
    ):
        super().__init__(x, y, width, height, speed, field)
        self.name = "DQN"
        self.color = "#ffd25a"  # "#c6dabf"
        self.logo = "../imgs/team_logos/dqn.png"

    def move_left_field(self, ball, **kwargs):

        decision = kwargs["move"]

        last_x, last_y = self.x, self.y

        if decision == "UP" and self.y > 0:
            self.y -= self.speed
        if decision == "DOWN" and self.y < c.HEIGHT + c.SCORE_HEIGT - c.PADDLE_HEIGHT:
            self.y += self.speed
        if decision == "LEFT" and self.x > 0:
            self.x -= self.speed
        if decision == "RIGHT" and self.x < c.LINE_X[0] - c.PADDLE_WIDTH:
            self.x += self.speed

        self.last_position = (last_x, last_y)

    def move_right_field(self, ball, **kwargs):

        decision = kwargs["move"]

        last_x, last_y = self.x, self.y

        if decision == "UP" and self.y > 0:
            self.y -= self.speed
        if decision == "DOWN" and self.y < c.HEIGHT + c.SCORE_HEIGT - c.PADDLE_HEIGHT:
            self.y += self.speed
        if decision == "LEFT" and self.x > c.LINE_X[0] + c.LINE_WIDTH:
            self.x -= self.speed
        if decision == "RIGHT" and self.x < c.WIDTH - c.PADDLE_WIDTH:
            self.x += self.speed

        self.last_position = (last_x, last_y)
