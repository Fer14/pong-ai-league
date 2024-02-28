from paddle import Paddle
import pygame
import constants as c


class HeuristicsUnited(Paddle):
    def __init__(self, x, y, width, height, speed, field):
        super().__init__(x, y, width, height, speed, field)
        self.name = "HEURISTICS UNITED"
        self.color = "#f55d98"
        self.logo = "../imgs/team_logos/heuristics.png"

    def move_left_field(self, ball, **kwargs):
        keys = pygame.key.get_pressed()
        last_x, last_y = self.x, self.y

        middle = self.y + self.height / 2

        if ball.y < middle and middle > 0:
            self.y -= self.speed
        if ball.y > middle and middle < c.HEIGHT + c.SCORE_HEIGT - self.height:
            self.y += self.speed
        self.last_position = (last_x, last_y)

    def move_right_field(self, ball, **kwargs):
        last_x, last_y = self.x, self.y

        middle = self.y + self.height / 2

        if ball.y < middle and middle > 0:
            self.y -= self.speed
        if ball.y > middle and middle < c.HEIGHT + c.SCORE_HEIGT - self.height:
            self.y += self.speed

        self.last_position = (last_x, last_y)
