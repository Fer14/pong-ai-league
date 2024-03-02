import pygame
import math
import constants as c


class Paddle:
    def __init__(self, x, y, width, height, speed, field):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.speed = speed
        self.last_position = (x, y)
        self.field = field
        self.color = c.WHITE
        self.blocked = False

    def move(self, ball, **kwargs):
        if self.field == "left":
            self.move_left_field(ball, **kwargs)
        else:
            self.move_right_field(ball, **kwargs)

    def move_left_field(self, ball, **kwargs):
        pass

    def move_right_field(self, ball, **kwargs):
        pass

    def draw(self, win):
        if self.field == "left":
            border_top_left_radius = 10
            border_bottom_left_radius = 10
            border_top_right_radius = 0
            border_bottom_right_radius = 0
        else:
            border_top_left_radius = 0
            border_bottom_left_radius = 0
            border_top_right_radius = 10
            border_bottom_right_radius = 10

        if not self.blocked:
            pygame.draw.rect(
                win,
                self.color,
                (self.x, self.y, self.width, self.height),
                border_top_left_radius=border_top_left_radius,
                border_bottom_left_radius=border_bottom_left_radius,
                border_top_right_radius=border_top_right_radius,
                border_bottom_right_radius=border_bottom_right_radius,
            )
        else:
            pygame.draw.rect(
                win,
                "#6c757d",
                (self.x, self.y, self.width, self.height),
                border_top_left_radius=border_top_left_radius,
                border_bottom_left_radius=border_bottom_left_radius,
                border_top_right_radius=border_top_right_radius,
                border_bottom_right_radius=border_bottom_right_radius,
            )

    def rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def velocity(self):
        x_t, y_t = self.x, self.y
        x_t_1, y_t_1 = self.last_position
        # Calculate the velocity based on the change in y position
        return math.sqrt((x_t - x_t_1) ** 2 + (y_t - y_t_1) ** 2)

    def block(self):
        self.blocked = True

    def unblock(self):
        self.blocked = False

    def restart(self):
        self.x, self.y = (
            c.LEFT_PADDLE_INIT_POS if self.field == "left" else c.RIGHT_PADDLE_INIT_POS
        )
        self.last_position = (self.x, self.y)


class UserPaddle(Paddle):

    def __init__(self, x, y, width, height, speed, field):
        super().__init__(x, y, width, height, speed, field)
        self.name = "HUMAN"
        self.color = "#1a936f"  # "#c6dabf"
        self.logo = "../imgs/team_logos/user.png"

    def move_left_field(self, ball, **kwargs):
        keys = pygame.key.get_pressed()
        last_x, last_y = self.x, self.y
        if keys[pygame.K_w] and self.y > 0:
            self.y -= self.speed
        if keys[pygame.K_s] and self.y < c.HEIGHT + c.SCORE_HEIGT - c.PADDLE_HEIGHT:
            self.y += self.speed
        if keys[pygame.K_a] and self.x > 0:
            self.x -= self.speed
        if keys[pygame.K_d] and self.x < c.LINE_X[0] - c.LINE_WIDTH:
            self.x += self.speed

        self.last_position = (last_x, last_y)

    def move_right_field(self, ball, **kwargs):
        keys = pygame.key.get_pressed()
        last_x, last_y = self.x, self.y
        if keys[pygame.K_UP] and self.y > 0:
            self.y -= self.speed
        if keys[pygame.K_DOWN] and self.y < c.HEIGHT + c.SCORE_HEIGT - c.PADDLE_HEIGHT:
            self.y += self.speed
        if keys[pygame.K_LEFT] and self.x > c.LINE_X[0] + c.LINE_WIDTH:
            self.x -= self.speed
        if keys[pygame.K_RIGHT] and self.x < c.WIDTH - c.PADDLE_WIDTH:
            self.x += self.speed

        self.last_position = (last_x, last_y)
