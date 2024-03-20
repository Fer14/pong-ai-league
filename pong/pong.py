import pygame
from ball import Ball, RealPhysicsBall, FunnierPhysicsBall
from paddle import Paddle, UserPaddle
from scorer import Scorer
import constants as c


class PongGame:

    def __init__(self, display=True, default_pong=True, logo="../imgs/big_logo_2.png"):
        self.display = display
        if self.display:
            pygame.init()
            self.screen = pygame.display.set_mode((c.WIDTH, c.HEIGHT))
            pygame.display.set_caption("PONG AI LEAGUE")
            logo_img = pygame.transform.smoothscale(
                pygame.image.load(logo),
                (50, 32),
            )

            # Set the window icon
            pygame.display.set_icon(logo_img)
        if default_pong:
            self.init_default_pong()

    def init_default_pong(self):

        self.paddle1 = UserPaddle(
            x=c.LEFT_PADDLE_INIT_POS[0],
            y=c.LEFT_PADDLE_INIT_POS[1],
            width=c.PADDLE_WIDTH,
            height=c.PADDLE_HEIGHT,
            speed=c.PADDLE_SPEED,
            field="left",
        )
        self.paddle2 = UserPaddle(
            x=c.RIGHT_PADDLE_INIT_POS[0],
            y=c.RIGHT_PADDLE_INIT_POS[1],
            width=c.PADDLE_WIDTH,
            height=c.PADDLE_HEIGHT,
            speed=c.PADDLE_SPEED,
            field="right",
        )
        self.ball = RealPhysicsBall(
            c.BALL_INIT_POS[0],
            c.BALL_INIT_POS[1],
            c.BALL_RADIUS,
        )
        self.scorer = Scorer(
            width=c.WIDTH,
            height=c.HEIGHT,
            left_color=self.paddle1.color,
            right_color=self.paddle2.color,
            left_name=self.paddle1.name,
            right_name=self.paddle2.name,
            left_logo=self.paddle1.logo,
            right_logo=self.paddle2.logo,
        )

    def init_pong(self, paddle1, paddle2, ball, scorer):
        self.paddle1 = paddle1
        self.paddle2 = paddle2
        self.ball = ball
        self.scorer = scorer

    def restart(self):
        self.ball.restart(paddle_left=self.paddle1, paddle_right=self.paddle2)
        self.paddle1.restart()
        self.paddle2.restart()
        self.scorer.restart()

    def draw(self):
        self.screen.fill(c.BLACK)
        pygame.draw.line(
            self.screen,
            c.WHITE,
            (c.LINE_X[0] + 5, c.LINE_X[1]),
            (c.LINE_Y[0] + 5, c.LINE_Y[1]),
            3,
        )

        self.paddle1.draw(self.screen)
        self.paddle2.draw(self.screen)
        self.ball.draw(self.screen)
        self.scorer.draw(self.screen)
        pygame.display.update()

    def step(self):
        self.paddle1.move(self.ball)
        self.paddle2.move(self.ball)
        self.ball.move(self.paddle1, self.paddle2, self.scorer)
        if self.display:
            self.draw()

    def play(self):
        # Game loop
        clock = pygame.time.Clock()
        running = True
        while running:
            clock.tick(60)  # Cap the frame rate

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.step()
