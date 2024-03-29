from heuristics_united import HeuristicsUnited
from paddle import UserPaddle
from pong import PongGame
from neat_trainer.neat_player import NeatPaddle
import constants as c
from ball import RealPhysicsBall
from scorer import Scorer
from q_learning.dqn_player import DQNPaddle
from policy_gradient.pg_player import PGPaddle
import pygame


def main():

    pong = PongGame(default_pong=True)
    paddle1 = PGPaddle(
        x=c.LEFT_PADDLE_INIT_POS[0],
        y=c.LEFT_PADDLE_INIT_POS[1],
        width=c.PADDLE_WIDTH,
        height=c.PADDLE_HEIGHT,
        speed=c.PADDLE_SPEED,
        field="left",
        train=False,
    )
    paddle2 = HeuristicsUnited(
        x=c.RIGHT_PADDLE_INIT_POS[0],
        y=c.RIGHT_PADDLE_INIT_POS[1],
        width=c.PADDLE_WIDTH,
        height=c.PADDLE_HEIGHT,
        speed=c.PADDLE_SPEED,
        field="right",
    )

    ball = RealPhysicsBall(
        c.BALL_INIT_POS[0],
        c.BALL_INIT_POS[1],
        c.BALL_RADIUS,
        ball_init_speeds_x=[2, -2],
        ball_init_speeds_y=[0.15, -0.15, 0.25, -0.25, 0.5, -0.5],
    )

    scorer = Scorer(
        width=c.WIDTH,
        height=c.HEIGHT,
        left_color=paddle1.color,
        right_color=paddle2.color,
        left_name=paddle1.name,
        right_name=paddle2.name,
        left_logo=paddle1.logo,
        right_logo=paddle2.logo,
    )

    pong.init_pong(
        paddle1=paddle1,
        paddle2=paddle2,
        ball=ball,
        scorer=scorer,
    )
    pong.play()
    pygame.quit()


if __name__ == "__main__":
    main()
