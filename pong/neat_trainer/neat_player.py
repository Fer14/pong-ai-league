import pickle
import sys
import os
import neat

sys.path.append("../")

import constants as c
from paddle import Paddle


class NeatPaddle(Paddle):

    def __init__(
        self, x, y, width, height, speed, field, train=False, training_speed=None
    ):
        super().__init__(x, y, width, height, speed, field)
        self.name = "NEAT PC"
        self.color = "#1a5e9d"  # "#c6dabf"
        self.logo = "../imgs/team_logos/neat.png"
        self.training_speed = training_speed
        self.decision_dict = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}
        self.train = train

        if not self.train:
            self.load_net()

    def load_net(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, "neat_config.txt")
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )
        with open("./neat_trainer/best_both_player.pickle", "rb") as f:
            genome = pickle.load(f)
            self.net = neat.nn.FeedForwardNetwork.create(genome, config)

    def mirror_decision(self, decision):
        if decision == "LEFT":
            return "RIGHT"
        elif decision == "RIGHT":
            return "LEFT"
        return decision

    def mirror_inputs(self, inputs):
        return (
            c.WIDTH - inputs[0],
            inputs[1],
            c.WIDTH - inputs[2],
            inputs[3],
            c.WIDTH - inputs[4],
            inputs[5],
        )

    def move_left_field(self, ball, **kwargs):

        if self.train:
            decision = kwargs["move"]
        else:
            output = self.net.activate(
                (self.x, self.y, ball.x, ball.y, ball.last_x, ball.last_y)
            )
            decision = self.decision_dict[output.index(max(output))]

        last_x, last_y = self.x, self.y

        if decision == "UP" and self.y > 0:
            self.y -= self.speed
        if decision == "DOWN" and self.y < c.HEIGHT + c.SCORE_HEIGT - c.PADDLE_HEIGHT:
            self.y += self.speed
        if decision == "LEFT" and self.x > 0:
            self.x -= self.speed
        if decision == "RIGHT" and self.x < c.LINE_X[0] - c.PADDLE_WIDTH - 15:
            self.x += self.speed

        self.last_position = (last_x, last_y)

    def move_right_field(self, ball, **kwargs):

        if self.train:
            decision = kwargs["move"]
        else:
            output = self.net.activate(
                self.mirror_inputs(
                    (self.x, self.y, ball.x, ball.y, ball.last_x, ball.last_y)
                )
            )
            decision = self.mirror_decision(
                self.decision_dict[output.index(max(output))]
            )

        last_x, last_y = self.x, self.y

        if decision == "UP" and self.y > 0:
            self.y -= self.speed
        if decision == "DOWN" and self.y < c.HEIGHT + c.SCORE_HEIGT - c.PADDLE_HEIGHT:
            self.y += self.speed
        if decision == "LEFT" and self.x > c.LINE_X[0] + c.LINE_WIDTH + 15:
            self.x -= self.speed
        if decision == "RIGHT" and self.x < c.WIDTH - c.PADDLE_WIDTH:
            self.x += self.speed

        self.last_position = (last_x, last_y)
