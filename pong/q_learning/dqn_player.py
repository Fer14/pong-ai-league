import sys

sys.path.append("../")

import constants as c
from paddle import Paddle
import torch
import torch.nn as nn
import numpy as np


class DQNPaddle(Paddle):

    def __init__(self, x, y, width, height, speed, field, train=True):
        super().__init__(x, y, width, height, speed, field)
        self.name = "DQN"
        self.color = "#ffd25a"  # "#c6dabf"
        self.logo = "../imgs/team_logos/dqn.png"
        self.decision_dict = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}

        self.train = train
        if not train:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.load_net()

    def load_net(self):
        # load pth model with torch
        model_state_dict = torch.load(
            "/home/fer/Escritorio/pong-ai-league/pong/q_learning/checkpoints/2024-03-20/model.pth"
        )
        model = nn.Sequential(
            nn.Linear(6, 6),
            nn.ReLU(),
            nn.Linear(6, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
        ).to(self.device)

        for param in model.parameters():
            param.requires_grad = False
        self.model = model
        self.model.load_state_dict(model_state_dict)

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
            inputs = (
                self.x,
                self.y,
                ball.x,
                ball.y,
                ball.last_x,
                ball.last_y,
            )
            inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
            decision = self.decision_dict[np.argmax(self.model(inputs).cpu().numpy())]

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
            inputs = self.mirror_inputs(
                (self.x, self.y, ball.x, ball.y, ball.last_x, ball.last_y)
            )
            inputs = torch.tensor(inputs, dtype=torch.float32)
            decision = self.mirror_decision(
                self.decision_dict[self.model(inputs).argmax().item()]
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
