import sys
import torch
import torch.nn as nn
import random

sys.path.append("../")

import constants as c
from paddle import Paddle


class PGPaddle(Paddle):

    def __init__(self, x, y, width, height, speed, field, train=True):
        super().__init__(x, y, width, height, speed, field)
        self.name = "PG"
        self.color = "#ef3036"  # "#c6dabf"
        self.logo = "../imgs/team_logos/pg.png"
        self.decision_dict = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}

        self.train = train
        if not train:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.load_net()

    def load_net(self):
        # load pth model with torch
        model_state_dict = torch.load(
            "/home/fer/Escritorio/pong-ai-league/pong/policy_gradient/checkpoints/2024-03-23/model.pth"
        )
        model = model = nn.Sequential(
            nn.Linear(6, 10), nn.ReLU(), nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5)
        ).to(self.device)

        for param in model.parameters():
            param.requires_grad = False
        self.model = model
        self.model.load_state_dict(model_state_dict)

    def training_restart(self):
        self.x = (
            c.LEFT_PADDLE_INIT_POS[0]
            if self.field == "left"
            else c.RIGHT_PADDLE_INIT_POS[0]
        )
        y1 = (c.HEIGHT - c.PADDLE_HEIGHT) // 2 - 100
        y2 = (c.HEIGHT - c.PADDLE_HEIGHT) // 2 + 100
        self.y = random.randint(y1, y2)
        self.last_position = (self.x, self.y)

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
            with torch.no_grad():
                state = torch.FloatTensor([inputs]).to(self.device)
                logits = self.model(state)
            prob_dist = torch.distributions.Categorical(logits=logits)
            decision = self.decision_dict[prob_dist.sample().item()]

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
            inputs = (
                self.x,
                self.y,
                ball.x,
                ball.y,
                ball.last_x,
                ball.last_y,
            )
            inputs = self.mirror_inputs(inputs)
            logits = self.model(inputs)
            prob_dist = torch.distributions.Categorical(logits=logits)
            decision = self.mirror_decision(
                self.decision_dict[prob_dist.sample().item()]
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
