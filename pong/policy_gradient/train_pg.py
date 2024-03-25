import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import sys
from tqdm import tqdm
import pygame
from datetime import date
import os
from pg_player import PGPaddle

sys.path.append("../")

from pong import PongGame
from ball import RealPhysicsBall
from scorer import Scorer
import constants as c


DISCOUNT = 0.95


class BaselineAgent:
    def __init__(self):
        pass

    def act(self, state):
        # Baseline agent moves paddle up if the ball is above the paddle, down if it's below
        paddle_y = state[1]
        ball_y = state[3]
        if paddle_y < ball_y:
            return 0  # Move up
        elif paddle_y > ball_y:
            return 1  # Move down
        else:
            return np.random.choice(2)


class PGAgent:

    def __init__(self, device):
        # Main model
        self.device = device
        self.model = self.create_model().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)  # Adam

        os.makedirs(f"checkpoints/{date.today()}", exist_ok=True)
        self.onpolicy_reset()

    def onpolicy_reset(self):
        self.log_probs = []

    def create_model(self):
        model = nn.Sequential(
            nn.Linear(6, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 4),
            # nn.Softmax(dim=-1),
        )
        # nn.init.normal_(model[-1].weight, std=0.1)
        model[-1].weight.data.fill_(0.0)
        model.add_module("softmax", nn.Softmax(dim=-1))
        return model

    def forward(self, x):
        return self.model(x)

    def act(self, state, save_logs=True):
        state = torch.FloatTensor(state).to(self.device)
        action_probs = self.forward(state)
        prob_dist = torch.distributions.Categorical(action_probs)
        action = prob_dist.sample().to(self.device)
        if save_logs:
            self.log_probs.append(prob_dist.log_prob(action))

        return action.item()

    def train(self, r, baselines):
        self.optimizer.zero_grad()

        log_probs = torch.stack(self.log_probs).to(self.device)
        r = self.disccount_n_standarise(r)
        rewards = torch.tensor(r).view(-1, 1).to(self.device)

        # Calculate advantages using baselines
        advantages = rewards - torch.tensor(baselines, dtype=torch.float32).view(
            -1, 1
        ).to(self.device)

        loss = torch.sum(-(log_probs * advantages))
        loss.backward()
        self.optimizer.step()
        self.onpolicy_reset()

        return loss.item()

    def discount_reward(self, rewards):
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * DISCOUNT + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def disccount_n_standarise(self, r):
        dr = self.discount_reward(r)
        dr -= np.mean(dr)
        dr /= np.std(dr) if np.std(dr) > 0 else 1
        return dr


class PongGamePGTraining(PongGame):
    def __init__(
        self, default_pong=True, logo="../../imgs/big_logo_2.png", display=True
    ):
        super().__init__(display=display, default_pong=default_pong, logo=logo)
        self.decision_dict = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    def restart_pg(self):
        self.ball.restart(paddle_left=self.paddle1, paddle_right=self.paddle2)
        self.paddle1.restart()
        self.paddle2.restart()
        self.scorer.restart()

        return self.state()

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

    def state(self):
        return (
            self.paddle1.x,
            self.paddle1.y,
            self.ball.x,
            self.ball.y,
            self.ball.last_x,
            self.ball.last_y,
        )

    def init_pg_learning(self):
        self.init_pong(
            PGPaddle(
                x=c.LEFT_PADDLE_INIT_POS[0],
                y=c.LEFT_PADDLE_INIT_POS[1],
                width=c.PADDLE_WIDTH,
                height=c.PADDLE_HEIGHT,
                speed=c.PADDLE_SPEED,
                field="left",
                train=True,
            ),
            PGPaddle(
                x=c.RIGHT_PADDLE_INIT_POS[0],
                y=c.RIGHT_PADDLE_INIT_POS[1],
                width=c.PADDLE_WIDTH,
                height=c.PADDLE_HEIGHT,
                speed=c.PADDLE_SPEED,
                field="right",
                train=True,
            ),
            RealPhysicsBall(
                c.BALL_INIT_POS[0],
                c.BALL_INIT_POS[1],
                c.BALL_RADIUS,
                ball_init_speeds_x=[1.5, -1.5],
                ball_init_speeds_y=[0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1, -1],
                # training_left=True,
            ),
            Scorer(
                width=c.WIDTH,
                height=c.HEIGHT,
                left_color="#ef3036",
                right_color="#ef3036",
                left_name="PG",
                right_name="PG",
                font="../../font/Gliker-Bold.ttf",
                logo="../../imgs/paddle.png",
                left_logo="../../imgs/team_logos/pg.png",
                right_logo="../../imgs/team_logos/pg.png",
                display=self.display,
            ),
        )
        return self.state()

    def step_pg(self, action, agent2):
        right_curr_score = self.scorer.right_score
        left_curr_score = self.scorer.left_score
        decision1 = self.decision_dict[action]
        agent2_input = self.mirror_inputs(
            (
                self.paddle2.x,
                self.paddle2.y,
                self.ball.x,
                self.ball.y,
                self.ball.last_x,
                self.ball.last_y,
            )
        )
        decision2 = agent2.act(agent2_input, save_logs=False)
        self.paddle1.move(self.ball, move=decision1)
        self.paddle2.move(
            self.ball, move=self.mirror_decision(self.decision_dict[decision2])
        )
        self.ball.move(self.paddle1, self.paddle2, self.scorer)
        if self.display:
            self.draw()

        # manage rewards
        reward = 0
        done = False

        # Reward for keeping the ball in play
        # reward += 0.01

        # Reward for hitting the ball
        if self.ball.collision_left:
            reward += 1

        # if left_curr_score < self.scorer.left_score:
        #     reward += 1

        # Penalty for missing the ball
        if right_curr_score < self.scorer.right_score:
            reward -= 1

        # # Reward for scoring
        # if self.scorer.left_score >= 1 and self.scorer.left_hits >= 1:
        #     reward += 1

        # Exploration bonus (optional)
        # You may uncomment and adjust this part to add an exploration bonus
        # if np.random.rand() < 0.15:
        #     reward += 0.01  # Small positive reward for exploration

        if self.scorer.left_score >= 1 or self.scorer.right_score >= 1:
            done = True
            self.restart_pg()

        return self.state(), reward, done


def main():

    EPISODES = 20000
    DISPLAY = False
    TRAIN_EVERY = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Detected device: ", device)

    agent = PGAgent(device=device)
    baseline_agent = BaselineAgent()  # Instantiate the baseline agent
    env = PongGamePGTraining(display=DISPLAY, default_pong=False)

    best_reward = -np.inf

    current_state = env.init_pg_learning()
    r = []
    baselines = []

    for episode in tqdm(range(1, EPISODES + 1), unit="episodes"):

        current_state = env.restart_pg()

        done = False
        while not done:
            if DISPLAY:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit()

            action = agent.act(current_state)
            baseline_action = baseline_agent.act(
                current_state
            )  # Get action from the baseline agent
            new_state, reward, done = env.step_pg(action, agent)
            current_state = new_state
            r.append(reward)
            baselines.append(baseline_action)

        if sum(r) > best_reward:
            best_reward = sum(r)
            torch.save(
                agent.model.state_dict(),
                os.path.join(f"checkpoints/{date.today()}", "model.pth"),
            )
            print("*****************************************************")
            print(
                f"Model saved at checkpoints/{date.today()}/model.pth at episode {episode} with reward {best_reward:.2f} "
            )
            print("*****************************************************")

        print(
            f"Episode: {episode},  Reward: {sum(r)}",
        )

        # end of episode
        if episode % TRAIN_EVERY == 0:
            if sum(r) > 0:
                loss = agent.train(r, baselines)
            # print("loss: ", loss)
            r = []
            baselines = []


if __name__ == "__main__":
    main()
