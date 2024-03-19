import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np
import random
import sys
from tqdm import tqdm
from dqn_player import DQNPaddle
import pygame
from datetime import date
import os

sys.path.append("../")

from pong import PongGame
from ball import RealPhysicsBall
from scorer import Scorer
import constants as c

REPLAY_MEMORY_SIZE = 500
MIN_REPLAY_MEMORY_SIZE = 100
MINIBATCH_SIZE = 32
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 10
AGGREGATE_STATS_EVERY = 100
EPSILON_DECAY = 0.9975
MIN_EPSILON = 0.001
MIN_REWARD = -200  # For model save

# set cuda to false
import torch


class DQNAgent:

    def __init__(self, epsilon):
        # Main model
        self.model = self.create_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        # Target model
        self.target_model = self.create_model(trainable=False)
        self.target_model.load_state_dict(self.model.state_dict())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

        self.epsilon = epsilon
        os.makedirs(f"checkpoints/{date.today()}", exist_ok=True)

    def create_model(self, trainable=True):
        # create a dense model with 6 input neurons, and 5 output neurons
        model = nn.Sequential(
            nn.Linear(6, 6),
            nn.ReLU(),
            nn.Linear(6, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
        )

        if not trainable:
            for param in model.parameters():
                param.requires_grad = False

        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        state = torch.FloatTensor([state])
        with torch.no_grad():
            return self.model(state).numpy()[0]

    def train(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = torch.FloatTensor([transition[0] for transition in minibatch])
        new_current_states = torch.FloatTensor(
            [transition[3] for transition in minibatch]
        )

        current_qs_list = self.model(current_states)
        future_qs_list = self.target_model(new_current_states)

        X = []
        y = []

        for index, (
            current_state,
            action,
            reward,
            new_current_state,
            done,
        ) in enumerate(minibatch):
            if not done:
                new_q = reward + DISCOUNT * torch.max(future_qs_list[index])
            else:
                new_q = reward

            current_qs = current_qs_list[index].tolist()
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)

        self.optimizer.zero_grad()
        loss = F.mse_loss(self.model(X), y)
        loss.backward()
        self.optimizer.step()

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0

    def print_stats(self, ep_rewards, episode):
        if not episode % AGGREGATE_STATS_EVERY:
            average_reward = np.mean(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            print(
                f"Episode: {episode}, average reward: {average_reward}, min: {min_reward}, max: {max_reward}"
            )
            if average_reward >= MIN_REWARD:
                print("Saving model...")
                torch.save(
                    self.model.state_dict(),
                    os.path.join(f"checkpoints/{date.today()}", "model.pth"),
                )

    def decay_epsilon(self):
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY
            # print(f"epsilon: {self.epsilon} of size {sys.getsizeof(self.epsilon)} ")
            self.epsilon = max(MIN_EPSILON, self.epsilon)

    def act_epsilon_greedy(self, current_state):
        if np.random.random() > self.epsilon:
            action = np.argmax(self.get_qs(current_state))
        else:
            action = np.random.choice(np.arange(0, 5))

        return action


class PongGameQTraining(PongGame):
    def __init__(
        self, default_pong=True, logo="../../imgs/big_logo_2.png", display=True
    ):
        super().__init__(default_pong=default_pong, logo=logo, display=display)
        self.decision_dict = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}

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

    def restart_(self):
        self.restart()
        return self.state()

    def init_q_learning(self):
        self.init_pong(
            DQNPaddle(
                x=c.LEFT_PADDLE_INIT_POS[0],
                y=c.LEFT_PADDLE_INIT_POS[1],
                width=c.PADDLE_WIDTH,
                height=c.PADDLE_HEIGHT,
                speed=c.PADDLE_SPEED,
                field="left",
            ),
            DQNPaddle(
                x=c.RIGHT_PADDLE_INIT_POS[0],
                y=c.RIGHT_PADDLE_INIT_POS[1],
                width=c.PADDLE_WIDTH,
                height=c.PADDLE_HEIGHT,
                speed=c.PADDLE_SPEED,
                field="right",
            ),
            RealPhysicsBall(
                c.BALL_INIT_POS[0],
                c.BALL_INIT_POS[1],
                c.BALL_RADIUS,
                ball_init_speeds=[-1, 1],
            ),
            Scorer(
                width=c.WIDTH,
                height=c.HEIGHT,
                left_color="#ffd25a",
                right_color="#ffd25a",
                left_name="DQN",
                right_name="DQN",
                font="../../font/Gliker-Bold.ttf",
                logo="../../imgs/paddle.png",
                left_logo="../../imgs/team_logos/dqn.png",
                right_logo="../../imgs/team_logos/dqn.png",
                display=self.display,
            ),
        )
        return self.state()

    def step_q(self, action):
        decision1 = self.decision_dict[action]
        self.paddle1.move(self.ball, move=decision1)
        self.paddle2.move(self.ball, move=self.mirror_decision(decision1))
        self.ball.move(self.paddle1, self.paddle2, self.scorer)
        if self.display:
            self.draw()

        # manage rewards
        reward = 0
        done = False

        if (
            (self.paddle1.x == 0 and decision1 == "LEFT")
            or (decision1 == "RIGHT" and self.paddle1.x == c.LINE_X[0] - 10)
            or (decision1 == "UP" and self.paddle1.y == 0)
            or (
                decision1 == "DOWN"
                and self.paddle1.y == c.HEIGHT + c.SCORE_HEIGT - c.PADDLE_HEIGHT
            )
        ):
            reward -= 1

        if decision1 == "STAY":
            reward -= 0.2

        if (
            (self.scorer.left_score == 1 and self.scorer.left_hits >= 1)
            or self.scorer.right_score == 1
            or self.scorer.left_hits >= 50
            or self.scorer.right_hits >= 50
        ):
            reward += self.scorer.left_hits
            reward -= self.scorer.right_hits
            done = True
        return self.state(), reward, done


def main():

    # scalene_profiler.start()
    # EPISODES = 20_000
    EPISODES = 2000

    DISPLAY = False

    agent = DQNAgent(epsilon=1)
    env = PongGameQTraining(default_pong=False, display=DISPLAY)

    ep_rewards = []

    current_state = env.init_q_learning()

    for episode in tqdm(range(1, EPISODES + 1), unit="episodes"):

        episode_reward = 0
        step = 1
        current_state = env.restart_()

        done = False

        while not done:
            if DISPLAY:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit()

            action = agent.act_epsilon_greedy(current_state)
            new_state, reward, done = env.step_q(action)
            episode_reward += reward

            current_state = new_state
            step += 1

            agent.update_replay_memory((current_state, action, reward, new_state, done))

        agent.train(done)

        ep_rewards = (
            ep_rewards[1:] if len(ep_rewards) >= REPLAY_MEMORY_SIZE else ep_rewards
        )

        ep_rewards.append(episode_reward)
        agent.print_stats(ep_rewards=ep_rewards, episode=episode)

        agent.decay_epsilon()

        if DISPLAY:
            pygame.display.quit()
            pygame.quit()

    # scalene_profiler.stop()


if __name__ == "__main__":
    main()
