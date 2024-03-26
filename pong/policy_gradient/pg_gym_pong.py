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

sys.path.append("../")

import gym


DISCOUNT = 0.99


class PGAgent:

    def __init__(self, device):
        # Main model
        self.device = device
        self.model = self.create_model().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0006)  # Adam

        os.makedirs(f"checkpoints/{date.today()}", exist_ok=True)
        self.onpolicy_reset()

    def onpolicy_reset(self):
        self.log_probs = []

    def create_model(self):
        ## build a cnn model that takes a 6400 as input  and returns a probabilty
        model = nn.Sequential(
            nn.Linear(6400, 300),
            nn.ReLU(),
            nn.Linear(300, 2),
            nn.Softmax(dim=-1),
        )

        return model

    def forward(self, x):
        return self.model(x)

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action_probs = self.forward(state)
        prob_dist = torch.distributions.Categorical(action_probs)
        action = prob_dist.sample().to(self.device)
        self.log_probs.append(prob_dist.log_prob(action))

        return action.item()

    def train(self, r, baselines):
        self.optimizer.zero_grad()

        log_probs = torch.stack(self.log_probs).to(self.device)
        r = self.disccount_n_standarise(r)
        rewards = torch.tensor(r).view(-1, 1).to(self.device)

        # Calculate advantages using baselines
        # advantages = rewards - torch.tensor(baselines, dtype=torch.float32).view(
        #     -1, 1
        # ).to(self.device)
        advantages = rewards

        entropy_coef = 0.01

        loss = torch.mean(-entropy_coef * (log_probs * advantages))

        loss.backward()
        self.optimizer.step()
        self.onpolicy_reset()

        return loss.item()

    def update_policy(self, rewards):
        eps = 1e-8
        R = 0
        policy_loss = []
        returns = []
        for r in rewards[::-1]:
            R = r + DISCOUNT * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append((-log_prob * R).unsqueeze(0))
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        self.onpolicy_reset()

    def discount_reward(self, rewards):
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = rewards[t] + running_add * DISCOUNT
            discounted_rewards[t] = running_add
        return discounted_rewards

    def disccount_n_standarise(self, r):
        dr = self.discount_reward(r)
        dr -= np.mean(dr)
        dr /= np.std(dr) if np.std(dr) > 0 else 1
        return dr


def preprocess(I):
    """prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector"""
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float64).ravel()


def main():

    EPISODES = 20000
    TRAIN_EVERY = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Detected device: ", device)

    agent = PGAgent(device=device)
    environment = gym.make("Pong-v0")
    best_reward = -np.inf

    r = []
    # baselines = []

    for episode in tqdm(range(1, EPISODES + 1), unit="episodes"):

        current_state, info = environment.reset()
        current_state = preprocess(current_state)

        done = False
        while not done:

            action = agent.act(current_state)
            # baseline_action = baseline_agent.act(current_state)
            new_state, reward, terminated, truncated, info = environment.step(action)
            done = terminated or truncated
            current_state = preprocess(new_state)
            r.append(reward)
            # baselines.append(baseline_action)

        if sum(r) > best_reward:
            best_reward = sum(r)
            # torch.save(
            #     agent.model.state_dict(),
            #     os.path.join(f"checkpoints/{date.today()}", "model.pth"),
            # )
            print("*****************************************************")
            print(
                f"Model saved at checkpoints/{date.today()}/model.pth at episode {episode} with reward {best_reward:.2f} "
            )
            print("*****************************************************")

        print(
            f"Episode: {episode},  Reward: {sum(r)}",
        )

        # end of episode
        if episode % TRAIN_EVERY == 0 and sum(r) != 0:
            # loss = agent.train(r)
            agent.update_policy(r)
            # print("loss: ", loss)
            r = []
            # baselines = []


if __name__ == "__main__":
    main()
