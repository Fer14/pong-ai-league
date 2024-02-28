from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import numpy as np
import random
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


REPLAY_MEMORY_SIZE = 1_000
MIN_REPLAY_MEMORY_SIZE = 200
MINIBATCH_SIZE = 128
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 10
AGGREGATE_STATS_EVERY = 100
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001
MIN_REWARD = -200  # For model save

# set cuda to false
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print(tf.config.list_physical_devices("GPU"))


class PGAgent:

    def __init__(self):
        # Main model
        self.model = self.create_model()

        os.makedirs(f"checkpoints/{date.today()}", exist_ok=True)

    def create_model(self):
        # create a dense model with 6 input neurons, and 5 output neurons
        model = Sequential()
        model.add(Dense(6, input_shape=(6,), activation="relu"))
        model.add(Dense(5, activation="relu"))
        model.add(Dense(5, activation="softmax"))

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=Adam(learning_rate=0.001),
            metrics=["accuracy"],
        )

        return model

    def act(self, current_state):
        return np.argmax(self.model.predict(current_state, verbose=0))

    def train(self, x, y, r, batch_size):
        self.model.fit(
            x,
            y,
            batch_size=batch_size,
            verbose=0,
            shuffle=False,
            sample_weight=r,
        )

    def eval(self, x, y, r, batch_size):
        return self.model.evaluate(
            x, y, batch_size=batch_size, verbose=0  # sample_weight=r,
        )

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
        dr /= np.std(dr)
        return dr

    def print_stats(self, episode, reward_sums, losses, accuracy):
        if episode % AGGREGATE_STATS_EVERY == 0:
            avg_reward = np.mean(
                reward_sums[max(0, episode - AGGREGATE_STATS_EVERY) : episode]
            )
            avg_loss = np.mean(
                losses[max(0, episode - AGGREGATE_STATS_EVERY) : episode]
            )
            avg_acc = np.mean(
                accuracy[max(0, episode - AGGREGATE_STATS_EVERY) : episode]
            )
            print(
                f"Episode: {episode}, Average Loss: {avg_reward}, Average Reward: {avg_loss}, Average Accuracy: {avg_acc}"
            )


class PongGamePGTraining(PongGame):
    def __init__(self, default_pong=True, logo="../../imgs/big_logo_2.png"):
        super().__init__(default_pong=default_pong, logo=logo)
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

    def init_pg_learning(self):
        self.init_pong(
            PGPaddle(
                x=c.LEFT_PADDLE_INIT_POS[0],
                y=c.LEFT_PADDLE_INIT_POS[1],
                width=c.PADDLE_WIDTH,
                height=c.PADDLE_HEIGHT,
                speed=c.PADDLE_SPEED,
                field="left",
            ),
            PGPaddle(
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
                left_color="#ef3036",
                right_color="#ef3036",
                left_name="PG",
                right_name="PG",
                font="../../font/Gliker-Bold.ttf",
                logo="../../imgs/paddle.png",
                left_logo="../../imgs/team_logos/pg.png",
                right_logo="../../imgs/team_logos/pg.png",
            ),
        )
        return self.state()

    def step_pg(self, action):
        decision1 = self.decision_dict[action]
        self.paddle1.move(self.ball, move=decision1)
        self.paddle2.move(self.ball, move=self.mirror_decision(decision1))
        self.ball.move(self.paddle1, self.paddle2, self.scorer)
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

    EPISODES = 2_000

    agent = PGAgent()
    env = PongGamePGTraining(default_pong=False)

    losses = np.zeros(EPISODES)
    reward_sums = np.zeros(EPISODES)
    accuracy = np.zeros(EPISODES)

    X = []
    y = []
    r = []

    for episode in tqdm(range(1, EPISODES + 1), unit="episodes"):

        current_state = env.init_pg_learning()
        episode_reward = 0

        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

            action = agent.act(np.array([current_state]))
            new_state, reward, done = env.step_pg(action)
            current_state = new_state

            episode_reward += reward

            X.append(current_state)
            y.append(action)
            r.append(reward)

        # end of episode
        discounted_rewards = agent.disccount_n_standarise(r)
        agent.train(
            x=np.array(X),
            y=np.array(y),
            r=np.array(discounted_rewards),
            batch_size=len(X),
        )
        loss, acc = agent.eval(
            x=np.array(X),
            y=np.array(y),
            r=np.array(discounted_rewards),
            batch_size=len(X),
        )

        losses[episode] = loss
        accuracy[episode] = acc

        reward_sums[episode] = episode_reward
        agent.print_stats(episode, episode_reward, losses, accuracy)


if __name__ == "__main__":
    main()
