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
from ppo_player import PPOPaddle
from memory_profiler import profile

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
AGGREGATE_STATS_EVERY = 5
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001
MIN_REWARD = -200  # For model save

# set cuda to false
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# print(tf.config.list_physical_devices("GPU"))


class PPOgent:

    def __init__(self):
        self.input_shappe = 6
        self.output_shape = 5
        # Main model
        self.model = self.create_model()
        self.epsilon_clip = 0.2

        os.makedirs(f"checkpoints/{date.today()}", exist_ok=True)

    def create_model(self):
        # create a dense model with 6 input neurons, and 5 output neurons
        model = Sequential()
        model.add(Dense(6, input_shape=(self.input_shappe,), activation="relu"))
        model.add(Dense(5, activation="relu"))
        model.add(Dense(self.output_shape, activation="softmax"))

        self.optimizer = Adam(learning_rate=0.001)
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=self.optimizer,
            metrics=["accuracy"],
        )

        return model

    def act(self, current_state):
        probs = self.model.predict(current_state, verbose=0)
        return np.argmax(probs), probs

    @tf.function
    def train_step(self, states, actions, advantages, old_probabilities):
        # https://keras.io/examples/rl/ppo_cartpole/
        with tf.GradientTape() as tape:
            probabilities = self.model(states, training=True)
            action_masks = tf.one_hot(actions, self.output_shape)
            action_probabilities = tf.reduce_sum(action_masks * probabilities, axis=1)
            old_action_probabilities = tf.reduce_sum(
                action_masks * old_probabilities, axis=1
            )
            ratios = tf.exp(
                tf.math.log(tf.cast(action_probabilities, dtype=tf.float32))
                - tf.math.log(tf.cast(old_action_probabilities, dtype=tf.float32))
            )
            clipped_ratios = tf.clip_by_value(
                ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip
            )
            surrogate1 = ratios * advantages
            surrogate2 = clipped_ratios * advantages
            loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train(self, states, actions, advantages, old_probabilities):
        states = np.array(states)
        actions = np.array(actions)
        advantages = np.array(advantages)
        old_probabilities = np.squeeze(np.array(old_probabilities))
        batch_size = len(states)

        states = tf.convert_to_tensor(states, dtype=tf.float32)

        loss = self.train_step(states, actions, advantages, old_probabilities)
        return loss

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
            start_idx = max(0, episode - AGGREGATE_STATS_EVERY)
            end_idx = episode + 1
            avg_reward = np.mean(reward_sums[start_idx:end_idx])
            avg_loss = np.mean(losses[start_idx:end_idx])
            avg_acc = np.mean(accuracy[start_idx:end_idx])
            print(
                f"Episode: {episode}, Average Loss: {avg_loss}, Average Reward: {avg_reward}, Average Accuracy: {avg_acc}"
            )


class PongGamePPOTraining(PongGame):
    def __init__(
        self, default_pong=True, logo="../../imgs/big_logo_2.png", display=True
    ):
        super().__init__(display=display, default_pong=default_pong, logo=logo)
        self.decision_dict = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}

    def restart_pg(self):
        self.restart()
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
            PPOPaddle(
                x=c.LEFT_PADDLE_INIT_POS[0],
                y=c.LEFT_PADDLE_INIT_POS[1],
                width=c.PADDLE_WIDTH,
                height=c.PADDLE_HEIGHT,
                speed=c.PADDLE_SPEED,
                field="left",
            ),
            PPOPaddle(
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
                display=self.display,
            ),
        )
        return self.state()

    def step_pg(self, action, draw=False):
        decision1 = self.decision_dict[action]
        self.paddle1.move(self.ball, move=decision1)
        self.paddle2.move(self.ball, move=self.mirror_decision(decision1))
        self.ball.move(self.paddle1, self.paddle2, self.scorer)
        if draw:
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
    DISPLAY = False

    agent = PPOgent()
    env = PongGamePPOTraining(display=DISPLAY, default_pong=False)

    losses = np.zeros(EPISODES)
    reward_sums = np.zeros(EPISODES)
    accuracy = np.zeros(EPISODES)

    best_loss = 100

    current_state = env.init_pg_learning()

    for episode in tqdm(range(1, EPISODES + 1), unit="episodes"):

        current_state = env.restart_pg()
        episode_reward = 0
        X = []
        y = []
        r = []
        old_probabilities = []

        done = False
        while not done:
            if DISPLAY:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit()

            action, old_probabily = agent.act(np.array([current_state]))
            old_probabilities.append(old_probabily)
            new_state, reward, done = env.step_pg(action)
            current_state = new_state

            episode_reward += reward

            X.append(current_state)
            y.append(action)
            r.append(reward)

        # end of episode
        discounted_rewards = agent.disccount_n_standarise(r)
        agent.train(X, y, discounted_rewards, old_probabilities)
        loss, acc = agent.eval(
            x=np.array(X),
            y=np.array(y),
            r=np.array(discounted_rewards),
            batch_size=len(X),
        )

        losses[episode] = loss
        accuracy[episode] = acc

        if loss < best_loss:
            best_loss = loss
            agent.model.save(f"checkpoints/{date.today()}/pg_model_{episode}_{loss}.h5")
            print(
                f"Model saved at checkpoints/{date.today()}/pg_model_{episode}_{loss}.h5"
            )

        reward_sums[episode] = episode_reward
        agent.print_stats(episode, reward_sums, losses, accuracy)


if __name__ == "__main__":
    main()
