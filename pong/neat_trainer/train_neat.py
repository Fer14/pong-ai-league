import os
import neat
import pickle
import sys
from neat_player import NeatPaddle
import pygame
import random
from datetime import date

sys.path.append("../")

from pong import PongGame
from ball import Ball, RealPhysicsBall, FunnierPhysicsBall
from scorer import Scorer
import constants as c

PUNISH_EVERY_X = 30
MAX_HITS = 25


class PongGameNeatTraining(PongGame):

    def __init__(
        self, default_pong=True, logo="../../imgs/big_logo_2.png", display=False
    ):
        super().__init__(default_pong=default_pong, logo=logo, display=display)

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

    def sim_neat(self, genome1, genome2, config):
        self.init_pong(
            NeatPaddle(
                x=c.LEFT_PADDLE_INIT_POS[0],
                y=c.LEFT_PADDLE_INIT_POS[1],
                width=c.PADDLE_WIDTH,
                height=c.PADDLE_HEIGHT,
                speed=c.PADDLE_SPEED,
                field="left",
                train=True,
                training_speed=c.PADDLE_SPEED,
            ),
            NeatPaddle(
                x=c.RIGHT_PADDLE_INIT_POS[0],
                y=c.RIGHT_PADDLE_INIT_POS[1],
                width=c.PADDLE_WIDTH,
                height=c.PADDLE_HEIGHT,
                speed=c.PADDLE_SPEED,
                field="right",
                train=True,
                training_speed=c.PADDLE_SPEED,
            ),
            RealPhysicsBall(
                c.BALL_INIT_POS[0],
                c.BALL_INIT_POS[1],
                c.BALL_RADIUS,
                ball_init_speeds=[-2, 2],
            ),
            Scorer(
                width=c.WIDTH,
                height=c.HEIGHT,
                left_color="#1a5e9d",
                right_color="#1a5e9d",
                left_name="NEAT",
                right_name="NEAT",
                font="../../font/Gliker-Bold.ttf",
                logo="../../imgs/paddle.png",
                left_logo="../../imgs/team_logos/neat.png",
                right_logo="../../imgs/team_logos/neat.png",
                display=self.display,
            ),
        )

        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

        decision_dict = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}

        genome1_x = []
        genome2_x = []
        running = True
        while running:
            if self.display:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit()

            inputs1 = (
                self.paddle1.x,
                self.paddle1.y,
                self.ball.x,
                self.ball.y,
                self.ball.last_x,
                self.ball.last_y,
            )

            inputs2 = (
                self.paddle2.x,
                self.paddle2.y,
                self.ball.x,
                self.ball.y,
                self.ball.last_x,
                self.ball.last_y,
            )

            output1 = net1.activate(inputs1)
            output2 = net2.activate(self.mirror_inputs(inputs2))

            decision1 = decision_dict[output1.index(max(output1))]
            decision2 = self.mirror_decision(decision_dict[output2.index(max(output2))])

            # if (
            #     (self.paddle1.x == 0 and decision1 == "LEFT")
            #     or (decision1 == "RIGHT" and self.paddle1.x == c.LINE_X[0] - 10)
            #     or (decision1 == "UP" and self.paddle1.y == 0)
            #     or (
            #         decision1 == "DOWN"
            #         and self.paddle1.y == c.HEIGHT + c.SCORE_HEIGT - c.PADDLE_HEIGHT
            #     )
            # ):
            #     genome1.fitness -= 0.2
            # if (
            #     (self.paddle2.x == c.WIDTH - c.LINE_X[0] and decision2 == "RIGHT")
            #     or (decision2 == "LEFT" and self.paddle2.x == c.LINE_X[0] + 10)
            #     or (decision2 == "UP" and self.paddle2.y == 0)
            #     or (
            #         decision2 == "DOWN"
            #         and self.paddle2.y == c.HEIGHT + c.SCORE_HEIGT - c.PADDLE_HEIGHT
            #     )
            # ):
            #     genome2.fitness -= 0.2

            self.paddle1.move(self.ball, move=decision1)
            self.paddle2.move(self.ball, move=decision2)
            self.ball.move(self.paddle1, self.paddle2, self.scorer)

            genome1_x.append(self.paddle1.x)
            genome2_x.append(self.paddle2.x)

            # if len(genome1_x) >= PUNISH_EVERY_X and genome1_x[-PUNISH_EVERY_X] in range(
            #     self.paddle1.x - 5, self.paddle1.x + 5
            # ):
            #     genome1.fitness -= 0.05

            # if len(genome2_x) >= PUNISH_EVERY_X and genome1_x[-PUNISH_EVERY_X] in range(
            #     self.paddle2.x - 5, self.paddle2.x + 5
            # ):
            #     genome2.fitness -= 0.05

            if self.display:
                self.draw()

            if (
                self.scorer.left_score == 1
                or self.scorer.right_score == 1
                or self.scorer.left_hits >= MAX_HITS
                or self.scorer.right_hits >= MAX_HITS
            ):

                if self.scorer.left_score == 1:
                    genome2.fitness -= 10

                    if self.scorer.left_hits == 0:
                        genome1.fitness += genome1.fitness / genome1.games
                    else:
                        genome1.fitness += self.scorer.left_hits

                if self.scorer.right_score == 1:
                    genome1.fitness -= 10

                    if self.scorer.right_hits == 0:
                        genome2.fitness += genome2.fitness / genome2.games
                    else:
                        genome2.fitness += self.scorer.right_hits

                break


def eval_genomes(genomes, config, shuffle=False):
    size = len(genomes)
    display = False
    game = PongGameNeatTraining(default_pong=False, display=display)

    for genomeid, genome in genomes:
        genome.fitness = 0
        genome.games = 0

    for i, (genome_id1, genome1) in enumerate(genomes):
        if i == size - 1:
            break
        for genome_id2, genome2 in genomes[i + 1 :]:
            # Make sure the genomes are different
            if genome_id1 == genome_id2:
                continue
            genome1.games += 1
            genome2.games += 1
            game.sim_neat(genome1, genome2, config)

        genome1.fitness /= size


def run_neat(config):
    # p = neat.Checkpointer.restore_checkpoint(
    #     f"./checkpoints/{date.today()}/neat-checkpoint-99"
    # )
    # p.config = config
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(
    #     neat.Checkpointer(
    #         100, filename_prefix=f"./checkpoints/{date.today()}/neat-checkpoint-"
    #     )
    # )

    winner = p.run(eval_genomes, 200)
    # with open(f"best_found_{date.today()}.pickle", "wb") as f:
    with open("best_both_player.pickle", "wb") as f:
        pickle.dump(winner, f)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat_config.txt")
    os.makedirs(f"checkpoints/{date.today()}", exist_ok=True)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    run_neat(config)
