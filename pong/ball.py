import random
import pygame
import math
import constants as c


class Ball:
    def __init__(self, x, y, radius, ball_init_speeds=None, training_left=False):
        self.x = x
        self.y = y
        self.radius = radius
        self.init_speeds = (
            ball_init_speeds if ball_init_speeds is not None else c.BAL_INIT_SPEED
        )

        self.training_left = training_left

        if self.training_left:
            self.vx = -1
            self.vy = random.choice(self.init_speeds)
        else:
            self.vx = random.choice(self.init_speeds)
            self.vy = random.choice(self.init_speeds)
        self.last_x = x
        self.last_y = y

        # self.im = pygame.transform.scale(
        #     pygame.image.load(
        #         "../imgs/ball.png",
        #     ),
        #     (40, 22.52),
        # )

    def restart(self, paddle_left, paddle_right):
        self.x = c.WIDTH // 2
        self.y = c.HEIGHT // 2
        if self.training_left:
            self.vx = -1
            self.vy = random.choice(self.init_speeds)
        else:
            self.vx = random.choice(self.init_speeds)
            self.vy = random.choice(self.init_speeds)
        paddle_left.unblock()
        paddle_right.unblock()

    def draw(self, win):
        pygame.draw.circle(win, c.WHITE, (self.x, self.y), self.radius)
        # blit_x = self.x - self.im.get_width() / 2
        # blit_y = self.y - self.im.get_height() / 2
        # win.blit(self.im, (blit_x, blit_y))

    def move(self, paddle_left, paddle_right, scorer):
        self.last_x, self.last_y = self.x, self.y
        self.x += self.vx
        self.y += self.vy
        self.check_collision(paddle_left, paddle_right, scorer)
        self.check_blocks(paddle_left, paddle_right)

    def rect(self):
        return pygame.Rect(
            self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2
        )

    def check_collision(self, paddle_left, paddle_right, scorer):
        if self.y - self.radius <= c.SCORE_HEIGT or self.y + self.radius >= c.HEIGHT:
            self.vy *= -1
        elif self.x - self.radius <= 0:
            scorer.right_score += 1
            self.restart(paddle_left, paddle_right)
        elif self.x + self.radius >= c.WIDTH:
            scorer.left_score += 1
            self.restart(paddle_left, paddle_right)
        else:
            self.check_collision_left_field(paddle_left, scorer)
            self.check_collision_right_field(paddle_right, scorer)

    def check_blocks(self, paddle_left, paddle_right):
        if (
            self.x < c.LINE_X[0] + c.LINE_WIDTH or paddle_left.blocked
        ) and paddle_right.blocked:
            paddle_right.unblock()
        if (
            self.x > c.LINE_X[0] - c.LINE_WIDTH or paddle_right.blocked
        ) and paddle_left.blocked:
            paddle_left.unblock()

    def check_collision_left_field(self, paddle, scorer):
        pass

    def check_collision_right_field(self, paddle, scorer):
        pass

    def update(self):
        pass


class RealPhysicsBall(Ball):

    def check_collision_left_field(self, paddle, scorer):
        if self.rect().colliderect(paddle.rect()) and not paddle.blocked:
            # Calculate the angle of reflection based on the angle of incidence
            # Determine the direction of the bounce based on the relative positions of the ball and paddle
            if self.x < paddle.x + paddle.width / 2:
                # Bounce left
                angle_incidence = math.atan2(-self.vy, -self.vx)
                angle_reflection = math.pi - angle_incidence
            else:
                # Bounce right
                angle_incidence = math.atan2(-self.vy, self.vx)
                angle_reflection = math.pi - angle_incidence

            self.update(paddle.velocity(), angle_reflection)

            paddle.block()
            scorer.left_hits += 1

    def check_collision_right_field(self, paddle, scorer):
        if self.rect().colliderect(paddle.rect()) and not paddle.blocked:
            # Calculate the angle of reflection based on the angle of incidence
            # Determine the direction of the bounce based on the relative positions of the ball and paddle
            if self.x < paddle.x + paddle.width / 2:
                # Bounce left
                angle_incidence = math.atan2(-self.vy, self.vx)
                angle_reflection = math.pi - angle_incidence
            else:
                # Bounce right
                angle_incidence = math.atan2(-self.vy, -self.vx)
                angle_reflection = math.pi - angle_incidence

            self.update(paddle.velocity(), angle_reflection)

            paddle.block()
            scorer.right_hits += 1

    def update(self, paddle_velocity, angle_reflection):

        speed = math.sqrt(paddle_velocity**2 + self.vx**2 + self.vy**2)
        # Update velocity based on the angle of reflection and energy loss
        self.vx = math.cos(angle_reflection) * speed * 0.95
        self.vy = -math.sin(angle_reflection) * speed * 0.95


class FunnierPhysicsBall(Ball):

    def check_collision_left_field(self, paddle):

        if self.rect().colliderect(paddle.rect()) and not paddle.blocked:
            self.vx *= -1
            paddle_middle = paddle.y + paddle.height / 2
            d = paddle_middle - self.y
            red = (paddle.height / 2) / 3.5
            self.vy = -d / red

            paddle.block()

    def check_collision_right_field(self, paddle):
        if self.rect().colliderect(paddle.rect()) and not paddle.blocked:
            self.vx *= -1
            paddle_middle = paddle.y + paddle.height / 2
            d = paddle_middle - self.y
            red = (paddle.height / 2) / 3.5
            self.vy = -d / red
            paddle.block()
