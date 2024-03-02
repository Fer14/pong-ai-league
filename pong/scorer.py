import pygame
import constants as c


def hex_to_rgb(hex_color):
    # Remove "#" if present
    hex_color = hex_color.lstrip("#")
    # Convert HEX to RGB
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def get_contrast_color(hex_color):
    # Convert HEX to RGB
    rgb_color = hex_to_rgb(hex_color)

    # Convert RGB to linear luminance (perceived brightness)
    r, g, b = [x / 255.0 for x in rgb_color]
    luminance = (0.2126 * r) + (0.7152 * g) + (0.0722 * b)

    # Choose text color based on luminance
    return c.BLACK if luminance > 0.5 else c.WHITE


class Scorer:
    def __init__(
        self,
        width,
        height,
        left_color,
        right_color,
        left_name,
        right_name,
        right_logo,
        left_logo,
        font="../font/Gliker-Bold.ttf",
        logo="../imgs/paddle.png",
        display=True,
    ):
        self.width = width
        self.height = height
        self.left_score = 0
        self.right_score = 0
        self.display = display
        if self.display:
            self.left_color = left_color
            self.right_color = right_color
            self.left_name = left_name
            self.right_name = right_name
            self.font = pygame.font.Font(font, 24)
            self.logo = pygame.transform.scale(
                pygame.image.load(logo),
                (89, 50),
            )
            self.left_logo = pygame.transform.scale(
                pygame.image.load(left_logo),
                (60, 33),
            )
            self.right_logo = pygame.transform.scale(
                pygame.image.load(right_logo),
                (60, 33),
            )
            self.left_text_color = get_contrast_color(left_color)
            self.right_text_color = get_contrast_color(right_color)
        self.left_hits = 0
        self.right_hits = 0

    def draw(self, win):
        pygame.draw.rect(win, self.right_color, (400, 0, 400, 50))

        pygame.draw.rect(win, self.left_color, (0, 0, 400, 50))
        pygame.draw.rect(win, self.left_color, (400, 25, 15, 25))
        pygame.draw.rect(win, self.left_color, (415, 40, 10, 10))

        left_score = self.font.render(str(self.left_score), True, self.left_text_color)
        right_score = self.font.render(
            str(self.right_score), True, self.right_text_color
        )

        win.blit(left_score, (340, 5))
        win.blit(right_score, (435, 5))

        left_name = self.font.render(self.left_name, True, self.left_text_color)
        right_name = self.font.render(self.right_name, True, self.right_text_color)

        win.blit(left_name, (70, 5))
        win.blit(right_name, (470, 5))

        win.blit(self.logo, (360, 0))

        win.blit(self.left_logo, (5, 5))
        win.blit(self.right_logo, (730, 5))
