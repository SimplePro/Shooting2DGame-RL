import pygame
import numpy as np
import random
import math
import torch


class ShootingGame:
    def __init__(self):
        pygame.init()

        self.WIDTH = 64
        self.HEIGHT = 64
        self.SCALE = 15

        self.character_size = 3

        self.player_color = (41, 249, 114)
        self.ai_color = (200, 40, 40)
        self.background_color = (0, 0, 0)

        self.bullet_size = 1
        self.player_bullet_color = (255, 255, 255)
        self.ai_bullet_color = (255, 0, 0)

        self.bullet_cooldown_time = 30  # Cooldown time in ticks

        self.player_initial_range_x = (self.character_size, self.WIDTH - self.character_size)
        self.player_initial_range_y = (self.character_size, self.HEIGHT - self.character_size)
        self.ai_initial_range_x = (self.character_size, self.WIDTH - self.character_size)
        self.ai_initial_range_y = (self.character_size, self.HEIGHT - self.character_size)

        self.clock = pygame.time.Clock()
        self.tick_rate = 20
        self.screen = pygame.display.set_mode(
            (self.WIDTH * self.SCALE, self.HEIGHT * self.SCALE)
        )
        pygame.display.set_caption("Shooting Game")

    def reset(self):

        self.observation = None
        self.done = False

        self.player_bullets = []
        self.ai_bullets = []

        self.player_bullet_cooldown = 0
        self.ai_bullet_cooldown = 0
        self.player_eye_color = (50, 50, 250)
        self.ai_eye_color = (50, 50, 250)

        self.player_position = (
            random.randint(
                self.player_initial_range_x[0], self.player_initial_range_x[1]
            ),
            random.randint(
                self.player_initial_range_y[0], self.player_initial_range_y[1]
            ),
        )
        self.ai_position = (
            random.randint(self.ai_initial_range_x[0], self.ai_initial_range_x[1]),
            random.randint(self.ai_initial_range_y[0], self.ai_initial_range_y[1]),
        )
        # self.player_position = tuple((i // 3) * 3 for i in self.player_position)
        # self.ai_position = tuple((i // 3) * 3 for i in self.ai_position)

        self.player_direction = 0  # Initial direction for player
        self.ai_direction = 0  # Initial direction for ai

        self.display()

        return self.observation

    def display(self):
        surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        surface.fill(self.background_color)

        player_x = self.player_position[0]
        player_y = self.player_position[1]
        ai_x = self.ai_position[0]
        ai_y = self.ai_position[1]

        # Draw bullets
        for bullet_x, bullet_y, _ in self.player_bullets:
            pygame.draw.rect(
                surface,
                self.player_bullet_color,
                (bullet_x, bullet_y, self.bullet_size, self.bullet_size),
            )
        for bullet_x, bullet_y, _ in self.ai_bullets:
            pygame.draw.rect(
                surface,
                self.ai_bullet_color,
                (bullet_x, bullet_y, self.bullet_size, self.bullet_size),
            )

        # Draw objects
        pygame.draw.rect(
            surface,
            self.player_color,
            (player_x, player_y, self.character_size, self.character_size),
        )
        pygame.draw.rect(
            surface, self.ai_color, (ai_x, ai_y, self.character_size, self.character_size)
        )

        # Draw direction indicator pixel
        player_direction_pixel = self.get_direction_pixel(self.player_direction)
        ai_direction_pixel = self.get_direction_pixel(self.ai_direction)
        surface.set_at(
            (
                player_x + player_direction_pixel[0],
                player_y + player_direction_pixel[1],
            ),
            self.player_eye_color,
        )
        surface.set_at(
            (ai_x + ai_direction_pixel[0], ai_y + ai_direction_pixel[1]),
            self.ai_eye_color,
        )

        self.observation = pygame.surfarray.array3d(surface)

        scaled_surface = pygame.transform.scale(
            surface, (self.WIDTH * self.SCALE, self.HEIGHT * self.SCALE)
        )

        self.screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()

    def move_p(self):
        player_speed = 1  # Modify as needed
        player_dx = (
            self.pressed_keys[pygame.K_d] - self.pressed_keys[pygame.K_a]
        ) * player_speed
        player_dy = (
            self.pressed_keys[pygame.K_s] - self.pressed_keys[pygame.K_w]
        ) * player_speed

        if player_dx > 0:
            self.player_direction = 1  # Right
        elif player_dx < 0:
            self.player_direction = 3  # Left
        elif player_dy > 0:
            self.player_direction = 2  # Down
        elif player_dy < 0:
            self.player_direction = 0  # Up

        return (
            self.player_position[0] + player_dx,
            self.player_position[1] + player_dy,
        )

    def move_ai(self, action):
        ai_speed = 1  # Modify as needed
        ai_dx = (action[0] - action[1]) * ai_speed
        ai_dy = (action[2] - action[3]) * ai_speed

        if ai_dx > 0:
            self.ai_direction = 1  # Right
                
        elif ai_dx < 0:
            self.ai_direction = 3  # Left

        elif ai_dy > 0:
            self.ai_direction = 2  # Down

        elif ai_dy < 0:
            self.ai_direction = 0  # Up

        return (self.ai_position[0] + ai_dx, self.ai_position[1] + ai_dy)
    

    def move_objects(self, action):
        new_player_position = self.move_p()
        if self.within_boundary(new_player_position, self.character_size):
            self.player_position = new_player_position

        new_ai_position = self.move_ai(action)
        if self.within_boundary(new_ai_position, self.character_size):
            self.ai_position = new_ai_position
        #     self.ai_at_wall = 0
        # else:
        #     self.ai_at_wall = 1

    def within_boundary(self, position, size):
        x, y = position
        return 0 <= x <= self.WIDTH - size and 0 <= y <= self.HEIGHT - size

    def get_direction_pixel(self, direction):
        if direction == 0:  # Up
            return (1, 0)
        elif direction == 1:  # Right
            return (2, 1)
        elif direction == 2:  # Down
            return (1, 2)
        elif direction == 3:  # Left
            return (0, 1)

    def player_shoot_bullet(self):
        if self.player_bullet_cooldown == 0:
            direction = self.get_direction_pixel(self.player_direction)
            bullet_x = (
                self.player_position[0]
                + self.character_size // 2
                - self.bullet_size // 2
                + (direction[0] - 1) * 1
            )
            bullet_y = (
                self.player_position[1]
                + self.character_size // 2
                - self.bullet_size // 2
                + (direction[1] - 1) * 1
            )
            bullet_direction = self.player_direction

            self.player_bullets.append((bullet_x, bullet_y, bullet_direction))
            self.player_bullet_cooldown = self.bullet_cooldown_time

    def ai_shoot_bullet(self):
        if self.ai_bullet_cooldown == 0:
            direction = self.get_direction_pixel(self.ai_direction)
            bullet_x = (
                self.ai_position[0]
                + self.character_size // 2
                - self.bullet_size // 2
                + (direction[0] - 1) * 1
            )
            bullet_y = (
                self.ai_position[1]
                + self.character_size // 2
                - self.bullet_size // 2
                + (direction[1] - 1) * 1
            )
            bullet_direction = self.ai_direction

            self.ai_bullets.append((bullet_x, bullet_y, bullet_direction))
            self.ai_bullet_cooldown = self.bullet_cooldown_time

            dis_x = (-abs(self.ai_position[0] - self.player_position[0]) / 64) + 1
            dis_y = (-abs(self.ai_position[1] - self.player_position[1]) / 64) + 1

            dis_matrix = np.array([dis_x, dis_y, dis_x, dis_y])
            direction_matrix = np.array([0, 0, 0, 0])
            direction_matrix[self.ai_direction] = 1

            ai_x, ai_y = self.ai_position
            player_x, player_y = self.player_position

            is_left = int(player_x <= ai_x)
            is_right = not is_left
            is_up = int(player_y <= ai_y)
            is_down = not is_up

            direction_label_matrix = np.array([is_up, is_right, is_down, is_left])

            self.reward += (sum(dis_matrix * direction_matrix * direction_label_matrix))**2
            print(self.reward)

    def check_bullet_collisions(self):

        # Check player bullet collisions
        player_hit_indices = []
        for i, bullet in enumerate(self.player_bullets):
            bullet_rect = pygame.Rect(
                bullet[0], bullet[1], self.bullet_size, self.bullet_size
            )
            if bullet_rect.colliderect(
                pygame.Rect(*self.ai_position, self.character_size, self.character_size)
            ):
                print("player_hit")
                player_hit_indices.append(i)
                self.reward -= 20
                self.done = 1

        # Remove collided player bullets
        for index in player_hit_indices:
            del self.player_bullets[index]

        # Check AI bullet collisions
        ai_hit_indices = []
        for i, bullet in enumerate(self.ai_bullets):
            bullet_rect = pygame.Rect(
                bullet[0], bullet[1], self.bullet_size, self.bullet_size
            )
            if bullet_rect.colliderect(
                pygame.Rect(*self.player_position, self.character_size, self.character_size)
            ):
                print('ai_hit')
                ai_hit_indices.append(i)
                self.reward += 50
                self.done = 1

        # Remove collided AI bullets
        for index in ai_hit_indices:
            del self.ai_bullets[index]

        return self.reward

    def move_bullets(self):
        bullet_speed = 3  # Modify as needed

        # Move player bullets
        new_player_bullets = []
        for i in range(len(self.player_bullets)):
            bullet_x, bullet_y, bullet_direction = self.player_bullets[i]
            if bullet_direction == 0:  # Up
                bullet_y -= bullet_speed
            elif bullet_direction == 1:  # Right
                bullet_x += bullet_speed
            elif bullet_direction == 2:  # Down
                bullet_y += bullet_speed
            elif bullet_direction == 3:  # Left
                bullet_x -= bullet_speed

            # Check if the bullet is still within the boundary

            if self.within_boundary((bullet_x, bullet_y), self.bullet_size):
                new_player_bullets.append((bullet_x, bullet_y, bullet_direction))
        self.player_bullets = new_player_bullets

        # Move ai bullets
        new_ai_bullets = []
        for i in range(len(self.ai_bullets)):
            bullet_x, bullet_y, bullet_direction = self.ai_bullets[i]
            if bullet_direction == 0:  # Up
                bullet_y -= bullet_speed
            elif bullet_direction == 1:  # Right
                bullet_x += bullet_speed
            elif bullet_direction == 2:  # Down
                bullet_y += bullet_speed
            elif bullet_direction == 3:  # Left
                bullet_x -= bullet_speed

            # Check if the bullet is still within the boundary

            if self.within_boundary((bullet_x, bullet_y), self.bullet_size):
                new_ai_bullets.append((bullet_x, bullet_y, bullet_direction))
        self.ai_bullets = new_ai_bullets

    def step(self, ai_input, dis_reward_alpha):
        self.reward = 0

        action = np.zeros(5, int)
        action[ai_input] = 1

        self.clock.tick(self.tick_rate)
        
        # Decrement bullet cooldowns
        if self.player_bullet_cooldown > 0:
            self.player_bullet_cooldown -= 1
            self.player_eye_color = (
                50,
                50,
                250
                - 200
                * (self.player_bullet_cooldown / self.bullet_cooldown_time),
            )
        if self.ai_bullet_cooldown > 0:
            self.ai_bullet_cooldown -= 1
            self.ai_eye_color = (
                50,
                50,
                250 - 200 * (self.ai_bullet_cooldown / self.bullet_cooldown_time),
            )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.player_shoot_bullet()

        self.pressed_keys = pygame.key.get_pressed()

        # if action1[4] == 1: 
        #     self.player_shoot_bullet()

        if action[4] == 1:
            self.ai_shoot_bullet()

        self.move_objects(action)

        self.move_bullets() 
        
        self.check_bullet_collisions()

        self.display()

        dis_x = abs(self.player_position[0] - self.ai_position[0]) / 64
        dis_y = abs(self.player_position[1] - self.ai_position[1]) / 64

        self.reward -= (dis_x * dis_y) * dis_reward_alpha

        return self.observation, self.reward, self.done, None


if __name__ == '__main__':

    shooting_game = ShootingGame()
    while True:
        shooting_game.reset()

        while not shooting_game.done:
            shooting_game.step(ai_input=4)

