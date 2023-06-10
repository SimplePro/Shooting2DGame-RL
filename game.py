import pygame
import numpy as np
import random
import math


class ShootingGame:
    def __init__(self):
        pygame.init()

        self.WIDTH = 64
        self.HEIGHT = 64
        self.SCALE = 15

        self.ai_size = 3

        self.ai1_color = (41, 249, 114)
        self.ai2_color = (200, 40, 40)
        self.background_color = (0, 0, 0)

        self.bullet_size = 1
        self.ai1_bullet_color = (255, 255, 255)
        self.ai2_bullet_color = (255, 0, 0)

        self.bullet_cooldown_time = 100  # Cooldown time in ticks

        self.ai1_initial_range_x = (self.ai_size, self.WIDTH - self.ai_size)
        self.ai1_initial_range_y = (self.ai_size, self.HEIGHT - self.ai_size)
        self.ai2_initial_range_x = (self.ai_size, self.WIDTH - self.ai_size)
        self.ai2_initial_range_y = (self.ai_size, self.HEIGHT - self.ai_size)

        self.clock = pygame.time.Clock()
        self.tick_rate = 20
        self.screen = pygame.display.set_mode(
            (self.WIDTH * self.SCALE, self.HEIGHT * self.SCALE)
        )
        pygame.display.set_caption("Shooting Game")

    def reset(self):

        self.observation = None
        self.done = False

        self.ai1_bullets = []
        self.ai2_bullets = []

        self.ai1_bullet_cooldown = 0
        self.ai2_bullet_cooldown = 0
        self.ai1_eye_color = (50, 50, 250)
        self.ai2_eye_color = (50, 50, 250)

        self.ai1_position = (
            random.randint(
                self.ai1_initial_range_x[0], self.ai1_initial_range_x[1]
            ),
            random.randint(
                self.ai1_initial_range_y[0], self.ai1_initial_range_y[1]
            ),
        )
        self.ai2_position = (
            random.randint(self.ai2_initial_range_x[0], self.ai2_initial_range_x[1]),
            random.randint(self.ai2_initial_range_y[0], self.ai2_initial_range_y[1]),
        )
        # self.ai1_position = tuple((i // 3) * 3 for i in self.ai1_position)
        # self.ai2_position = tuple((i // 3) * 3 for i in self.ai2_position)

        self.ai1_direction = 0  # Initial direction for ai1
        self.ai2_direction = 0  # Initial direction for ai2

        self.display()

        return self.observation

    def display(self):
        surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        surface.fill(self.background_color)

        ai1_x = self.ai1_position[0]
        ai1_y = self.ai1_position[1]
        ai2_x = self.ai2_position[0]
        ai2_y = self.ai2_position[1]

        # Draw bullets
        for bullet_x, bullet_y, _ in self.ai1_bullets:
            pygame.draw.rect(
                surface,
                self.ai1_bullet_color,
                (bullet_x, bullet_y, self.bullet_size, self.bullet_size),
            )
        for bullet_x, bullet_y, _ in self.ai2_bullets:
            pygame.draw.rect(
                surface,
                self.ai2_bullet_color,
                (bullet_x, bullet_y, self.bullet_size, self.bullet_size),
            )

        # Draw objects
        pygame.draw.rect(
            surface,
            self.ai1_color,
            (ai1_x, ai1_y, self.ai_size, self.ai_size),
        )
        pygame.draw.rect(
            surface, self.ai2_color, (ai2_x, ai2_y, self.ai_size, self.ai_size)
        )

        # Draw direction indicator pixel
        ai1_direction_pixel = self.get_direction_pixel(self.ai1_direction)
        ai2_direction_pixel = self.get_direction_pixel(self.ai2_direction)
        surface.set_at(
            (
                ai1_x + ai1_direction_pixel[0],
                ai1_y + ai1_direction_pixel[1],
            ),
            self.ai1_eye_color,
        )
        surface.set_at(
            (ai2_x + ai2_direction_pixel[0], ai2_y + ai2_direction_pixel[1]),
            self.ai2_eye_color,
        )

        self.observation = pygame.surfarray.array3d(surface)

        scaled_surface = pygame.transform.scale(
            surface, (self.WIDTH * self.SCALE, self.HEIGHT * self.SCALE)
        )

        self.screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()

    # def move_p(self):
    #     player_speed = 1  # Modify as needed
    #     player_dx = (
    #         self.pressed_keys[pygame.K_d] - self.pressed_keys[pygame.K_a]
    #     ) * player_speed
    #     player_dy = (
    #         self.pressed_keys[pygame.K_s] - self.pressed_keys[pygame.K_w]
    #     ) * player_speed

    #     if player_dx > 0:
    #         self.player_direction = 1  # Right
    #     elif player_dx < 0:
    #         self.player_direction = 3  # Left
    #     elif player_dy > 0:
    #         self.player_direction = 2  # Down
    #     elif player_dy < 0:
    #         self.player_direction = 0  # Up

    #     return (
    #         self.player_position[0] + player_dx,
    #         self.player_position[1] + player_dy,
    #     )

    def move_ai(self, action, ai_number=1):
        ai_speed = 1  # Modify as needed
        ai_dx = (action[0] - action[1]) * ai_speed
        ai_dy = (action[2] - action[3]) * ai_speed

        if ai_dx > 0:
            if ai_number == 1:
                self.ai1_direction = 1  # Right
            elif ai_number == 2:
                self.ai2_direction = 1
                
        elif ai_dx < 0:
            if ai_number == 1:
                self.ai1_direction = 3  # Left
            elif ai_number == 2:
                self.ai2_direction = 3

        elif ai_dy > 0:
            
            if ai_number == 1:
                self.ai1_direction = 2  # DOwn
            elif ai_number == 2:
                self.ai2_direction = 2

        elif ai_dy < 0:
            
            if ai_number == 1:
                self.ai1_direction = 0  # Up
            elif ai_number == 2:
                self.ai2_direction = 0

        if ai_number == 1:
            return (self.ai1_position[0] + ai_dx, self.ai1_position[1] + ai_dy)
        
        elif ai_number == 2:
            return (self.ai2_position[0] + ai_dx, self.ai2_position[1] + ai_dy)

    def move_objects(self, action1, action2):
        new_ai1_position = self.move_ai(action1, ai_number=1)
        if self.within_boundary(new_ai1_position, self.ai_size):
            self.ai1_position = new_ai1_position

        new_ai2_position = self.move_ai(action2, ai_number=2)
        if self.within_boundary(new_ai2_position, self.ai_size):
            self.ai2_position = new_ai2_position
        #     self.ai2_at_wall = 0
        # else:
        #     self.ai2_at_wall = 1

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

    # def player_shoot_bullet(self):
    #     if self.player_bullet_cooldown == 0:
    #         direction = self.get_direction_pixel(self.player_direction)
    #         bullet_x = (
    #             self.player_position[0]
    #             + self.player_size // 2
    #             - self.bullet_size // 2
    #             + (direction[0] - 1) * 1
    #         )
    #         bullet_y = (
    #             self.player_position[1]
    #             + self.player_size // 2
    #             - self.bullet_size // 2
    #             + (direction[1] - 1) * 1
    #         )
    #         bullet_direction = self.player_direction

    #         self.player_bullets.append((bullet_x, bullet_y, bullet_direction))
    #         self.player_bullet_cooldown = self.player_bullet_cooldown_time


    def ai1_shoot_bullet(self):
        if self.ai1_bullet_cooldown == 0:
            direction = self.get_direction_pixel(self.ai1_direction)
            bullet_x = (
                self.ai1_position[0]
                + self.ai_size // 2
                - self.bullet_size // 2
                + (direction[0] - 1) * 1
            )
            bullet_y = (
                self.ai1_position[1]
                + self.ai_size // 2
                - self.bullet_size // 2
                + (direction[1] - 1) * 1
            )
            bullet_direction = self.ai1_direction

            self.ai1_bullets.append((bullet_x, bullet_y, bullet_direction))
            self.ai1_bullet_cooldown = self.bullet_cooldown_time

            dis_x = (-abs(self.ai1_position[0] - self.ai2_position[0]) / 64) + 1
            dis_y = (-abs(self.ai1_position[1] - self.ai2_position[1]) / 64) + 1

            dis_matrix = np.array([dis_y, dis_y, dis_x, dis_x])
            direction_matrix = np.array([0, 0, 0, 0])
            direction_matrix[self.ai1_direction] = 1

            self.ai1_reward += sum(dis_matrix * direction_matrix) * 0.01


    def ai2_shoot_bullet(self):
        if self.ai2_bullet_cooldown == 0:
            direction = self.get_direction_pixel(self.ai2_direction)
            bullet_x = (
                self.ai2_position[0]
                + self.ai_size // 2
                - self.bullet_size // 2
                + (direction[0] - 1) * 1
            )
            bullet_y = (
                self.ai2_position[1]
                + self.ai_size // 2
                - self.bullet_size // 2
                + (direction[1] - 1) * 1
            )
            bullet_direction = self.ai2_direction

            self.ai2_bullets.append((bullet_x, bullet_y, bullet_direction))
            self.ai2_bullet_cooldown = self.bullet_cooldown_time

            dis_x = (-abs(self.ai2_position[0] - self.ai1_position[0]) / 64) + 1
            dis_y = (-abs(self.ai2_position[1] - self.ai1_position[1]) / 64) + 1

            dis_matrix = np.array([dis_y, dis_y, dis_x, dis_x])
            direction_matrix = np.array([0, 0, 0, 0])
            direction_matrix[self.ai2_direction] = 1

            self.ai2_reward += sum(dis_matrix * direction_matrix) * 0.01

    def check_bullet_collisions(self):

        # Check player bullet collisions
        ai1_hit_indices = []
        for i, bullet in enumerate(self.ai1_bullets):
            bullet_rect = pygame.Rect(
                bullet[0], bullet[1], self.bullet_size, self.bullet_size
            )
            if bullet_rect.colliderect(
                pygame.Rect(*self.ai1_position, self.ai_size, self.ai_size)
            ):
                ai1_hit_indices.append(i)
                self.ai1_reward -= 1
                self.ai2_reward += 2
                # print("ai1_hit reward:", self.ai1_reward)
                self.done = 1

        # Remove collided ai1 bullets
        for index in ai1_hit_indices:
            del self.ai1_bullets[index]

        # Check AI bullet collisions
        ai2_hit_indices = []
        for i, bullet in enumerate(self.ai2_bullets):
            bullet_rect = pygame.Rect(
                bullet[0], bullet[1], self.bullet_size, self.bullet_size
            )
            if bullet_rect.colliderect(
                pygame.Rect(*self.ai1_position, self.ai_size, self.ai_size)
            ):
                ai2_hit_indices.append(i)
                self.ai2_reward += 2
                self.ai1_reward -= 1
                # print("ai2_hit reward:", self.ai2_reward)
                self.done = 1

        # Remove collided AI bullets
        for index in ai2_hit_indices:
            del self.ai2_bullets[index]

        return self.ai1_reward, self.ai2_reward

    def move_bullets(self):
        bullet_speed = 3  # Modify as needed

        # Move ai1 bullets
        new_ai1_bullets = []
        for i in range(len(self.ai1_bullets)):
            bullet_x, bullet_y, bullet_direction = self.ai1_bullets[i]
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
                new_ai1_bullets.append((bullet_x, bullet_y, bullet_direction))
        self.ai1_bullets = new_ai1_bullets

        # Move ai2 bullets
        new_ai2_bullets = []
        for i in range(len(self.ai2_bullets)):
            bullet_x, bullet_y, bullet_direction = self.ai2_bullets[i]
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
                new_ai2_bullets.append((bullet_x, bullet_y, bullet_direction))
        self.ai2_bullets = new_ai2_bullets

    def step(self, input1, input2):
        self.ai1_reward = 0
        self.ai2_reward = 0

        action1 = np.zeros(5, int)
        action1[input1] = 1

        action2 = np.zeros(5, int)
        action2[input2] = 1

        # self.clock.tick(self.tick_rate)
        
        # Decrement bullet cooldowns
        if self.ai1_bullet_cooldown > 0:
            self.ai1_bullet_cooldown -= 1
            self.ai1_eye_color = (
                50,
                50,
                250
                - 200
                * (self.ai1_bullet_cooldown / self.bullet_cooldown_time),
            )
        if self.ai2_bullet_cooldown > 0:
            self.ai2_bullet_cooldown -= 1
            self.ai2_eye_color = (
                50,
                50,
                250 - 200 * (self.ai2_bullet_cooldown / self.bullet_cooldown_time),
            )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.player_shoot_bullet()

        self.pressed_keys = pygame.key.get_pressed()

        if action1[4] == 1:
            self.ai1_shoot_bullet()

        if action2[4] == 1:
            self.ai2_shoot_bullet()

        self.move_objects(action1, action2)

        self.move_bullets() 
        
        self.check_bullet_collisions()

        self.display()

        return self.observation, (self.ai1_reward, self.ai2_reward), self.done, None


if __name__ == '__main__':

    shooting_game = ShootingGame()
    while True:
        shooting_game.reset()

        while not shooting_game.done:
            shooting_game.step(ai_input=1)

