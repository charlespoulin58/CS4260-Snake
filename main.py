import gymnasium as gym
import gym_snake
from gym_snake.envs.snake_env import SnakeEnv

gym.register(
    id='snake-v0',
    entry_point='gym_snake.envs.snake_env:SnakeEnv',
)


# Construct Environment
env = gym.make('snake-v0')
observation = env.reset() # Constructs an instance of the game

# Controller
game_controller = env.unwrapped.controller

# Grid
grid_object = game_controller.grid
grid_pixels = grid_object.grid

# Snake(s)
snakes_array = game_controller.snakes
snake_object1 = snakes_array[0]