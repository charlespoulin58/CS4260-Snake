# from gordongames.envs.ggames.constants import *
import gymnasium as gym
import gym_snake
from gym_snake.envs.snake_env import SnakeEnv

gym.register(
    id='snake-v0',
    entry_point='gym_snake.envs.snake_env:SnakeEnv',
)

if __name__ == "__main__":
    print("PRESS q to quit")
    print("wasd to move, f to press")
    env = gym.make('snake-v0')
    obs, info = env.reset(), {}
    key = ""
    action = 0
    while key != "q":
        env.render()
        key = input("action: ")
        if key   == "w": action = 0
        elif key == "d": action = 1
        elif key == "s": action = 2
        elif key == "a": action = 3
        else: pass
        obs, reward, terminated, truncated, info = env.step(action)
        print("reward:", reward)
        print("terminated:", terminated)
        print("truncated:", truncated)
        print("info:", info)
        if terminated or truncated:
            obs, info = env.reset(), {}
