import gymnasium as gym
import gym_snake
from gym_snake.envs.snake_env import SnakeEnv
from agents.down_agent import DownAgent

gym.register(
    id='snake-v0',
    entry_point='gym_snake.envs.snake_env:SnakeEnv',
)

if __name__ == "__main__":
    env = gym.make('snake-v0')
    agent = DownAgent()
    obs, info = env.reset(), {}
    episode = 1
    total_reward = 0
    steps = 0
    while True:
        env.render()
        action = agent.select_action(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        print(f"Episode: {episode}, Step: {steps}, Reward: {reward}")
        if terminated or truncated:
            print(f"Game Over! Final stats for episode {episode}:")
            print(f"Total steps: {steps}")
            print(f"Total reward: {total_reward}")
            break
    env.close()
