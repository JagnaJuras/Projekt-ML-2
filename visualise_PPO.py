import gymnasium as gym
from stable_baselines3 import PPO
import imageio

# Load the environment
env = gym.make("BipedalWalker-v3", render_mode="rgb_array")

# Load the trained model
model = PPO.load("ppo_bipedalwalker_v1")

# Create a list to store frames
frames = []

# Evaluate and visualize the agent
episodes = 1
for episode in range(episodes):
    obs, _ = env.reset()  # Unpack the observation if reset returns a tuple
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)  # Unpack if step returns a tuple
        frame = env.render()
        frames.append(frame)

env.close()

# Save the frames as a gif
imageio.mimsave('bipedal_PPO_v1.gif', frames, fps=24)
