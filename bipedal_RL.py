import gymnasium as gym
from stable_baselines3 import PPO
import torch

# Check if a GPU is available and if PyTorch can use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = gym.make("BipedalWalker-v3")

# Create the RL agent using PPO, and specify the device
model = PPO("MlpPolicy", env, verbose=1, device=device)

# Train the agent
model.learn(total_timesteps=1_000_000)

# Save the model
model.save("ppo_bipedalwalker_v2")

# Load the model (optional)s
# model = PPO.load("ppo_bipedalwalker_v2")

env.close()
