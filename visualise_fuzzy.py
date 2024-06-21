import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import gymnasium as gym
import pickle

# Load the fuzzy controller and simulation
with open('fuzzy_controller.pkl', 'rb') as f:
    control_system, sim = pickle.load(f)

env = gym.make("BipedalWalker-v3", render_mode="human")

def fuzzy_controller(obs):
    # Unpack the observation tuple if necessary
    if isinstance(obs, tuple):
        obs = obs[0]

    hull_angle = obs[2]
    sim.input['angle'] = hull_angle
    sim.compute()

    # Get the action value from the simulation
    action_value = sim.output['action']

    # Return a properly formatted action array
    action = np.array([action_value, 0, action_value, 0])
    return action

episodes = 5
for episode in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action = fuzzy_controller(obs)
        obs, reward, done, _, _ = env.step(action)
        env.render()
env.close()
