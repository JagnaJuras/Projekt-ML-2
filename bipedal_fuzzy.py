import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import gymnasium as gym
import pickle

# Define fuzzy variables and rules
angle = ctrl.Antecedent(np.arange(-1, 1, 0.01), 'angle')
action = ctrl.Consequent(np.arange(-1, 1, 0.01), 'action')

angle['left'] = fuzz.trimf(angle.universe, [-1, -1, 0])
angle['center'] = fuzz.trimf(angle.universe, [-1, 0, 1])
angle['right'] = fuzz.trimf(angle.universe, [0, 1, 1])

action['left'] = fuzz.trimf(action.universe, [-1, -1, 0])
action['none'] = fuzz.trimf(action.universe, [-1, 0, 1])
action['right'] = fuzz.trimf(action.universe, [0, 1, 1])

rule1 = ctrl.Rule(angle['left'], action['right'])
rule2 = ctrl.Rule(angle['center'], action['none'])
rule3 = ctrl.Rule(angle['right'], action['left'])

control_system = ctrl.ControlSystem([rule1, rule2, rule3])
sim = ctrl.ControlSystemSimulation(control_system)

# Save the fuzzy controller and simulation
with open('fuzzy_controller.pkl', 'wb') as f:
    pickle.dump((control_system, sim), f)

env = gym.make("BipedalWalker-v3")

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
