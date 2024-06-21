import gymnasium as gym

env = gym.make("BipedalWalker-v3", render_mode="human")

def simple_controller(obs):
    # A very basic controller that tries to balance the walker
    # obs[2] is the hull angle, obs[4] and obs[6] are the joint angles
    angle = obs[2]
    if angle < 0:
        action = 1  # try to push to the right
    else:
        action = -1  # try to push to the left
    return action

episodes = 5
for episode in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action = simple_controller(obs)
        obs, reward, done, info, _ = env.step(action)
        env.render()
env.close()
