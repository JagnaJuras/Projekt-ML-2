import gymnasium as gym
import imageio

# Load the environment
env = gym.make("BipedalWalker-v3", render_mode="rgb_array")

def simple_controller(obs):
    # A very basic controller that tries to balance the walker
    # obs[2] is the hull angle, obs[4] and obs[6] are the joint angles
    angle = obs[2]
    if angle < 0:
        return [1, 0, 0, 0]  # try to push to the right
    else:
        return [0, 0, 0, 1]  # try to push to the left

# Create a list to store frames
frames = []

episodes = 5
for episode in range(episodes):
    obs, _ = env.reset()  # Unpack the observation if reset returns a tuple
    done = False
    while not done:
        action = simple_controller(obs)
        obs, reward, done, _, _ = env.step(action)  # Unpack if step returns a tuple
        frame = env.render()
        frames.append(frame)

env.close()

# Save the frames as a gif
imageio.mimsave('bipedal_simple.gif', frames, fps=24)
