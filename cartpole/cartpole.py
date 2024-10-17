import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv  # Use DummyVecEnv for a single environment
import os

class CustomCartPoleEnv(gym.Env):
    def __init__(self):
        super(CustomCartPoleEnv, self).__init__()
        self.env = gym.make("CartPole-v1")  # Use Gymnasium's CartPole environment
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        # Adjust this to unpack the correct number of returned values
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Penalize for going away from the center
        cart_position = observation[0]  # The cart's position
        penalty = abs(cart_position) * -1  # Negative penalty proportional to the distance from center
        reward += penalty  # Apply the penalty to the reward

        return observation, reward, terminated, truncated, info  # Return all five values

    def render(self, mode='human'):
        self.env.render()

    def close(self):
        self.env.close()

# Create a DummyVecEnv with your custom environment
env = DummyVecEnv([lambda: CustomCartPoleEnv()])  # Wrap the custom environment

# Define the path to save/load the model
model_path = "dqn_cartpole.zip"

print("Welcome to Cartpole RL. Choose an option: ")
print("1. Load and run")
print("2. Load and train")
print("3. Train new agent")
choice = input("")

if choice == "1":
    # Check if the model already exists
    if os.path.exists(model_path):
        # Load the existing model
        model = DQN.load(model_path, env=env)
        print("Model loaded!")
    else:
        print("No model found. Please train the model first.")
        exit()
elif choice == "2":
    timesteps = int(input("Enter the amount of timesteps (10k is small, 100k is average, 500k is a lot): "))
    if os.path.exists(model_path):
        # Load the existing model
        model = DQN.load(model_path, env=env)
        print("Model loaded!")
        model.learn(total_timesteps=timesteps)  # You can adjust the number of timesteps
        model.save(model_path)
        print("Model trained and saved!")
    else:
        print("No model found. Please train the model first.")
        exit()
elif choice == "3":
    timesteps = int(input("Enter the amount of timesteps (10k is small, 100k is average, 500k is a lot): "))
    # If model doesn't exist, train a new one
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)  # You can adjust the number of timesteps
    model.save(model_path)
    print("Model trained and saved!")
else:
    exit()

# Step 3: Close the training environment
env.close()

# Step 4: Create a regular environment for rendering (for visualizing the results)
render_env = gym.make("CartPole-v1", render_mode="human")  # To see the environment

# Step 5: Reset the environment for evaluation (after training)
obs, _ = render_env.reset()  # Gym API returns obs, info

for _ in range(1000):
    # Predict the action using the trained model
    action, _states = model.predict(obs, deterministic=True)

    # Take a step in the environment
    obs, reward, terminated, truncated, info = render_env.step(action)  # Update to handle new return values

    # Render the environment (this will show the visual output)
    render_env.render()

    # Reset the environment if an episode ends
    if terminated:
        obs, _ = render_env.reset()  # Make sure to handle the tuple response correctly

# Step 6: Close the environment after rendering
render_env.close()
