# Import needed stuff
import gymnasium as gym
from stable_baselines3 import DQN # DQN is the RL algorithm used in this project
from stable_baselines3.common.vec_env import DummyVecEnv  # Use DummyVecEnv for a single environment
import os # For checking filepaths for saving agents

#######################################################################
# Class for custom cartpole environment                               #
# This is needed to add a custom reward for being close to the middle #
#######################################################################

class CustomCartPoleEnv(gym.Env):
    def __init__(self):
        super(CustomCartPoleEnv, self).__init__()
        self.env = gym.make("CartPole-v1")  # create the gym environment
        self.action_space = self.env.action_space # set action space 
        self.observation_space = self.env.observation_space # set observation space

    def reset(self, seed=None, options=None): # add reset function for the env
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action): # timestep function, defines what to do each "frame"

        observation, reward, terminated, truncated, info = self.env.step(action) # return values for each timestep

        # Custom reward / penalty, incentivizes staying towards the middle
        cart_position = observation[0]  # The cart's position
        penalty = abs(cart_position) * -1  # Negative penalty proportional to the distance from center
        reward += penalty  # Apply the penalty to the reward

        return observation, reward, terminated, truncated, info  # Return all five values

    def render(self, mode='human'): # function for renderng (that also sets the render mode to a visible pygame window)
        self.env.render()

    def close(self): # function for closing the env
        self.env.close()

# Create a DummyVecEnv with the custom environment
env = DummyVecEnv([lambda: CustomCartPoleEnv()])

# Defining the path for saving and loading models
model_path = "dqn_cartpole.zip"

# Simple CLI for different options
print("Welcome to Cartpole RL. Choose an option: ")
print("1. Load and run")
print("2. Load and train")
print("3. Train new agent")
choice = input("")

# Choice 1 loads and runs an existing model
if choice == "1":
    # Check if the model already exists
    if os.path.exists(model_path):
        # Load the existing model
        model = DQN.load(model_path, env=env)
        print("Model loaded!")
    else:
        print("No model found. Please train the model first.")
        exit()

# Choice 2 loads an existing model and continues training it
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

# Choie 3 creates a new model and begins training it
elif choice == "3":
    confirmation = input("Are you sure you wish to create a new agent? (yes/no): ")
    if confirmation.lower() == "yes":
        timesteps = int(input("Enter the interval of timesteps between saves (100k is avg): "))
        # If model doesn't exist, train a new one
        model = DQN("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=timesteps)  # You can adjust the number of timesteps
        model.save(model_path)
        print("Model trained and saved!")
else:
    exit()

# Close the environment after training
env.close()

##################################################################################
# The below code creates a new environment with the model that was just trained. #
##################################################################################

# Create a normal environment with human render mode so its a pygame window
render_env = gym.make("CartPole-v1", render_mode="human")  # To see the environment

# reset the environment after training
obs, _ = render_env.reset()  # Gym API returns obs, info


# Play the game with the model
for _ in range(1000):
    # Predict the action using the trained model
    action, _states = model.predict(obs, deterministic=True)

    # Take a step in the environment
    obs, reward, terminated, truncated, info = render_env.step(action)  # Update to handle new return values

    # Render the environment
    render_env.render()

    # Reset the environment if an episode ends
    if terminated:
        obs, _ = render_env.reset() 

# Finally close the environment
render_env.close()
