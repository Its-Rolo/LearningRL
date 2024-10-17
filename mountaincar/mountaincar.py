import gymnasium as gym
from stable_baselines3 import DQN
import os
import time
import numpy as np

model_path = "dqn_mountaincar_v2.zip"

# CLI for different options
print("Welcome to MountainCar RL. Choose an option: ")
print("1. Load and run")
print("2. Load and train")
print("3. Train new agent")
choice = input("")

totalTimesteps = 0

# Create the environment with custom reward adjustments
class RewardShapingMountainCar(gym.Env):
    def __init__(self):
        super(RewardShapingMountainCar, self).__init__()
        self.env = gym.make("MountainCar-v0")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Custom reward shaping: add a small negative reward for every step to encourage faster solutions
        if not terminated and not truncated:
            reward = reward - 0.1

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        self.env.close()

# Create the environment
env = RewardShapingMountainCar()

# Choice 1 loads and runs an existing model
if choice == "1":
    if os.path.exists(model_path):
        model = DQN.load(model_path, env=env)
        print("Model loaded!")
    else:
        print("No model found. Please train the model first.")
        exit()

# Choice 2 loads an existing model and continues training it
elif choice == "2":
    if os.path.exists(model_path):
        model = DQN.load(model_path, env=env)
        print("Model loaded!")
        timesteps = int(input("Enter the amount of timesteps per cycle (10k is small, 100k is average, 500k is a lot): "))

        try:
            while True:
                model.learn(total_timesteps=timesteps)  # Train the model in cycles
                totalTimesteps += timesteps
                model.save(model_path)  # Save the updated model
                print(f"Model trained and saved! Total timesteps: {totalTimesteps}")
                time.sleep(1)  # Optional delay to reduce console output
        except KeyboardInterrupt:
            print("Training interrupted. Saving model...")
            model.save(model_path)  # Save the model before exiting
            print("Model saved!")
            exit()

    else:
        print("No model found. Please train the model first.")
        exit()

# Choice 3 creates a new model and begins training it
elif choice == "3":
    timesteps = int(input("Enter the amount of timesteps per cycle (10k is small, 100k is average, 500k is a lot): "))
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,  # Lower learning rate
        buffer_size=50000,
        batch_size=64,  # Larger batch size
        tau=0.005,
        gamma=0.999,  # Higher gamma
        train_freq=4,
        exploration_fraction=0.2,  # Higher exploration rate
        target_update_interval=500,
        verbose=1
    )

    try:
        while True:
            model.learn(total_timesteps=timesteps)  # Train the model in cycles
            totalTimesteps += timesteps
            model.save(model_path)  # Save the newly trained model
            print(f"Model trained and saved! Total timesteps: {totalTimesteps}")
            time.sleep(1)  # Optional delay to reduce console output
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        model.save(model_path)  # Save the newly trained model
        print("Model saved!")
        exit()

else:
    print("Invalid choice. Exiting.")
    exit()

# Run the trained model in the environment
obs, _ = env.reset()  # Reset the environment to generate the first observation
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)  # Use the model to predict the action
    obs, reward, terminated, truncated, info = env.step(action)  # Step through the environment
    env.render()
    
    if terminated or truncated:
        obs, _ = env.reset()

env.close()  # Close the environment
