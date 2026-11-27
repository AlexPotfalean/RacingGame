"""Train with PPO, save the model, then watch it drive."""
import sys
from stable_baselines3 import PPO
from racing import RaceEnv

if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    model = PPO("MlpPolicy", RaceEnv(), verbose=1)
    model.learn(total_timesteps=steps)
    model.save("car_oval")
    print("Saved car_oval.zip - run watch.py to see it drive")
