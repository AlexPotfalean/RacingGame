"""Watch the trained car drive (close the window to quit)."""
from stable_baselines3 import PPO
from racing import RaceEnv

env = RaceEnv(render_mode="human")
model = PPO.load("car_oval")
obs, _ = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, r, done, trunc, _ = env.step(action)
    if done or trunc:
        obs, _ = env.reset()
