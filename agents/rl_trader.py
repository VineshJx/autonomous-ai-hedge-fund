import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from agents.trading_env import TradingEnv

MODEL_PATH = "models/ppo_trading_model"


def make_env(data):
    return lambda: TradingEnv(data)


def train_agent(data, timesteps=20000):

    env = DummyVecEnv([make_env(data)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64
    )

    print("🚀 Training RL Agent...")
    model.learn(total_timesteps=timesteps)

    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)

    print("✅ Model saved")

    return model


def load_agent(path=MODEL_PATH, data=None):
    env = DummyVecEnv([make_env(data)])
    model = PPO.load(path, env=env)
    return model