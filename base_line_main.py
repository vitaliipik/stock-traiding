import numpy as np
import torch
# from gym.vector.utils import spaces
from gym_anytrading.envs import StocksEnv
import gymnasium as gym
import gym_trading_env
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common import callbacks
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dqn import MlpPolicy

# from agent import Agent
# from rat.test_env import StockTradingEnv
import yfinance as yf

class Config:
    def __init__(self):
        self.gamma=0.95
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.episodes=200
        self.epsilon_start = 0.90
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.epsilon= 0.9
        self.learning_rate = 0.001
        self.memory_capacity = 1000
        self.replay_memory_size = 1000


def evaluate(model, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward


def main():
    config = Config()
    models_dir=r"D:\all projects\stock-traiding\models"


    start="2015-01-01"
    end="2021-01-01"
    # data = yf.download("AAPL",start= start,end=end, interval="1h")
    data = yf.download("AAPL",period="730d", interval="1h")
    print(len(data))
    env = StocksEnv(df=data, window_size=30, frame_bound=(30, len(data)))
    data.columns=[i.lower() for i in data.columns]

    env=gym.make("TradingEnv",
             name="Apple",
             df=data,  # Your dataset with your custom features
             positions=[-1, 0, 1],  # -1 (=SHORT), 0(=OUT), +1 (=LONG)
             trading_fees=0.18 / 100,  # 0.01% per stock buy / sell (Binance fees)
             borrow_interest_rate=0.00001 / 100,  # 0.0003% per timestep (one timestep = 1h here)
             # borrow_interest_rate=0,  # 0.0003% per timestep (one timestep = 1h here)
             )
    # # env.save_rendering(r"D:\all projects\stock-traiding\render_logs")
    # env = DummyVecEnv([lambda: env])
    batch_size=32
    # Example usage:
    # Initialize an agent
    # Initialize DQN model
    print("env creatre")
    train=True
    model = DQN(MlpPolicy, env, verbose=1,
                exploration_fraction=0.5,
                exploration_final_eps=0.1,
                tensorboard_log=r"D:\all projects\stock-traiding\tensor_log")
    # path=r"D:\all projects\stock-traiding\models_backup\back_gum_0_18%\rl_model_10000000_steps.zip"
    # model = DQN.load(path,env=env)
    TIMESTEPS = 10000000
    checkpoint_callback=CheckpointCallback(save_freq=1e5,save_path=models_dir)
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False,callback=[checkpoint_callback])
    epoches=0
    epoches_end=1000
    # if train:
    #
    #     iters = 0
    #     for i in range(epoches,epoches_end):
    #         model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    #
    #         model.save(f"{models_dir}/{TIMESTEPS * i}")
    #         print(f"_____________{epoches}_____________")
    model.save(f"{models_dir}/{TIMESTEPS}-3")

        # Save the model
        # model.save("dqn_stock_trading")

    # Load the model
    # loaded_model = DQN.load("dqn_stock_trading")

    # mean_reward_before_train = evaluate(model, num_episodes=100)
    # print(mean_reward_before_train)
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    #
    # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    #
    # mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=100)
    #
    # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    # episodes = 10
    # # Example of using the trained model for prediction
    # for ep in range(episodes):
    #     obs = env.reset()
    #     done = False
    #     while not done:
    #         action, _states = loaded_model.predict(obs)
    #         obs, rewards, done, info = env.step(action)
    #         env.render()
    #         print(rewards)
if __name__ == '__main__':
    main()